#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vetolib Web Data Agent — Berlin (DE)
- Overpass mirrors + retries + tiling to avoid throttling
- Optional Nominatim reverse geocode to backfill address bits
- Heavy opening-hours parsing:
    * prefer OSM opening_hours tag
    * else fetch clinic website (robots-aware) and parse German/English hours
    * normalize to OSM OpeningHours (e.g. "Mo-Fr 09:00-18:00; Sa 10:00-14:00; Su off" or "24/7")
- Maps and exports EXACTLY to the Supabase schema like clinics_rows.csv (lng, not lon)

Outputs:
  out/clinics_berlin.csv
  out/clinics_berlin.jsonl
"""

import os
import re
import csv
import json
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple

import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# ----------------------------
# Config / constants
# ----------------------------
CITY_DEFAULT = "Berlin"
STATE_DEFAULT = "Berlin"
COUNTRY_DEFAULT = "DE"

# Berlin bbox (minlon, minlat, maxlon, maxlat)
BERLIN_BBOX = (13.08835, 52.33826, 13.76116, 52.67551)

# Overpass mirrors (ordered)
OVERPASS_URLS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

NOMINATIM_URL = "https://nominatim.openstreetmap.org"
HEADERS = {"User-Agent": "Vetolib-Agent/1.2 (+https://vetolib.app)"}
EMAIL = os.getenv("NOMINATIM_EMAIL", "you@example.com")

# --- Opening-hours parsing flags ---
# We rely on opentimeparser if available; otherwise we fall back to simple heuristics.
TRY_OPENTIMEPARSER = True

# ----------------------------
# Supabase schema columns (aligned to clinics_rows.csv)
# NOTE: If your live table differs, adjust names/order here.
# ----------------------------
SB_COLUMNS = [
    # identity + contact
    "name","website","phone","email",
    # address + geo
    "street","housenumber","postcode","city","lat","lng",
    # hours + flags used by your app
    "opening_hours","emergency","emergency_boolean","twentyfour_seven",
    "supports_mobile","booking_enabled","active","active_boolean",
    # optional business flags in schema (default false)
    "offers_checkup","offers_dental","offers_illness","offers_prescription","offers_vaccination",
    # placeholders in schema (string)
    "description","district","onboarding_status",
    "created_at","updated_at",
    # economics / notes (string/number-ish; keep as string if uncertain)
    "got_min_multiplier","got_max_multiplier","got_notes",
    "weekend_min_multiplier","weekend_policy_notes",
    "notdienst_fee_eur","travel_cost_min_eur","travel_cost_per_double_km_eur",
    # ids/invites
    "id","auth_user_id","invite_sent_at","last_login_at",
]

# ----------------------------
# Helpers (general)
# ----------------------------

def within_bbox(lat: float, lon: float, bbox: Tuple[float,float,float,float]) -> bool:
    minlon, minlat, maxlon, maxlat = bbox
    return (minlat <= lat <= maxlat) and (minlon <= lon <= maxlon)

def norm_url(url: str) -> str:
    if not url: return ""
    u = url.strip()
    if u.startswith("//"): u = "https:" + u
    if not re.match(r"^https?://", u): u = "https://" + u
    return u.rstrip("/")

def norm_phone(phone: str) -> str:
    # Keep digits and leading +; short and safe
    return re.sub(r"[^\d+]", "", phone or "")[:25]

def http_json_with_retries(method, url, *, data=None, params=None, headers=None, timeout=90, tries=4, polite_delay=1.0):
    delay = polite_delay
    last_err = None
    for i in range(tries):
        try:
            resp = requests.request(method, url, data=data, params=params, headers=headers, timeout=timeout)
            ctype = resp.headers.get("content-type", "")
            if resp.status_code in (429, 502, 503, 504) or "json" not in ctype.lower():
                last_err = Exception(f"{resp.status_code} {ctype} (attempt {i+1}/{tries})")
                time.sleep(delay); delay = min(delay * 2, 20)
                continue
            return resp.json()
        except Exception as e:
            last_err = e
            time.sleep(delay); delay = min(delay * 2, 20)
    raise last_err

def tile_bbox(bbox: Tuple[float,float,float,float], tiles_per_side: int) -> List[Tuple[float,float,float,float]]:
    """Split bbox into N x N tiles."""
    minlon, minlat, maxlon, maxlat = bbox
    lons = [minlon + i*(maxlon-minlon)/tiles_per_side for i in range(tiles_per_side)] + [maxlon]
    lats = [minlat + i*(maxlat-minlat)/tiles_per_side for i in range(tiles_per_side)] + [maxlat]
    tiles = []
    for i in range(tiles_per_side):
        for j in range(tiles_per_side):
            tiles.append((lons[i], lats[j], lons[i+1], lats[j+1]))
    return tiles

# ----------------------------
# Opening hours parsing / extraction
# ----------------------------

OH_DAY_MAP = {
    # German
    "mo":"Mo","montag":"Mo",
    "di":"Tu","dienstag":"Tu",
    "mi":"We","mittwoch":"We",
    "do":"Th","donnerstag":"Th",
    "fr":"Fr","freitag":"Fr",
    "sa":"Sa","samstag":"Sa",
    "so":"Su","sonntag":"Su",
    # English
    "mon":"Mo","monday":"Mo",
    "tue":"Tu","tues":"Tu","tuesday":"Tu",
    "wed":"We","wednesday":"We",
    "thu":"Th","thur":"Th","thurs":"Th","thursday":"Th",
    "fri":"Fr","friday":"Fr",
    "sat":"Sa","saturday":"Sa",
    "sun":"Su","sunday":"Su",
}

def looks_247(s: str) -> bool:
    s = (s or "").lower()
    return any(k in s for k in ["24/7","24-7","24h","24 h","24 stunden","rund um die uhr"])

EMERGENCY_KWS = ["notdienst","emergency","24/7","24 stunden","24h","24 h"]

def derive_emergency(opening_hours: str, name: str, extra_text: str = "") -> bool:
    blob = " ".join([opening_hours or "", name or "", extra_text or ""]).lower()
    return any(kw in blob for kw in EMERGENCY_KWS)

def group_days(day_list):
    order = ["Mo","Tu","We","Th","Fr","Sa","Su"]
    idx = {d:i for i,d in enumerate(order)}
    day_list = [d for d in day_list if d in idx]
    day_list = sorted(set(day_list), key=lambda d: idx[d])
    ranges = []
    start = prev = None
    for d in day_list:
        if start is None:
            start = prev = d
            continue
        if idx[d] == idx[prev] + 1:
            prev = d
        else:
            ranges.append((start, prev))
            start = prev = d
    if start is not None:
        ranges.append((start, prev))
    out = []
    for a,b in ranges:
        out.append(a if a==b else f"{a}-{b}")
    return out

def normalize_osm_oh(segments):
    """Segments: list of (days_str, time_str) -> 'Mo-Fr 09:00-18:00; Sa 10:00-14:00; Su off'"""
    parts = []
    for days, times in segments:
        days = str(days).strip()
        times = str(times).strip()
        if not days or not times: 
            continue
        parts.append(f"{days} {times}")
    return "; ".join(parts).strip("; ").strip()

def parse_hours_with_opentimeparser(raw_text: str):
    try:
        from opentimeparser import parse as otp_parse
    except Exception:
        return []
    try:
        parsed = otp_parse(raw_text or "")
        # parsed like: [{'days': ['Mo','Tu',...], 'hours': [{'from':'09:00','to':'18:00'}, ...]}]
        segments = []
        for block in parsed:
            days = block.get("days") or []
            day_ranges = group_days(days) if days else ["Mo-Su"]
            hours = block.get("hours") or []
            if not hours:
                for dr in day_ranges:
                    segments.append((dr, "off"))
                continue
            intervals = []
            for h in hours:
                f = (h.get("from") or "").replace(".",":")
                t = (h.get("to") or "").replace(".",":")
                if f and t:
                    intervals.append(f"{f}-{t}")
            if intervals:
                times = ", ".join(intervals)
                for dr in day_ranges:
                    segments.append((dr, times))
        return segments
    except Exception:
        return []

def extract_hours_text_from_html(html: str) -> str:
    """
    Pull a compact hours string from the page:
    - prefer schema.org openingHours
    - else headings/labels like Öffnungszeiten, Sprechzeiten, Opening Hours
    """
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return ""
    # 1) schema.org first
    for tag in soup.find_all("script", {"type":"application/ld+json"}):
        try:
            j = json.loads(tag.string or "{}")
            items = j if isinstance(j, list) else [j]
            for o in items:
                oh = o.get("openingHours") or o.get("openingHoursSpecification")
                if isinstance(oh, list):
                    txt = "; ".join([str(x) for x in oh])
                    if txt.strip(): return txt
                elif isinstance(oh, str):
                    if oh.strip(): return oh
        except Exception:
            pass
    # 2) look for labels near blocks
    labels = ["öffnungszeiten", "sprechzeiten", "sprechstunde", "opening hours", "hours"]
    blocks = []
    for el in soup.find_all(text=True):
        s = (el or "").strip()
        if not s: 
            continue
        low = s.lower()
        if any(lbl in low for lbl in labels):
            parent = getattr(el, "parent", None)
            if parent:
                trail = []
                for sib in parent.next_siblings:
                    try:
                        if hasattr(sib, "get_text"):
                            t = sib.get_text(" ", strip=True)
                        elif isinstance(sib, str):
                            t = sib.strip()
                        else:
                            t = ""
                    except Exception:
                        t = ""
                    if t:
                        trail.append(t)
                    if len(" ".join(trail)) > 300:
                        break
                snippet = " ".join([s] + trail)
                if snippet:
                    blocks.append(snippet)
    return max(blocks, key=len) if blocks else ""

def fetch_site(url: str) -> str:
    if not url: return ""
    try:
        parsed = requests.utils.urlparse(url)
        root = f"{parsed.scheme}://{parsed.netloc}"
        robots = requests.get(root + "/robots.txt", timeout=15)
        if robots.status_code == 200 and "Disallow: /" in robots.text and "User-agent: *" in robots.text:
            return ""  # respect robots
    except Exception:
        pass
    time.sleep(1.0)  # polite
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and r.text:
            return r.text
    except Exception:
        return ""
    return ""

# ----------------------------
# Data model
# ----------------------------

@dataclass
class Clinic:
    name: str
    website: str
    phone: str
    email: str
    street: str
    housenumber: str
    postcode: str
    city: str
    lat: Optional[float]
    lon: Optional[float]        # internal; we export as 'lng'
    opening_hours: str = ""     # normalized to OSM OpeningHours when possible
    emergency_flag: bool = False

    def as_dict(self): return asdict(self)

# ----------------------------
# Overpass + Nominatim
# ----------------------------

def overpass_query(bbox: Tuple[float,float,float,float], max_results: int) -> List[Dict[str, Any]]:
    minlon, minlat, maxlon, maxlat = bbox
    q = f"""
    [out:json][timeout:60];
    (
      node["amenity"="veterinary"]({minlat},{minlon},{maxlat},{maxlon});
      way["amenity"="veterinary"]({minlat},{minlon},{maxlat},{maxlon});
      relation["amenity"="veterinary"]({minlat},{minlon},{maxlat},{maxlon});
    );
    out center {max_results};
    """.strip()

    for base in OVERPASS_URLS:
        try:
            time.sleep(1.0)  # polite per mirror
            data = http_json_with_retries("POST", base, data={"data": q}, headers=HEADERS, timeout=90, tries=4, polite_delay=1.5)
            return data.get("elements", [])
        except Exception as e:
            print(f"  ⚠️ Overpass mirror failed: {base} → {e}")
            continue
    raise RuntimeError("All Overpass mirrors failed.")

def fetch_osm_veterinary(bbox: Tuple[float,float,float,float], max_results: int, city: str, enrich_hours_from_site: bool) -> List[Clinic]:
    print("🔎 Fetching data from OpenStreetMap (amenity=veterinary)…")
    raw = overpass_query(bbox, max_results)
    clinics: List[Clinic] = []

    for el in raw:
        tags = el.get("tags", {}) or {}
        name = tags.get("name") or ""
        if not name:
            continue

        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lon = el.get("lon") or (el.get("center") or {}).get("lon")
        try:
            lat = float(lat) if lat is not None else None
            lon = float(lon) if lon is not None else None
        except Exception:
            lat, lon = None, None

        street = tags.get("addr:street", "") or ""
        housenumber = tags.get("addr:housenumber", "") or ""
        postcode = tags.get("addr:postcode", "") or ""
        website = norm_url(tags.get("website", "") or tags.get("contact:website", ""))
        phone = norm_phone(tags.get("phone", "") or tags.get("contact:phone", ""))
        email = tags.get("email", "") or tags.get("contact:email", "")

        # Opening hours:
        opening_hours = ""
        raw_oh = (tags.get("opening_hours") or "").strip()
        if looks_247(raw_oh):
            opening_hours = "24/7"
        elif raw_oh:
            segs = parse_hours_with_opentimeparser(raw_oh)
            opening_hours = normalize_osm_oh(segs) if segs else raw_oh

        # If missing, optionally try website
        if not opening_hours and enrich_hours_from_site and website:
            html = fetch_site(website)
            if html:
                candidate = extract_hours_text_from_html(html)
                if looks_247(candidate):
                    opening_hours = "24/7"
                else:
                    segs = parse_hours_with_opentimeparser(candidate)
                    opening_hours = normalize_osm_oh(segs) if segs else (candidate[:200] if candidate else "")

        is_247 = looks_247(opening_hours)
        is_emergency = derive_emergency(opening_hours, name, (tags.get("name") or "") + " " + (tags.get("description") or ""))

        c = Clinic(
            name=name,
            website=website,
            phone=phone,
            email=email,
            street=street,
            housenumber=housenumber,
            postcode=postcode,
            city=city,
            lat=lat,
            lon=lon,
            opening_hours=opening_hours or "",
            emergency_flag=is_emergency or is_247,
        )
        if c.lat and c.lon and within_bbox(c.lat, c.lon, bbox):
            clinics.append(c)

    print(f"✅ Found {len(clinics)} candidates from OSM (this tile)")
    return clinics

def nominatim_reverse(lat: float, lon: float, tries: int = 4, base_delay: float = 1.0) -> Optional[Dict[str, Any]]:
    delay = base_delay
    params = {
        "lat": lat,
        "lon": lon,
        "format": "jsonv2",
        "addressdetails": 1,
        "zoom": 18,
        "email": EMAIL,
    }
    last_err = None
    for _ in range(tries):
        time.sleep(1.0)  # polite
        try:
            r = requests.get(NOMINATIM_URL + "/reverse", params=params, headers=HEADERS, timeout=30)
            if r.status_code in (429, 502, 503, 504):
                last_err = RuntimeError(f"HTTP {r.status_code}")
                time.sleep(delay); delay = min(delay * 2, 15)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(delay); delay = min(delay * 2, 15)
    return None

def fill_missing_address(clinics: List[Clinic], city: str) -> List[Clinic]:
    print("🧭 Reverse-geocoding missing address parts (Nominatim)…")
    for c in tqdm(clinics):
        if not c.lat or not c.lon:
            continue
        need_street = not bool((c.street or "").strip())
        need_hnr = not bool((c.housenumber or "").strip())
        need_pc = not bool((c.postcode or "").strip())
        if not (need_street or need_hnr or need_pc):
            continue
        data = nominatim_reverse(c.lat, c.lon)
        if not data: 
            continue
        addr = (data or {}).get("address", {}) or {}
        if need_street:
            c.street = addr.get("road") or addr.get("pedestrian") or addr.get("footway") or c.street
        if need_hnr:
            c.housenumber = addr.get("house_number") or c.housenumber
        if need_pc:
            pc = addr.get("postcode") or ""
            m = re.search(r"\b(\d{5})\b", str(pc))
            if m:
                c.postcode = m.group(1)
    return clinics

# ----------------------------
# Export mapping to Supabase schema
# ----------------------------

def bool_to_db(v: bool):
    return bool(v)

def clinic_to_sb_row(c: Clinic) -> dict:
    opening = c.opening_hours or ""
    is_247 = looks_247(opening)
    is_emergency = bool(c.emergency_flag) or is_247
    row = {
        "name": c.name or "",
        "website": c.website or "",
        "phone": c.phone or "",
        "email": c.email or "",
        "street": c.street or "",
        "housenumber": c.housenumber or "",
        "postcode": c.postcode or "",
        "city": c.city or "",
        "lat": c.lat if c.lat is not None else None,
        "lng": c.lon if c.lon is not None else None,

        "opening_hours": opening,
        "emergency": "Notdienst" if is_emergency else "",
        "emergency_boolean": bool_to_db(is_emergency),
        "twentyfour_seven": bool_to_db(is_247),

        "supports_mobile": bool_to_db(False),
        "booking_enabled": bool_to_db(False),
        "active": bool_to_db(True),         # set True so they appear; change to False if you prefer QA
        "active_boolean": bool_to_db(True),

        "offers_checkup": bool_to_db(False),
        "offers_dental": bool_to_db(False),
        "offers_illness": bool_to_db(False),
        "offers_prescription": bool_to_db(False),
        "offers_vaccination": bool_to_db(False),

        "description": "",
        "district": "",
        "onboarding_status": "",

        "created_at": "",
        "updated_at": "",

        "got_min_multiplier": "",
        "got_max_multiplier": "",
        "got_notes": "",

        "weekend_min_multiplier": "",
        "weekend_policy_notes": "",

        "notdienst_fee_eur": "",
        "travel_cost_min_eur": "",
        "travel_cost_per_double_km_eur": "",

        "id": "",
        "auth_user_id": "",
        "invite_sent_at": "",
        "last_login_at": "",
    }
    return row

def export_supabase_csv(clinics: list, out_dir: str, city_slug: str):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"clinics_{city_slug}.csv")
    jsonl_path = os.path.join(out_dir, f"clinics_{city_slug}.jsonl")

    rows = [clinic_to_sb_row(c) for c in clinics]

    # CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SB_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in SB_COLUMNS})

    # JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r.get(k, "") for k in SB_COLUMNS}, ensure_ascii=False) + "\n")

    print(f"\n✅ Exported Supabase-layout CSV → {csv_path}")
    print(f"✅ Exported JSONL → {jsonl_path}\n")

# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default=CITY_DEFAULT)
    parser.add_argument("--max-results", type=int, default=5000)
    parser.add_argument("--output-dir", default="./out")
    parser.add_argument("--tiles", type=int, default=2, choices=[1,2,3,4], help="Split bbox into N×N tiles")
    parser.add_argument("--no-geocode", action="store_true", help="Skip Nominatim reverse for missing address")
    parser.add_argument("--enrich-websites", action="store_true", help="Visit site to extract opening hours when OSM lacks it")
    args = parser.parse_args()

    city = args.city
    city_slug = re.sub(r"[^a-z0-9]+", "-", city.lower()).strip("-") or "city"
    bbox = BERLIN_BBOX  # this script targets Berlin

    all_clinics: List[Clinic] = []

    tiles = tile_bbox(bbox, args.tiles)
    print(f"📦 Querying {len(tiles)} tile(s)…")
    for i, tb in enumerate(tiles, start=1):
        print(f"— Tile {i}/{len(tiles)}")
        part = fetch_osm_veterinary(tb, args.max_results, city, enrich_hours_from_site=args.enrich_websites)
        all_clinics.extend(part)

    # Deduplicate by (name, lat/lon rounded 5)
    seen = set()
    uniq: List[Clinic] = []
    for c in all_clinics:
        key = (c.name.lower().strip(), round(c.lat or 0, 5), round(c.lon or 0, 5))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    # Optional address backfill
    if not args.no_geocode:
        uniq = fill_missing_address(uniq, city)

    # Keep rows with coords inside bbox
    uniq = [c for c in uniq if c.lat and c.lon and within_bbox(c.lat, c.lon, bbox)]

    print(f"🧹 Final count: {len(uniq)} clinics after merge/dedupe")

    export_supabase_csv(uniq, args.output_dir, city_slug)

if __name__ == "__main__":
    main()
