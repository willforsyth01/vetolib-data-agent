#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vetolib Web Data Agent — Berlin (DE)

Aligned to Supabase table public.clinics:

  id, name, street, district, city, postcode, lat, lng,
  phone, email, website, opening_hours (jsonb), emergency (bool),
  active (bool), created_at, updated_at, description,
  emergency_boolean, active_boolean, booking_enabled, supports_mobile,
  twentyfour_seven, got_min_multiplier, got_max_multiplier,
  weekend_min_multiplier, notdienst_fee_eur, got_notes,
  weekend_policy_notes, travel_cost_per_double_km_eur,
  travel_cost_min_eur, mobile_service_area_km,
  offers_vaccination, offers_checkup, offers_illness,
  offers_prescription, offers_dental, auth_user_id, contact_email,
  onboarding_status, invite_sent_at, last_login_at

Exports:
  out/clinics_berlin.csv
  out/clinics_berlin.jsonl
"""

import os
import re
import csv
import json
import time
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ----------------------------
# Config / constants
# ----------------------------

CITY_DEFAULT = "Berlin"
BERLIN_BBOX = (13.08835, 52.33826, 13.76116, 52.67551)

OVERPASS_URLS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

NOMINATIM_URL = "https://nominatim.openstreetmap.org"
HEADERS = {"User-Agent": "Vetolib-Agent/1.3 (+https://vetolib.app)"}
EMAIL = os.getenv("NOMINATIM_EMAIL", "you@example.com")

# ----------------------------
# Supabase CSV columns
# EXACTLY matching public.clinics schema
# ----------------------------

SB_COLUMNS = [
    "id",
    "name",
    "street",
    "district",
    "city",
    "postcode",
    "lat",
    "lng",
    "phone",
    "email",
    "website",
    "opening_hours",
    "emergency",
    "active",
    "created_at",
    "updated_at",
    "description",
    "emergency_boolean",
    "active_boolean",
    "booking_enabled",
    "supports_mobile",
    "twentyfour_seven",
    "got_min_multiplier",
    "got_max_multiplier",
    "weekend_min_multiplier",
    "notdienst_fee_eur",
    "got_notes",
    "weekend_policy_notes",
    "travel_cost_per_double_km_eur",
    "travel_cost_min_eur",
    "mobile_service_area_km",
    "offers_vaccination",
    "offers_checkup",
    "offers_illness",
    "offers_prescription",
    "offers_dental",
    "auth_user_id",
    "contact_email",
    "onboarding_status",
    "invite_sent_at",
    "last_login_at",
]

# ----------------------------
# Helpers
# ----------------------------

def within_bbox(lat: float, lon: float, bbox: Tuple[float,float,float,float]) -> bool:
    minlon, minlat, maxlon, maxlat = bbox
    return (minlat <= lat <= maxlat) and (minlon <= lon <= maxlon)

def norm_url(url: str) -> str:
    if not url:
        return ""
    u = url.strip()
    if u.startswith("//"):
        u = "https:" + u
    if not re.match(r"^https?://", u):
        u = "https://" + u
    return u.rstrip("/")

def norm_phone(phone: str) -> str:
    return re.sub(r"[^\d+]", "", phone or "")[:25]

def http_json_with_retries(
    method, url, *, data=None, params=None, headers=None,
    timeout=90, tries=4, polite_delay=1.0
):
    delay = polite_delay
    last_err = None
    for i in range(tries):
        try:
            resp = requests.request(
                method, url, data=data, params=params,
                headers=headers, timeout=timeout
            )
            ctype = resp.headers.get("content-type", "")
            if resp.status_code in (429, 502, 503, 504) or "json" not in ctype.lower():
                last_err = Exception(f"{resp.status_code} {ctype} (attempt {i+1}/{tries})")
                time.sleep(delay)
                delay = min(delay * 2, 20)
                continue
            return resp.json()
        except Exception as e:
            last_err = e
            time.sleep(delay)
            delay = min(delay * 2, 20)
    raise last_err

def tile_bbox(bbox: Tuple[float,float,float,float], tiles_per_side: int) -> List[Tuple[float,float,float,float]]:
    minlon, minlat, maxlon, maxlat = bbox
    tiles = []
    for i in range(tiles_per_side):
        for j in range(tiles_per_side):
            lon1 = minlon + i * (maxlon - minlon) / tiles_per_side
            lon2 = minlon + (i + 1) * (maxlon - minlon) / tiles_per_side
            lat1 = minlat + j * (maxlat - minlat) / tiles_per_side
            lat2 = minlat + (j + 1) * (maxlat - minlat) / tiles_per_side
            tiles.append((lon1, lat1, lon2, lat2))
    return tiles

# ----------------------------
# Opening hours helpers
# ----------------------------

def looks_247(s: str) -> bool:
    s = (s or "").lower()
    return any(k in s for k in ["24/7", "24h", "24 h", "24 stunden", "rund um die uhr"])

EMERGENCY_KWS = ["notdienst", "emergency", "24/7", "24 stunden", "24h", "24 h"]

def derive_emergency(opening_hours: str, name: str, extra_text: str = "") -> bool:
    blob = " ".join([opening_hours or "", name or "", extra_text or ""]).lower()
    return any(kw in blob for kw in EMERGENCY_KWS)

def parse_hours_with_opentimeparser(raw_text: str):
    try:
        from opentimeparser import parse as otp_parse
    except Exception:
        return []
    try:
        parsed = otp_parse(raw_text or "")
        segments = []
        for block in parsed:
            days = block.get("days") or []
            hours = block.get("hours") or []
            if not hours:
                for d in days:
                    segments.append((",".join(days), "off"))
            else:
                for h in hours:
                    fr = (h.get("from") or "").replace(".", ":")
                    to = (h.get("to") or "").replace(".", ":")
                    if fr and to:
                        segments.append((", ".join(days), f"{fr}-{to}"))
        return segments
    except Exception:
        return []

def normalize_osm_oh(segments) -> str:
    parts = []
    for d, t in segments:
        parts.append(f"{d} {t}")
    return "; ".join(parts)

def extract_hours_text_from_html(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return ""
    # 1) JSON-LD
    for tag in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            j = json.loads(tag.string or "{}")
            if isinstance(j, dict):
                oh = j.get("openingHours") or j.get("openingHoursSpecification")
                if isinstance(oh, str):
                    return oh
                if isinstance(oh, list):
                    return "; ".join(map(str, oh))
            if isinstance(j, list):
                for o in j:
                    oh = o.get("openingHours") or o.get("openingHoursSpecification")
                    if isinstance(oh, str):
                        return oh
                    if isinstance(oh, list):
                        return "; ".join(map(str, oh))
        except Exception:
            pass
    # 2) Label-based heuristic
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
                blocks.append(" ".join([s] + trail))
    return max(blocks, key=len) if blocks else ""

def fetch_site(url: str) -> str:
    if not url:
        return ""
    try:
        parsed = requests.utils.urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        robots = requests.get(base + "/robots.txt", timeout=15)
        if robots.status_code == 200 and "Disallow: /" in robots.text:
            return ""
    except Exception:
        pass
    time.sleep(1.0)
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and r.text:
            return r.text
    except Exception:
        return ""
    return ""

# ----------------------------
# Service enrichment
# ----------------------------

def derive_service_flags(tags: Dict[str, str], text_blob: str = "") -> Dict[str, bool]:
    blob = " ".join(list(tags.values()) + [text_blob or ""]).lower()

    def any_kw(words: List[str]) -> bool:
        return any(w in blob for w in words)

    supports_mobile = any_kw([
        "hausbesuch", "hausbesuche", "mobile tierarzt", "fahrpraxis",
        "home visit", "house-call"
    ])

    offers_checkup = any_kw([
        "vorsorge", "check-up", "routineuntersuchung", "jahrescheck",
        "gesundheitscheck"
    ])

    offers_dental = any_kw([
        "zahn", "zahnheilkunde", "zahnbehandlung", "zahnstein",
        "zahnsanierung", "dental", "dentistry"
    ])

    offers_illness = any_kw([
        "chirurgie", "operation", "op ", "op-", "kardiologie", "onkologie",
        "dermatologie", "neurologie", "krankheit", "erkrankung", "notfall"
    ])

    # defaults: almost all vets do prescriptions + vaccinations
    offers_prescription = True
    offers_vaccination = True

    return {
        "supports_mobile": supports_mobile,
        "offers_checkup": offers_checkup,
        "offers_dental": offers_dental,
        "offers_illness": offers_illness,
        "offers_prescription": offers_prescription,
        "offers_vaccination": offers_vaccination,
    }

# ----------------------------
# Data model
# ----------------------------

@dataclass
class Clinic:
    name: str
    street: str
    district: str
    city: str
    postcode: str
    lat: Optional[float]
    lng: Optional[float]
    phone: str
    email: str
    website: str
    opening_hours: str
    emergency_flag: bool
    supports_mobile: bool = False
    offers_checkup: bool = False
    offers_illness: bool = False
    offers_prescription: bool = True
    offers_dental: bool = False
    offers_vaccination: bool = True

# ----------------------------
# OSM + enrichment
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
            time.sleep(1.0)
            data = http_json_with_retries(
                "POST", base, data={"data": q},
                headers=HEADERS, timeout=90, tries=4, polite_delay=1.5
            )
            return data.get("elements", [])
        except Exception as e:
            print(f"⚠️ Overpass mirror failed {base}: {e}")
    raise RuntimeError("All Overpass mirrors failed.")

def fetch_osm_veterinary(
    bbox: Tuple[float,float,float,float],
    max_results: int,
    city: str,
    enrich_websites: bool
) -> List[Clinic]:
    print("🔎 Fetching data from OSM (amenity=veterinary)…")
    raw = overpass_query(bbox, max_results)
    clinics: List[Clinic] = []

    for el in raw:
        tags = el.get("tags", {}) or {}
        name = tags.get("name", "").strip()
        if not name:
            continue

        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lng = el.get("lon") or (el.get("center") or {}).get("lon")
        try:
            lat = float(lat) if lat is not None else None
            lng = float(lng) if lng is not None else None
        except Exception:
            lat, lng = None, None

        street = tags.get("addr:street", "") or ""
        postcode = tags.get("addr:postcode", "") or ""
        district = tags.get("addr:suburb", "") or ""

        website = norm_url(tags.get("website", "") or tags.get("contact:website", ""))
        phone = norm_phone(tags.get("phone", "") or tags.get("contact:phone", ""))
        email = tags.get("email", "") or tags.get("contact:email", "")

        # Opening hours from OSM
        opening_hours = ""
        raw_oh = (tags.get("opening_hours") or "").strip()
        if looks_247(raw_oh):
            opening_hours = "24/7"
        elif raw_oh:
            segs = parse_hours_with_opentimeparser(raw_oh)
            opening_hours = normalize_osm_oh(segs) if segs else raw_oh

        # Website HTML
        html = ""
        if enrich_websites and website:
            html = fetch_site(website)

        # Derive opening hours from HTML if missing
        if not opening_hours and html:
            candidate = extract_hours_text_from_html(html)
            if looks_247(candidate):
                opening_hours = "24/7"
            else:
                segs = parse_hours_with_opentimeparser(candidate)
                opening_hours = normalize_osm_oh(segs) if segs else (candidate[:200] if candidate else "")

        is_247 = looks_247(opening_hours)
        is_emergency = derive_emergency(opening_hours, name, tags.get("description", ""))

        blob = " ".join([
            name,
            tags.get("description", "") or "",
            tags.get("services", "") or "",
            opening_hours or "",
            (html[:2000] if html else ""),
        ])
        svc = derive_service_flags(tags, blob)

        c = Clinic(
            name=name,
            street=street,
            district=district,
            city=city,
            postcode=postcode,
            lat=lat,
            lng=lng,
            phone=phone,
            email=email,
            website=website,
            opening_hours=opening_hours or "",
            emergency_flag=is_emergency or is_247,
            supports_mobile=svc["supports_mobile"],
            offers_checkup=svc["offers_checkup"],
            offers_illness=svc["offers_illness"],
            offers_prescription=svc["offers_prescription"],
            offers_dental=svc["offers_dental"],
            offers_vaccination=svc["offers_vaccination"],
        )

        if c.lat and c.lng and within_bbox(c.lat, c.lng, bbox):
            clinics.append(c)

    print(f"✅ Found {len(clinics)} clinics in this tile")
    return clinics

# ----------------------------
# Nominatim reverse geocoding
# ----------------------------

def nominatim_reverse(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    params = {
        "lat": lat,
        "lon": lon,
        "format": "jsonv2",
        "addressdetails": 1,
        "zoom": 18,
        "email": EMAIL,
    }
    try:
        time.sleep(1.0)
        r = requests.get(NOMINATIM_URL + "/reverse", params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def fill_missing_address(clinics: List[Clinic]) -> List[Clinic]:
    print("🧭 Reverse-geocoding missing address info…")
    for c in tqdm(clinics):
        if not c.lat or not c.lng:
            continue
        need_street = not bool(c.street.strip())
        need_pc = not bool(c.postcode.strip())
        need_district = not bool(c.district.strip())
        if not (need_street or need_pc or need_district):
            continue

        data = nominatim_reverse(c.lat, c.lng)
        if not data:
            continue
        addr = (data or {}).get("address", {}) or {}
        if need_street:
            c.street = addr.get("road") or addr.get("pedestrian") or c.street
        if need_pc:
            pc = addr.get("postcode") or ""
            m = re.search(r"\b(\d{5})\b", pc)
            if m:
                c.postcode = m.group(1)
        if need_district:
            c.district = addr.get("suburb") or addr.get("neighbourhood") or c.district
    return clinics

# ----------------------------
# Export mapping
# ----------------------------

def clinic_to_sb_row(c: Clinic) -> Dict[str, Any]:
    opening = c.opening_hours or ""
    is_247 = looks_247(opening)
    is_emergency = bool(c.emergency_flag) or is_247

    row = {
        "id": "",  # let DB default gen_random_uuid()
        "name": c.name or "",
        "street": c.street or "",
        "district": c.district or "",
        "city": c.city or "",
        "postcode": c.postcode or "",
        "lat": c.lat,
        "lng": c.lng,
        "phone": c.phone or "",
        "email": c.email or "",
        "website": c.website or "",
        "opening_hours": opening,  # text; DB will store as jsonb string
        "emergency": bool(is_emergency),
        "active": True,
        "created_at": "",
        "updated_at": "",
        "description": "",
        "emergency_boolean": bool(is_emergency),
        "active_boolean": True,
        "booking_enabled": False,
        "supports_mobile": bool(c.supports_mobile),
        "twentyfour_seven": bool(is_247),
        "got_min_multiplier": "",
        "got_max_multiplier": "",
        "weekend_min_multiplier": "",
        "notdienst_fee_eur": "",
        "got_notes": "",
        "weekend_policy_notes": "",
        "travel_cost_per_double_km_eur": "",
        "travel_cost_min_eur": "",
        "mobile_service_area_km": "",
        "offers_vaccination": bool(c.offers_vaccination),
        "offers_checkup": bool(c.offers_checkup),
        "offers_illness": bool(c.offers_illness),
        "offers_prescription": bool(c.offers_prescription),
        "offers_dental": bool(c.offers_dental),
        "auth_user_id": "",
        "contact_email": c.email or "",
        "onboarding_status": "not_invited",
        "invite_sent_at": "",
        "last_login_at": "",
    }
    return row

def export_supabase_csv(clinics: List[Clinic], out_dir: str, city_slug: str):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"clinics_{city_slug}.csv")
    jsonl_path = os.path.join(out_dir, f"clinics_{city_slug}.jsonl")

    rows = [clinic_to_sb_row(c) for c in clinics]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SB_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in SB_COLUMNS})

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r.get(k, "") for k in SB_COLUMNS}, ensure_ascii=False) + "\n")

    print(f"✅ Exported CSV → {csv_path}")
    print(f"✅ Exported JSONL → {jsonl_path}")

# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default=CITY_DEFAULT)
    parser.add_argument("--max-results", type=int, default=5000)
    parser.add_argument("--output-dir", default="./out")
    parser.add_argument("--tiles", type=int, default=2)
    parser.add_argument("--no-geocode", action="store_true")
    parser.add_argument("--enrich-websites", action="store_true")
    args = parser.parse_args()

    city = args.city
    city_slug = re.sub(r"[^a-z0-9]+", "-", city.lower()).strip("-") or "city"
    bbox = BERLIN_BBOX

    tiles = tile_bbox(bbox, args.tiles)
    print(f"📦 Querying {len(tiles)} tiles…")

    all_clinics: List[Clinic] = []
    for i, tb in enumerate(tiles, start=1):
        print(f"— Tile {i}/{len(tiles)}")
        all_clinics.extend(fetch_osm_veterinary(tb, args.max_results, city, args.enrich_websites))

    # de-duplicate by (name, lat, lng)
    seen = set()
    uniq: List[Clinic] = []
    for c in all_clinics:
        key = (c.name.lower().strip(), round(c.lat or 0, 5), round(c.lng or 0, 5))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    if not args.no_geocode:
        uniq = fill_missing_address(uniq)

    # final bbox filter
    uniq = [c for c in uniq if c.lat and c.lng and within_bbox(c.lat, c.lng, bbox)]

    print(f"🧹 Final count: {len(uniq)} clinics after merge/dedupe")
    export_supabase_csv(uniq, args.output_dir, city_slug)

if __name__ == "__main__":
    main()
