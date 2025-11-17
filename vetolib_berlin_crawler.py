#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vetolib Web Data Agent — Berlin (DE)

Full crawler with:
- Overpass + retry + tiles
- Nominatim reverse geocoding
- Website scraping for opening hours + service enrichment
- Default assumptions for prescriptions + vaccinations
- Export EXACTLY matching Supabase schema (using `housenumber`, NOT `house_number`)

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
from bs4 import BeautifulSoup
from tqdm import tqdm

# ----------------------------
# Config / constants
# ----------------------------
CITY_DEFAULT = "Berlin"
STATE_DEFAULT = "Berlin"
COUNTRY_DEFAULT = "DE"

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
# Supabase-schema CSV columns
# NOTE: housenumber is required (NOT house_number)
# ----------------------------
SB_COLUMNS = [
    "name","website","phone","email",
    "street","housenumber","postcode","city","lat","lng",
    "opening_hours","emergency","emergency_boolean","twentyfour_seven",
    "supports_mobile","booking_enabled","active","active_boolean",
    "offers_checkup","offers_dental","offers_illness","offers_prescription","offers_vaccination",
    "description","district","onboarding_status",
    "created_at","updated_at",
    "got_min_multiplier","got_max_multiplier","got_notes",
    "weekend_min_multiplier","weekend_policy_notes",
    "notdienst_fee_eur","travel_cost_min_eur","travel_cost_per_double_km_eur",
    "id","auth_user_id","invite_sent_at","last_login_at",
]

# ----------------------------
# Helpers
# ----------------------------

def within_bbox(lat: float, lon: float, bbox):
    minlon, minlat, maxlon, maxlat = bbox
    return (minlat <= lat <= maxlat) and (minlon <= lon <= maxlon)

def norm_url(url):
    if not url: return ""
    u = url.strip()
    if u.startswith("//"):
        u = "https:" + u
    if not re.match(r"^https?://", u):
        u = "https://" + u
    return u.rstrip("/")

def norm_phone(phone):
    return re.sub(r"[^\d+]", "", phone or "")[:25]

def http_json_with_retries(method, url, *, data=None, params=None, headers=None,
                           timeout=90, tries=4, polite_delay=1.0):
    delay = polite_delay
    last = None
    for i in range(tries):
        try:
            resp = requests.request(method, url, data=data, params=params,
                                    headers=headers, timeout=timeout)
            ctype = resp.headers.get("content-type", "")
            if resp.status_code in (429,502,503,504) or "json" not in ctype.lower():
                last = Exception(f"{resp.status_code} {ctype}")
                time.sleep(delay)
                delay = min(delay*2, 15)
                continue
            return resp.json()
        except Exception as e:
            last = e
            time.sleep(delay)
            delay = min(delay*2, 15)
    raise last

def tile_bbox(bbox, n):
    minlon, minlat, maxlon, maxlat = bbox
    tiles = []
    for i in range(n):
        for j in range(n):
            lon1 = minlon + i*(maxlon-minlon)/n
            lon2 = minlon + (i+1)*(maxlon-minlon)/n
            lat1 = minlat + j*(maxlat-minlat)/n
            lat2 = minlat + (j+1)*(maxlat-minlat)/n
            tiles.append((lon1, lat1, lon2, lat2))
    return tiles

# ----------------------------
# Opening hours helpers
# ----------------------------

def looks_247(s):
    s = (s or "").lower()
    return any(k in s for k in ["24/7","24h","24 h","24 stunden","rund um die uhr"])

EMERGENCY_KWS = ["notdienst","emergency","24/7","24 stunden","24h","24 h"]

def derive_emergency(opening_hours, name, extra=""):
    blob = " ".join([opening_hours or "", name or "", extra or ""]).lower()
    return any(kw in blob for kw in EMERGENCY_KWS)

def parse_hours_with_opentimeparser(raw):
    try:
        from opentimeparser import parse as otp
    except:
        return []
    try:
        parsed = otp(raw or "")
        segs = []
        for block in parsed:
            days = block.get("days") or []
            hours = block.get("hours") or []
            if not hours:
                for d in days:
                    segs.append((d, "off"))
            else:
                for h in hours:
                    fr = (h.get("from") or "").replace(".",":")
                    to = (h.get("to") or "").replace(".",":")
                    if fr and to:
                        segs.append((",".join(days), f"{fr}-{to}"))
        return segs
    except:
        return []

def normalize_osm_oh(segs):
    outs = []
    for d,t in segs:
        outs.append(f"{d} {t}")
    return "; ".join(outs)

def extract_hours_text_from_html(html):
    try: soup = BeautifulSoup(html, "lxml")
    except: return ""

    # JSON-LD first
    for tag in soup.find_all("script", {"type":"application/ld+json"}):
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
        except:
            pass

    labels = ["öffnungszeiten","sprechzeiten","opening hours","hours"]
    blocks = []
    for el in soup.find_all(text=True):
        s = (el or "").strip()
        if not s: continue
        low = s.lower()
        if any(lbl in low for lbl in labels):
            parent = getattr(el,"parent",None)
            if parent:
                trail=[]
                for sib in parent.next_siblings:
                    try:
                        if hasattr(sib,"get_text"):
                            t = sib.get_text(" ",strip=True)
                        elif isinstance(sib,str):
                            t = sib.strip()
                        else:
                            t=""
                    except:
                        t=""
                    if t: trail.append(t)
                    if len(" ".join(trail))>300: break
                blocks.append(" ".join([s]+trail))
    return max(blocks,key=len) if blocks else ""

def fetch_site(url):
    if not url: return ""
    try:
        parsed = requests.utils.urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        robots = requests.get(base+"/robots.txt", timeout=15)
        if robots.status_code == 200 and "Disallow: /" in robots.text:
            return ""
    except: pass
    time.sleep(1.0)
    try:
        r = requests.get(url,timeout=30)
        if r.status_code==200:
            return r.text
    except:
        return ""
    return ""

# ----------------------------
# Service enrichment
# ----------------------------

def derive_service_flags(tags, blob):
    blob = (blob or "").lower()

    def any_kw(words): return any(w in blob for w in words)

    supports_mobile = any_kw([
        "hausbesuch","hausbesuche","mobile tierarzt","fahrpraxis",
        "home visit","house-call"
    ])

    offers_checkup = any_kw([
        "vorsorge","check-up","routineuntersuchung","jahrescheck"
    ])

    offers_dental = any_kw([
        "zahn","dental","zahnsanierung","zahnstein","zahn-op"
    ])

    offers_illness = any_kw([
        "chirurgie","operation","kardiologie","onkologie","dermatologie","neurologie",
        "krankheit","erkrankung","notfall"
    ])

    # Defaults: vets always do these
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
# Clinic model
# ----------------------------

@dataclass
class Clinic:
    name: str
    website: str
    phone: str
    email: str
    street: str
    house_number: str
    postcode: str
    city: str
    lat: Optional[float]
    lon: Optional[float]
    opening_hours: str = ""
    emergency_flag: bool = False

    supports_mobile: bool = False
    offers_checkup: bool = False
    offers_dental: bool = False
    offers_illness: bool = False
    offers_prescription: bool = True
    offers_vaccination: bool = True

# ----------------------------
# OSM fetch
# ----------------------------

def overpass_query(bbox, max_results):
    minlon, minlat, maxlon, maxlat = bbox
    q = f"""
    [out:json][timeout:60];
    (
      node["amenity"="veterinary"]({minlat},{minlon},{maxlat},{maxlon});
      way["amenity"="veterinary"]({minlat},{minlon},{maxlat},{maxlon});
      relation["amenity"="veterinary"]({minlat},{minlon},{maxlat},{maxlon});
    );
    out center {max_results};
    """

    for url in OVERPASS_URLS:
        try:
            time.sleep(1)
            data = http_json_with_retries("POST", url, data={"data":q},
                                          headers=HEADERS, timeout=90,
                                          tries=4, polite_delay=1.5)
            return data.get("elements",[])
        except Exception as e:
            print(f"⚠️ Overpass failed {url}: {e}")
    raise RuntimeError("All Overpass mirrors failed.")

def fetch_osm_veterinary(bbox, max_results, city, enrich_websites):
    print("🔎 Fetching OSM data…")
    raw = overpass_query(bbox, max_results)
    out = []

    for el in raw:
        tags = el.get("tags",{}) or {}
        name = tags.get("name","").strip()
        if not name: continue

        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lon = el.get("lon") or (el.get("center") or {}).get("lon")
        try:
            lat = float(lat) if lat else None
            lon = float(lon) if lon else None
        except:
            lat,lon = None,None

        street = tags.get("addr:street","")
        hnum  = tags.get("addr:housenumber","")
        pc    = tags.get("addr:postcode","")
        website = norm_url(tags.get("website","") or tags.get("contact:website",""))
        phone   = norm_phone(tags.get("phone","") or tags.get("contact:phone",""))
        email   = tags.get("email","") or tags.get("contact:email","")

        # Opening hours from OSM
        opening_hours = ""
        raw_oh = tags.get("opening_hours","").strip()
        if looks_247(raw_oh):
            opening_hours = "24/7"
        elif raw_oh:
            segs = parse_hours_with_opentimeparser(raw_oh)
            opening_hours = normalize_osm_oh(segs) if segs else raw_oh

        # Fetch website HTML if requested
        html = ""
        if enrich_websites and website:
            html = fetch_site(website)

        # Derive opening hours from site if missing
        if not opening_hours and html:
            cand = extract_hours_text_from_html(html)
            if looks_247(cand):
                opening_hours = "24/7"
            else:
                segs = parse_hours_with_opentimeparser(cand)
                opening_hours = normalize_osm_oh(segs) if segs else (cand[:200] if cand else "")

        is_247 = looks_247(opening_hours)
        is_emergency = derive_emergency(opening_hours, name, tags.get("description",""))

        # Build text blob for enrichment
        blob = " ".join([
            name, tags.get("description",""), tags.get("services",""),
            opening_hours or "",
            (html[:2000] if html else "")
        ])

        svc = derive_service_flags(tags, blob)

        c = Clinic(
            name=name,
            website=website,
            phone=phone,
            email=email,
            street=street,
            house_number=hnum,
            postcode=pc,
            city=city,
            lat=lat,
            lon=lon,
            opening_hours=opening_hours,
            emergency_flag=is_emergency or is_247,

            supports_mobile=svc["supports_mobile"],
            offers_checkup=svc["offers_checkup"],
            offers_dental=svc["offers_dental"],
            offers_illness=svc["offers_illness"],
            offers_prescription=svc["offers_prescription"],
            offers_vaccination=svc["offers_vaccination"],
        )

        if c.lat and c.lon and within_bbox(c.lat, c.lon, bbox):
            out.append(c)

    print(f"✅ Found {len(out)} clinics in this tile")
    return out

# ----------------------------
# Nominatim reverse
# ----------------------------

def nominatim_reverse(lat, lon):
    params = {
        "lat":lat,"lon":lon,"format":"jsonv2",
        "addressdetails":1,"zoom":18,"email":EMAIL
    }
    try:
        r = requests.get(NOMINATIM_URL+"/reverse", params=params,
                         headers=HEADERS, timeout=30)
        r.raise_for_status()
        return r.json()
    except:
        return None

def fill_missing_address(clinics):
    print("🧭 Reverse-geocoding missing addresses…")
    for c in tqdm(clinics):
        if not c.lat or not c.lon: continue
        need_street = not c.street
        need_hnum   = not c.house_number
        need_pc     = not c.postcode
        if not (need_street or need_hnum or need_pc): continue

        data = nominatim_reverse(c.lat, c.lon)
        if not data: continue
        addr = data.get("address",{}) or {}

        if need_street:
            c.street = addr.get("road") or addr.get("pedestrian") or c.street
        if need_hnum:
            c.house_number = addr.get("house_number") or c.house_number
        if need_pc:
            pc = addr.get("postcode") or ""
            m = re.search(r"\b(\d{5})\b", pc)
            if m: c.postcode = m.group(1)
    return clinics

# ----------------------------
# Export
# ----------------------------

def clinic_to_sb_row(c: Clinic):
    opening = c.opening_hours or ""
    is_247 = looks_247(opening)
    is_emergency = c.emergency_flag or is_247

    return {
        "name": c.name,
        "website": c.website,
        "phone": c.phone,
        "email": c.email,

        "street": c.street,
        "housenumber": c.house_number,   # <-- FIXED HERE
        "postcode": c.postcode,
        "city": c.city,
        "lat": c.lat,
        "lng": c.lon,

        "opening_hours": opening,
        "emergency": "Notdienst" if is_emergency else "",
        "emergency_boolean": bool(is_emergency),
        "twentyfour_seven": bool(is_247),

        "supports_mobile": bool(c.supports_mobile),
        "booking_enabled": False,
        "active": True,
        "active_boolean": True,

        "offers_checkup": bool(c.offers_checkup),
        "offers_dental": bool(c.offers_dental),
        "offers_illness": bool(c.offers_illness),
        "offers_prescription": bool(c.offers_prescription),
        "offers_vaccination": bool(c.offers_vaccination),

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

def export_supabase_csv(clinics, out_dir, city_slug):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"clinics_{city_slug}.csv")
    jsonl_path = os.path.join(out_dir, f"clinics_{city_slug}.jsonl")

    rows = [clinic_to_sb_row(c) for c in clinics]

    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SB_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k:r.get(k,"") for k in SB_COLUMNS})

    with open(jsonl_path,"w",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k:r.get(k,"") for k in SB_COLUMNS}, ensure_ascii=False)+"\n")

    print(f"✅ Exported CSV → {csv_path}")
    print(f"✅ Exported JSONL → {jsonl_path}")

# ----------------------------
# Main
# ----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--city", default=CITY_DEFAULT)
    p.add_argument("--max-results", type=int, default=5000)
    p.add_argument("--output-dir", default="./out")
    p.add_argument("--tiles", type=int, default=2)
    p.add_argument("--no-geocode", action="store_true")
    p.add_argument("--enrich-websites", action="store_true")
    a = p.parse_args()

    city = a.city
    slug = re.sub(r"[^a-z0-9]+","-",city.lower()).strip("-") or "city"

    tiles = tile_bbox(BERLIN_BBOX, a.tiles)
    allc = []

    print(f"📦 Querying {len(tiles)} tiles…")

    for i,tb in enumerate(tiles,1):
        print(f"— Tile {i}/{len(tiles)}")
        allc.extend(fetch_osm_veterinary(tb, a.max_results, city, a.enrich_websites))

    # Dedupe
    seen=set()
    uniq=[]
    for c in allc:
        key=(c.name.lower().strip(), round(c.lat or 0,5), round(c.lon or 0,5))
        if key in seen: continue
        seen.add(key)
        uniq.append(c)

    if not a.no_geocode:
        uniq = fill_missing_address(uniq)

    uniq = [c for c in uniq if c.lat and c.lon and within_bbox(c.lat,c.lon,BERLIN_BBOX)]

    print(f"🧹 Final count: {len(uniq)}")

    export_supabase_csv(uniq, a.output_dir, slug)

if __name__ == "__main__":
    main()
