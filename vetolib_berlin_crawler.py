#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vetolib Web Data Agent — Berlin (DE)
====================================

This lightweight crawler collects basic public data about veterinary clinics in Berlin
using the OpenStreetMap Overpass API and optional Nominatim geocoding.

✅ Exports two files:
    ./out/clinics_berlin.csv
    ./out/clinics_berlin.jsonl

❌ Does NOT connect to Supabase or modify your app.
⚖️ Respects robots.txt and polite rate limits.
"""

import os
import re
import csv
import json
import math
import time
import requests
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List
from tqdm import tqdm

# ----------------------------
# Config
# ----------------------------
CITY = "Berlin"
COUNTRY = "DE"
STATE = "Berlin"
EMAIL = os.getenv("NOMINATIM_EMAIL", "you@example.com")
OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"
NOMINATIM_URL = "https://nominatim.openstreetmap.org"
HEADERS = {"User-Agent": "Vetolib-Agent/1.0 (+https://vetolib.example)"}
RATE_LIMIT = 1.0  # seconds between requests
OUT_DIR = "./out"

BERLIN_BBOX = (13.08835, 52.33826, 13.76116, 52.67551)  # minlon, minlat, maxlon, maxlat


# ----------------------------
# Helpers
# ----------------------------
def sleep():
    time.sleep(RATE_LIMIT)


def within_bbox(lat, lon, bbox=BERLIN_BBOX):
    minlon, minlat, maxlon, maxlat = bbox
    return (minlat <= lat <= maxlat) and (minlon <= lon <= maxlon)


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return text or "unknown"


def norm_phone(phone: str) -> str:
    phone = re.sub(r"[^\d+]", "", phone or "")
    return phone[:20]


def norm_url(url: str) -> str:
    if not url:
        return ""
    if url.startswith("//"):
        url = "https:" + url
    if not url.startswith("http"):
        url = "https://" + url
    return url.strip("/")


def nominatim_geocode(addr: str) -> Optional[Dict]:
    sleep()
    try:
        r = requests.get(
            NOMINATIM_URL + "/search",
            params={"q": addr, "format": "jsonv2", "limit": 1, "email": EMAIL},
            headers=HEADERS,
            timeout=30,
        )
        r.raise_for_status()
        arr = r.json()
        return arr[0] if arr else None
    except Exception:
        return None


# ----------------------------
# Models
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
    lon: Optional[float]
    source: str

    def as_dict(self):
        return asdict(self)


# ----------------------------
# Overpass fetch
# ----------------------------
def fetch_osm_veterinary(bbox=BERLIN_BBOX, max_results=5000) -> List[Clinic]:
    print("🔎 Fetching data from OpenStreetMap (amenity=veterinary)…")
    minlon, minlat, maxlon, maxlat = bbox
    query = f"""
    [out:json][timeout:60];
    (
      node["amenity"="veterinary"]({minlat},{minlon},{maxlat},{maxlon});
      way["amenity"="veterinary"]({minlat},{minlon},{maxlat},{maxlon});
      relation["amenity"="veterinary"]({minlat},{minlon},{maxlat},{maxlon});
    );
    out center {max_results};
    """
    sleep()
    r = requests.post(OVERPASS_URL, data={"data": query}, headers=HEADERS, timeout=90)
    r.raise_for_status()
    data = r.json()
    clinics = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name") or ""
        if not name:
            continue
        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lon = el.get("lon") or (el.get("center") or {}).get("lon")
        street = tags.get("addr:street", "")
        housenumber = tags.get("addr:housenumber", "")
        postcode = tags.get("addr:postcode", "")
        website = norm_url(tags.get("website", ""))
        phone = norm_phone(tags.get("phone", ""))
        email = tags.get("email", "")
        clinics.append(
            Clinic(
                name=name,
                website=website,
                phone=phone,
                email=email,
                street=street,
                housenumber=housenumber,
                postcode=postcode,
                city=CITY,
                lat=float(lat) if lat else None,
                lon=float(lon) if lon else None,
                source="osm",
            )
        )
    print(f"✅ Found {len(clinics)} candidates from OSM")
    return clinics


# ----------------------------
# Geocode missing coords
# ----------------------------
def fill_missing_coords(clinics: List[Clinic]) -> List[Clinic]:
    print("🧭 Geocoding missing coordinates…")
    for c in tqdm(clinics):
        if not c.lat or not c.lon:
            if c.street and c.housenumber and c.postcode:
                addr = f"{c.street} {c.housenumber}, {c.postcode} {CITY}, Germany"
                res = nominatim_geocode(addr)
                if res:
                    c.lat = float(res.get("lat"))
                    c.lon = float(res.get("lon"))
    return clinics


# ----------------------------
# Export
# ----------------------------
def export_files(clinics: List[Clinic]):
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "clinics_berlin.csv")
    jsonl_path = os.path.join(OUT_DIR, "clinics_berlin.jsonl")

    # CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(clinics[0]).keys()))
        writer.writeheader()
        for c in clinics:
            writer.writerow(c.as_dict())

    # JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for c in clinics:
            f.write(json.dumps(c.as_dict(), ensure_ascii=False) + "\n")

    print(f"\n✅ Exported CSV → {csv_path}")
    print(f"✅ Exported JSONL → {jsonl_path}\n")


# ----------------------------
# Main
# ----------------------------
def main():
    clinics = fetch_osm_veterinary()
    clinics = fill_missing_coords(clinics)
    clinics = [c for c in clinics if c.lat and c.lon and within_bbox(c.lat, c.lon)]
    export_files(clinics)


if __name__ == "__main__":
    main()

