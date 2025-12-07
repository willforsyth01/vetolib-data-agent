#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vetolib Berlin/Brandenburg Crawler — Clinics + Pet Types + Services + Specialists

Scope:
- Berlin + slightly extended region (Teltow / Brandenburg fringe)
- Vets only (OSM amenity=veterinary + Google Places "veterinary_care")
- Exports 5 files in the chosen --output-dir (default: .):

  1) clinics_berlin.csv
     → aligned with public.clinics (subset of columns, but includes stable uuid id)

  2) clinics_berlin.jsonl
     → same columns as CSV, one JSON object per line

  3) clinic_pet_types_berlin.csv
     → columns: clinic_id, pet            (for public.clinic_pet_types)

  4) clinic_services_berlin.csv
     → columns: clinic_id, service_code, label, got_ref,
                price_min_eur, price_max_eur, notes, name, description, icon
        (for public.clinic_services; id/created_at/updated_at come from DB defaults)

  5) clinic_specialists_berlin.csv
     → columns: clinic_id, area           (for public.clinic_specialists)

Google:
- Requires env GOOGLE_API_KEY

Nominatim:
- Uses env NOMINATIM_EMAIL for polite reverse geocoding
"""

import os
import re
import csv
import json
import time
import math
import uuid
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Set

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ----------------------------
# Config / constants
# ----------------------------

CITY_DEFAULT = "Berlin"

# Extended region bbox (Berlin + a bit of Brandenburg / Teltow area)
# (minlon, minlat, maxlon, maxlat)
REGION_BBOX = (12.9, 52.2, 13.9, 52.8)

OVERPASS_URLS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

NOMINATIM_URL = "https://nominatim.openstreetmap.org"

HEADERS = {"User-Agent": "Vetolib-Agent/2.0 (+https://vetolib.app)"}
NOMINATIM_EMAIL = os.getenv("NOMINATIM_EMAIL", "you@example.com")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

# ----------------------------
# Supabase clinics CSV columns
# (subset of public.clinics; includes id, omits created_at/updated_at/etc)
# ----------------------------

CLINIC_COLUMNS = [
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
    "description",
    "emergency_boolean",
    "active_boolean",
    "booking_enabled",
    "supports_mobile",
    "twentyfour_seven",
    "offers_vaccination",
    "offers_checkup",
    "offers_illness",
    "offers_prescription",
    "offers_dental",
    "contact_email",
    "onboarding_status",
]

PET_TYPES_COLUMNS = [
    "clinic_id",
    "pet",  # public.pet_type enum
]

CLINIC_SERVICES_COLUMNS = [
    "clinic_id",
    "service_code",
    "label",
    "got_ref",
    "price_min_eur",
    "price_max_eur",
    "notes",
    "name",
    "description",
    "icon",
]

CLINIC_SPECIALISTS_COLUMNS = [
    "clinic_id",
    "area",  # public.specialist_area enum
]

# ----------------------------
# Helpers (general)
# ----------------------------

def within_bbox(lat: float, lon: float, bbox: Tuple[float, float, float, float]) -> bool:
    """Check if a point is within bbox (minlon, minlat, maxlon, maxlat)."""
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
    # Keep digits and leading +
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
            if resp.status_code in (429, 502, 503, 504):
                last_err = Exception(f"{resp.status_code} (attempt {i+1}/{tries})")
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

def haversine_m(lat1, lon1, lat2, lon2):
    """Distance in meters."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# Deterministic UUID for clinics (so clinic_id is stable across runs)
def clinic_uuid(name: str, street: str, postcode: str, city: str) -> str:
    base = f"{(name or '').strip().lower()}|{(street or '').strip().lower()}|{(postcode or '').strip()}|{(city or '').strip().lower()}"
    if not base.strip("|"):
        base = "vetolib-unnamed"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, base))

# ----------------------------
# Opening-hours + emergency helpers
# ----------------------------

DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

DAY_ALIASES = {
    "mo": "mon", "mon": "mon", "monday": "mon", "montag": "mon",
    "tu": "tue", "tue": "tue", "tues": "tue", "tuesday": "tue", "dienstag": "tue",
    "we": "wed", "wed": "wed", "weds": "wed", "wednesday": "wed", "mittwoch": "wed",
    "th": "thu", "thu": "thu", "thur": "thu", "thurs": "thu",
    "thursday": "thu", "donnerstag": "thu",
    "fr": "fri", "fri": "fri", "friday": "fri", "freitag": "fri",
    "sa": "sat", "sat": "sat", "saturday": "sat", "samstag": "sat",
    "su": "sun", "sun": "sun", "sunday": "sun", "sonntag": "sun",
}

SPECIAL_DAY_WORDS = {
    "täglich": DAYS,
    "taeglich": DAYS,
    "daily": DAYS,
    "werktags": ["mon", "tue", "wed", "thu", "fri"],
    "wochentags": ["mon", "tue", "wed", "thu", "fri"],
}

# Focus emergency terms on real out-of-hours / Notdienst semantics
EMERGENCY_KWS = [
    "notdienst",
    "notfall",
    "notfälle",
    "notfallmedizin",
    "tierärztlicher notdienst",
    "tierarzt notdienst",
    "tiernotdienst",
    "emergency",
    "24/7",
    "24 h",
    "24h",
    "24-stunden",
    "24 stunden",
    "rund um die uhr",
]

def looks_247(s: str) -> bool:
    s = (s or "").lower()
    return any(k in s for k in ["24/7", "24 h", "24h", "rund um die uhr"])

def parse_day_token(token: str) -> List[str]:
    t = token.strip().lower().rstrip(".")
    if not t:
        return []
    if t in SPECIAL_DAY_WORDS:
        return SPECIAL_DAY_WORDS[t]
    if "-" in t:
        a, b = [x.strip() for x in t.split("-", 1)]
        a = DAY_ALIASES.get(a, None)
        b = DAY_ALIASES.get(b, None)
        if not a or not b:
            return []
        ai = DAYS.index(a)
        bi = DAYS.index(b)
        if ai <= bi:
            return DAYS[ai : bi + 1]
        return DAYS[ai:] + DAYS[: bi + 1]
    if t in DAY_ALIASES:
        return [DAY_ALIASES[t]]
    return []

def parse_days_expr(expr: str) -> List[str]:
    parts = re.split(r"[,\s]+", expr)
    days: List[str] = []
    for p in parts:
        if not p:
            continue
        days.extend(parse_day_token(p))
    # dedupe, preserve order
    seen = set()
    ordered = []
    for d in days:
        if d not in seen:
            seen.add(d)
            ordered.append(d)
    return ordered

def parse_time_ranges(expr: str) -> List[List[float]]:
    expr = expr.replace("–", "-").replace("—", "-")
    ranges: List[List[float]] = []
    for m in re.finditer(
        r"(\d{1,2})(?::|\.|h)?(\d{2})?\s*-\s*(\d{1,2})(?::|\.|h)?(\d{2})?",
        expr,
    ):
        h1, m1, h2, m2 = m.groups()
        m1 = m1 or "00"
        m2 = m2 or "00"
        start = int(h1) + int(m1) / 60.0
        end = int(h2) + int(m2) / 60.0
        if end == 0:
            end = 24.0
        if 0 <= start < 24 and 0 < end <= 24 and end > start:
            ranges.append([round(start, 2), round(end, 2)])
    return ranges

def parse_opening_hours_to_struct(raw: str) -> Dict[str, List[List[float]]]:
    """
    Heuristic parser for structured hours from a free-text / OSM / Google style string.
    """
    res: Dict[str, List[List[float]]] = {d: [] for d in DAYS}
    if not raw or not isinstance(raw, str):
        return res

    txt = raw.strip()
    low = txt.lower()

    if any(k in low for k in ["24/7", "24 h", "24h", "rund um die uhr"]):
        for d in DAYS:
            res[d] = [[0.0, 24.0]]
        return res

    txt = txt.replace("\n", "; ")
    txt = txt.replace("–", "-").replace("—", "-")
    txt = re.sub(r"\s+", " ", txt)

    pattern = re.compile(
        r"(?P<days>(?:[A-Za-zÄÖÜäöü\.]{2,}(?:\s*[-,]\s*)?)+)\s*[: ]\s*"
        r"(?P<times>\d{1,2}[:\.h]?\d{0,2}\s*-\s*\d{1,2}[:\.h]?\d{0,2}"
        r"(?:\s*,\s*\d{1,2}[:\.h]?\d{0,2}\s*-\s*\d{1,2}[:\.h]?\d{0,2})*)"
    )

    for m in pattern.finditer(txt):
        days_expr = m.group("days")
        times_expr = m.group("times")
        days = parse_days_expr(days_expr)
        ranges = parse_time_ranges(times_expr)
        if not days or not ranges:
            continue
        for d in days:
            res[d].extend(ranges)

    return res

def infer_is_247(opening_raw: str, struct: Dict[str, List[List[float]]]) -> bool:
    low = (opening_raw or "").lower()
    if any(k in low for k in ["24/7", "24 h", "24h", "rund um die uhr"]):
        return True
    if struct:
        all_full = True
        for d in DAYS:
            slots = struct.get(d) or []
            if not slots:
                all_full = False
                break
            if not any(s <= 0 and e >= 24 for s, e in slots):
                all_full = False
                break
        if all_full:
            return True
    return False

def infer_emergency_from_hours(struct: Dict[str, List[List[float]]]) -> bool:
    """
    Heuristic: emergency if any interval goes late (after 22:00) or very early (before 07:00).
    """
    for slots in (struct or {}).values():
        for s, e in slots:
            if e > 22.0 or s < 7.0:
                return True
    return False

def derive_emergency(opening_hours: str, name: str, extra: str = "", types: Optional[List[str]] = None) -> bool:
    """
    Conservative emergency detection:
    - Emergency keywords in text (Notdienst, 24/7 etc.)
    - Google types indicating emergency
    """
    blob = " ".join([
        opening_hours or "",
        name or "",
        extra or "",
    ]).lower()

    if any(kw in blob for kw in EMERGENCY_KWS):
        return True

    if types:
        t_low = {t.lower() for t in types}
        if any(t in t_low for t in ["animal_hospital", "emergency_service", "hospital"]):
            return True

    return False

def extract_hours_text_from_html(html: str) -> str:
    """
    Pulls a compact hours string from the page:
    - prefer schema.org openingHours / openingHoursSpecification
    - else headings/labels like Öffnungszeiten, Sprechzeiten, Opening Hours
    """
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return ""

    # 1) JSON-LD (schema.org)
    for tag in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            j = json.loads(tag.string or "{}")
            items = j if isinstance(j, list) else [j]
            for o in items:
                if not isinstance(o, dict):
                    continue
                oh = o.get("openingHours") or o.get("openingHoursSpecification")
                if isinstance(oh, list):
                    txt = "; ".join([str(x) for x in oh])
                    if txt.strip():
                        return txt
                elif isinstance(oh, str):
                    if oh.strip():
                        return oh
        except Exception:
            pass

    # 2) Label-based heuristics
    labels = ["öffnungszeiten", "sprechzeiten", "sprechstunde", "opening hours", "hours"]
    blocks: List[str] = []
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
# Pet types & specialist detection
# ----------------------------

PET_TYPE_MAP = {
    "dog": ["hund", "hunde", "dog", "dogs"],
    "cat": ["katze", "katzen", "kater", "cat", "cats"],
    "rabbit": ["kaninchen", "kaninchenpraxis", "rabbit", "rabbits", "hase", "hasen"],
    "small_mammal": [
        "kleintier", "kleintiere", "nagetier", "nagetiere",
        "meerschweinchen", "hamster", "chinchilla", "ratte", "ratten",
        "maus", "mäuse", "gerbil",
    ],
    "bird": ["vogel", "vögel", "papagei", "papageien", "sittich", "sittiche", "kanarienvogel", "kanarien"],
    "reptile": ["reptil", "reptilien", "schildkröte", "schildkröten", "schlange", "schlangen", "echse", "echsen", "leguan"],
    "amphibian": ["amphib", "frosch", "frösche", "molch", "salamander"],
    "ferret": ["frettchen", "ferret", "ferrets"],
    "exotic": ["exotische", "exoten", "exotic"],
}

SPECIALIST_MAP = {
    "dentistry": ["zahn", "zahnheilkunde", "zahnarzt", "dental"],
    "dermatology": ["dermatologie", "haut", "hautkrankheit", "hautprobleme", "skin"],
    "orthopedics": ["orthopädie", "orthopaedie", "orthopädisch"],
    "cardiology": ["kardiologie", "cardiology", "herz", "herzerkrankung"],
    "ophthalmology": ["augenheilkunde", "ophthalmologie", "auge", "augen"],
    "physio_rehab": ["physiotherapie", "rehabilitation", "rehab", "physio"],
    "oncology": ["onkologie", "oncology", "krebs", "tumor", "tumour"],
    "neurology": ["neurologie", "neurologisch", "nervensystem"],
    "behavior": ["verhaltenstherapie", "verhalten", "behavior", "training"],
    "nutrition": ["ernährungsberatung", "ernährung", "nutrition", "diet"],
    "imaging": ["röntgen", "roentgen", "x-ray", "ultraschall", "ultrasound", "ct", "mrt", "mri"],
    "surgery": ["chirurgie", "operation", "operativ", "chirurg", "op "],
    "exotics": ["exoten", "exotenmedizin", "exotic"],
}

SERVICE_LABELS = {
    "vaccination": "Vaccination",
    "checkup": "Check-up / General exam",
    "illness": "Illness / Treatment",
    "prescription": "Prescriptions",
    "dental": "Dental care",
    "surgery": "Surgery",
    "imaging": "Imaging (X-ray / US)",
    "emergency": "Emergency care",
    "home_visit": "Home visits",
    "mobile_vet": "Mobile vet",
    "tele_vet": "Tele-vet / Online consult",
    "lab": "Lab diagnostics",
    "ultrasound": "Ultrasound",
    "xray": "X-ray",
    "endoscopy": "Endoscopy",
    "physio_therapy": "Physiotherapy / Rehab",
    "behavior": "Behaviour therapy",
    "nutrition": "Nutrition counselling",
    "exotics": "Exotic animal care",
    "dermatology": "Dermatology",
    "cardiology": "Cardiology",
    "oncology": "Oncology",
    "orthopedics": "Orthopedics",
    "ophthalmology": "Ophthalmology",
    "neurology": "Neurology",
}

def detect_pet_types(text: str) -> Set[str]:
    text_l = (text or "").lower()
    found: Set[str] = set()
    for pet, kws in PET_TYPE_MAP.items():
        for kw in kws:
            if kw in text_l:
                found.add(pet)
                break
    # Default dog+cat if nothing else detected
    if not found:
        found.update(["dog", "cat"])
    return found

def detect_specialists(text: str) -> Set[str]:
    text_l = (text or "").lower()
    found: Set[str] = set()
    for area, kws in SPECIALIST_MAP.items():
        for kw in kws:
            if kw in text_l:
                found.add(area)
                break
    return found

def detect_services(text: str, has_emergency: bool) -> Set[str]:
    text_l = (text or "").lower()
    services: Set[str] = set()

    # Core services for almost every clinic
    services.update(["vaccination", "checkup", "illness", "prescription"])

    # Dental
    if any(kw in text_l for kw in ["zahn", "dental", "zahnarzt", "zahnheilkunde"]):
        services.add("dental")

    # Surgery
    if any(kw in text_l for kw in ["chirurgie", "operation", "operativ", "chirurg", "op "]):
        services.add("surgery")

    # Imaging
    if any(kw in text_l for kw in ["röntgen", "roentgen", "x-ray", "ultraschall", "ultrasound", "ct", "mrt", "mri"]):
        services.add("imaging")
        if "ultraschall" in text_l or "ultrasound" in text_l:
            services.add("ultrasound")
        if "röntgen" in text_l or "x-ray" in text_l:
            services.add("xray")

    # Emergency
    if has_emergency or any(kw in text_l for kw in EMERGENCY_KWS):
        services.add("emergency")

    # Home visits / mobile vet
    if any(kw in text_l for kw in ["hausbesuch", "hausbesuche", "home visit", "house call"]):
        services.add("home_visit")
        services.add("mobile_vet")
    if any(kw in text_l for kw in ["mobile tierarzt", "mobile praxis", "fahrpraxis"]):
        services.add("mobile_vet")

    # Tele-vet
    if any(kw in text_l for kw in ["video", "online-sprechstunde", "telefonsprechstunde", "online-termin", "telemedizin"]):
        services.add("tele_vet")

    # Lab / diagnostics
    if any(kw in text_l for kw in ["labor", "labordiagnostik", "blutuntersuchung", "diagnostik", "laboratory"]):
        services.add("lab")

    # Physio, behaviour, nutrition, exotics (from text)
    if any(kw in text_l for kw in ["physiotherapie", "rehab", "rehabilitation", "physio"]):
        services.add("physio_therapy")
    if any(kw in text_l for kw in ["verhaltenstherapie", "verhalten", "behavior"]):
        services.add("behavior")
    if any(kw in text_l for kw in ["ernährungsberatung", "ernährung", "nutrition", "diet"]):
        services.add("nutrition")
    if any(kw in text_l for kw in ["exoten", "exotenmedizin", "exotic"]):
        services.add("exotics")

    # Map specialist areas to services as well
    spec = detect_specialists(text)
    for s in spec:
        if s in SERVICE_LABELS:
            services.add(s)

    return services

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
    offers_checkup: bool = True
    offers_illness: bool = True
    offers_prescription: bool = True
    offers_dental: bool = False
    offers_vaccination: bool = True
    pet_types: Set[str] = field(default_factory=set)
    specialists: Set[str] = field(default_factory=set)
    service_codes: Set[str] = field(default_factory=set)
    source: str = "osm"  # "osm" or "google"

# ----------------------------
# OSM fetch
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

        if not lat or not lng:
            continue
        if not within_bbox(lat, lng, REGION_BBOX):
            continue

        street = tags.get("addr:street", "") or ""
        postcode = tags.get("addr:postcode", "") or ""
        district = tags.get("addr:suburb", "") or ""

        website = norm_url(tags.get("website", "") or tags.get("contact:website", ""))
        phone = norm_phone(tags.get("phone", "") or tags.get("contact:phone", ""))
        email = tags.get("email", "") or tags.get("contact:email", "")

        # Opening hours
        opening_hours = ""
        raw_oh = (tags.get("opening_hours") or "").strip()
        if looks_247(raw_oh):
            opening_hours = "24/7"
        elif raw_oh:
            opening_hours = raw_oh

        html = ""
        if enrich_websites and website:
            html = fetch_site(website)

        if not opening_hours and html:
            candidate = extract_hours_text_from_html(html)
            if looks_247(candidate):
                opening_hours = "24/7"
            elif candidate:
                opening_hours = candidate[:400]

        # Emergency based on text + name
        is_emergency = derive_emergency(
            opening_hours,
            name,
            (tags.get("description") or "") + " " + (tags.get("services") or ""),
            types=None,
        )

        is_247 = looks_247(opening_hours)

        blob = " ".join([
            name,
            tags.get("description", "") or "",
            tags.get("services", "") or "",
            opening_hours or "",
            (html[:2000] if html else ""),
        ])

        pet_types = detect_pet_types(blob)
        specialists = detect_specialists(blob)
        services = detect_services(blob, is_emergency or is_247)

        supports_mobile = any(kw in blob.lower() for kw in ["hausbesuch", "mobile tierarzt", "fahrpraxis"])

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
            supports_mobile=supports_mobile,
            offers_checkup=True,
            offers_illness=True,
            offers_prescription=True,
            offers_dental=("dental" in services),
            offers_vaccination=True,
            pet_types=pet_types,
            specialists=specialists,
            service_codes=services,
            source="osm",
        )

        clinics.append(c)

    print(f"✅ OSM clinics in region: {len(clinics)}")
    return clinics

# ----------------------------
# Google Places
# ----------------------------

def google_nearby_for_tile(center_lat: float, center_lng: float, api_key: str) -> List[Dict[str, Any]]:
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{center_lat},{center_lng}",
        "radius": 12000,
        "type": "veterinary_care",
        "key": api_key,
    }
    results = []
    page_token = None
    for _ in range(5):
        local_params = params.copy()
        if page_token:
            local_params["pagetoken"] = page_token
            time.sleep(2.1)
        data = http_json_with_retries(
            "GET", url, params=local_params,
            headers=HEADERS, timeout=40, tries=4, polite_delay=1.5
        )
        batch = data.get("results", [])
        results.extend(batch)
        page_token = data.get("next_page_token")
        if not page_token:
            break
    return results

def google_text_search_for_city(query: str, api_key: str) -> List[Dict[str, Any]]:
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": query,
        "region": "de",
        "key": api_key,
    }
    results = []
    page_token = None
    for _ in range(5):
        local_params = params.copy()
        if page_token:
            local_params["pagetoken"] = page_token
            time.sleep(2.1)
        data = http_json_with_retries(
            "GET", url, params=local_params,
            headers=HEADERS, timeout=40, tries=4, polite_delay=1.5
        )
        batch = data.get("results", [])
        results.extend(batch)
        page_token = data.get("next_page_token")
        if not page_token:
            break
    return results

def google_fetch_all_places(bbox: Tuple[float,float,float,float], tiles_per_side: int, api_key: str) -> Dict[str, Dict[str, Any]]:
    if not api_key:
        print("⚠️ GOOGLE_API_KEY not set; skipping Google Places.")
        return {}

    tiles = tile_bbox(bbox, tiles_per_side)
    places_by_id: Dict[str, Dict[str, Any]] = {}

    print("🔎 Fetching clinics from Google Places (Nearby Search)…")
    for i, tb in enumerate(tiles, start=1):
        minlon, minlat, maxlon, maxlat = tb
        center_lat = (minlat + maxlat) / 2.0
        center_lng = (minlon + maxlon) / 2.0
        print(f"  • Nearby tile {i}/{len(tiles)} (center {center_lat:.5f},{center_lng:.5f})")
        try:
            results = google_nearby_for_tile(center_lat, center_lng, api_key)
            for r in results:
                pid = r.get("place_id")
                if not pid:
                    continue
                types = r.get("types", []) or []
                if "veterinary_care" not in types:
                    continue
                if pid not in places_by_id:
                    places_by_id[pid] = r
        except Exception as e:
            print(f"    ⚠️ Google Nearby tile {i} failed: {e}")
            continue

    print("🔎 Fetching clinics from Google Places (Text Search)…")
    text_queries = [
        "Tierarzt Berlin",
        "Tierärztliche Klinik Berlin",
        "Tierarztpraxis Berlin",
        "Tierklinik Berlin",
    ]
    for q in text_queries:
        print(f"  • Text search: {q!r}")
        try:
            results = google_text_search_for_city(q, api_key)
            for r in results:
                pid = r.get("place_id")
                if not pid:
                    continue
                types = r.get("types", []) or []
                if "veterinary_care" not in types:
                    continue
                if pid not in places_by_id:
                    places_by_id[pid] = r
        except Exception as e:
            print(f"    ⚠️ Text search {q!r} failed: {e}")
            continue

    print(f"✅ Unique Google Places (veterinary_care) found: {len(places_by_id)}")
    return places_by_id

def google_place_details_to_clinic(place_id: str, basic: Dict[str, Any], api_key: str, city_fallback: str) -> Optional[Clinic]:
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    fields = "name,formatted_address,address_components,formatted_phone_number,geometry,website,opening_hours,types"
    params = {"place_id": place_id, "fields": fields, "key": api_key}

    try:
        data = http_json_with_retries("GET", url, params=params, headers=HEADERS, timeout=40, tries=4, polite_delay=1.5)
    except Exception as e:
        print(f"    ⚠️ Details failed for {place_id}: {e}")
        return None

    result = data.get("result") or {}
    name = (result.get("name") or basic.get("name") or "").strip()
    if not name:
        return None

    geom = result.get("geometry", {}).get("location", {})
    lat = geom.get("lat")
    lng = geom.get("lng")
    try:
        lat = float(lat) if lat is not None else None
        lng = float(lng) if lng is not None else None
    except Exception:
        lat, lng = None, None

    if not lat or not lng:
        return None
    if not within_bbox(lat, lng, REGION_BBOX):
        return None

    street = ""
    postcode = ""
    district = ""
    city = city_fallback

    for comp in result.get("address_components", []):
        types = comp.get("types", [])
        long_name = comp.get("long_name", "")
        if "route" in types:
            street = long_name
        elif "street_number" in types:
            if street:
                street = f"{street} {long_name}"
            else:
                street = long_name
        elif "postal_code" in types:
            postcode = long_name
        elif any(t in types for t in ["sublocality", "sublocality_level_1", "neighborhood"]):
            district = long_name
        elif "locality" in types:
            city = long_name

    phone = norm_phone(result.get("formatted_phone_number") or "")
    website = norm_url(result.get("website") or "")
    email = ""  # Google doesn't expose email

    opening_hours = ""
    oh = result.get("opening_hours") or {}
    weekday_text = oh.get("weekday_text")
    if isinstance(weekday_text, list) and weekday_text:
        opening_hours = "; ".join(weekday_text)

    types_all = basic.get("types", []) or result.get("types", []) or []

    is_247 = looks_247(opening_hours)
    is_emergency = derive_emergency(opening_hours, name, "", types=types_all)

    blob = " ".join([name] + types_all + [opening_hours or ""])

    pet_types = detect_pet_types(blob)
    specialists = detect_specialists(blob)
    services = detect_services(blob, is_emergency or is_247)

    supports_mobile = any(kw in blob.lower() for kw in ["hausbesuch", "mobile tierarzt", "fahrpraxis"])

    c = Clinic(
        name=name,
        street=street,
        district=district,
        city=city or city_fallback,
        postcode=postcode,
        lat=lat,
        lng=lng,
        phone=phone,
        email=email,
        website=website,
        opening_hours=opening_hours or "",
        emergency_flag=is_emergency or is_247,
        supports_mobile=supports_mobile,
        offers_checkup=True,
        offers_illness=True,
        offers_prescription=True,
        offers_dental=("dental" in services),
        offers_vaccination=True,
        pet_types=pet_types,
        specialists=specialists,
        service_codes=services,
        source="google",
    )
    return c

def fetch_google_clinics(bbox: Tuple[float,float,float,float], tiles_per_side: int, city: str) -> List[Clinic]:
    if not GOOGLE_API_KEY:
        print("⚠️ GOOGLE_API_KEY not set; skipping Google Places entirely.")
        return []

    basic_places = google_fetch_all_places(bbox, tiles_per_side, GOOGLE_API_KEY)
    clinics: List[Clinic] = []
    print("🔎 Fetching Google Place details…")
    for i, (pid, basic) in enumerate(basic_places.items(), start=1):
        if i % 25 == 0:
            print(f"  • Details {i}/{len(basic_places)}")
        c = google_place_details_to_clinic(pid, basic, GOOGLE_API_KEY, city)
        if not c:
            continue
        clinics.append(c)
    print(f"✅ Google clinics with details in region: {len(clinics)}")
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
        "email": NOMINATIM_EMAIL,
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
        need_street = not bool((c.street or "").strip())
        need_pc = not bool((c.postcode or "").strip())
        need_district = not bool((c.district or "").strip())
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
# Merge OSM + Google
# ----------------------------

def norm_name_for_match(name: str) -> str:
    n = (name or "").lower()
    n = re.sub(
        r"\b(dr\.?|doktor|tierarztpraxis|tierarzt|tierärztin|tierklinik|praxis|kleintierpraxis)\b",
        "",
        n,
    )
    n = re.sub(r"[^a-z0-9]+", "", n)
    return n.strip()

def merge_osm_and_google(osm_clinics: List[Clinic], google_clinics: List[Clinic]) -> List[Clinic]:
    if not google_clinics:
        return osm_clinics

    merged: List[Clinic] = list(osm_clinics)
    print("🔗 Merging OSM + Google clinics…")

    for g in google_clinics:
        if not g.lat or not g.lng:
            merged.append(g)
            continue

        g_norm = norm_name_for_match(g.name)
        best_idx = None
        best_dist = 999999.0

        for idx, o in enumerate(merged):
            if not o.lat or not o.lng:
                continue
            d = haversine_m(o.lat, o.lng, g.lat, g.lng)
            if d > 200:
                continue
            o_norm = norm_name_for_match(o.name)
            if not g_norm or not o_norm:
                continue
            if g_norm == o_norm or g_norm in o_norm or o_norm in g_norm:
                if d < best_dist:
                    best_dist = d
                    best_idx = idx

        if best_idx is not None:
            o = merged[best_idx]
            # Name: trust Google if present
            o.name = g.name or o.name

            # Prefer Google contact if missing
            if not o.phone and g.phone:
                o.phone = g.phone
            if not o.website and g.website:
                o.website = g.website

            # Address fill
            if not o.street and g.street:
                o.street = g.street
            if not o.postcode and g.postcode:
                o.postcode = g.postcode
            if not o.district and g.district:
                o.district = g.district

            # Opening hours: prefer longer text
            if len(g.opening_hours or "") > len(o.opening_hours or ""):
                o.opening_hours = g.opening_hours

            # Merge flags & enrichments
            o.emergency_flag = o.emergency_flag or g.emergency_flag
            o.supports_mobile = o.supports_mobile or g.supports_mobile

            o.pet_types |= g.pet_types
            o.specialists |= g.specialists
            o.service_codes |= g.service_codes

            if "dental" in o.service_codes:
                o.offers_dental = True
        else:
            merged.append(g)

    print(f"✅ Merged clinic count: {len(merged)}")
    return merged

# ----------------------------
# Export helpers
# ----------------------------

def clinic_to_row(c: Clinic, cid: str) -> Dict[str, Any]:
    opening = c.opening_hours or ""

    struct = parse_opening_hours_to_struct(opening)
    is_247 = infer_is_247(opening, struct)

    hours_emergency = infer_emergency_from_hours(struct)
    text_emergency = derive_emergency(opening, c.name, "", None)
    is_emergency = bool(c.emergency_flag) or hours_emergency or text_emergency

    opening_obj = {
        "raw": opening,
        "structured": struct,
        "is_247": is_247,
        "emergency": is_emergency,
        "source": c.source,
    }

    return {
        "id": cid,
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
        "opening_hours": json.dumps(opening_obj, ensure_ascii=False),
        "emergency": bool(is_emergency),
        "active": True,
        "description": "",
        "emergency_boolean": bool(is_emergency),
        "active_boolean": True,
        "booking_enabled": False,
        "supports_mobile": bool(c.supports_mobile),
        "twentyfour_seven": bool(is_247),
        "offers_vaccination": True,
        "offers_checkup": True,
        "offers_illness": True,
        "offers_prescription": True,
        "offers_dental": bool(c.offers_dental),
        "contact_email": c.email or "",
        "onboarding_status": "not_invited",
    }

def export_all(clinics: List[Clinic], out_dir: str, city_slug: str):
    os.makedirs(out_dir, exist_ok=True)

    clinics_csv_path = os.path.join(out_dir, f"clinics_{city_slug}.csv")
    clinics_jsonl_path = os.path.join(out_dir, f"clinics_{city_slug}.jsonl")
    pet_types_path = os.path.join(out_dir, f"clinic_pet_types_{city_slug}.csv")
    services_path = os.path.join(out_dir, f"clinic_services_{city_slug}.csv")
    specialists_path = os.path.join(out_dir, f"clinic_specialists_{city_slug}.csv")

    # Generate deterministic ids for all clinics
    id_map: Dict[Clinic, str] = {}
    rows: List[Dict[str, Any]] = []

    for c in clinics:
        cid = clinic_uuid(c.name, c.street, c.postcode, c.city)
        id_map[c] = cid
        rows.append(clinic_to_row(c, cid))

    # Clinics CSV
    with open(clinics_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CLINIC_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CLINIC_COLUMNS})

    # Clinics JSONL
    with open(clinics_jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r.get(k, "") for k in CLINIC_COLUMNS}, ensure_ascii=False) + "\n")

    # Pet types
    with open(pet_types_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PET_TYPES_COLUMNS)
        w.writeheader()
        for c in clinics:
            cid = id_map[c]
            for pet in sorted(c.pet_types):
                w.writerow({
                    "clinic_id": cid,
                    "pet": pet,
                })

    # Services
    with open(services_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CLINIC_SERVICES_COLUMNS)
        w.writeheader()
        for c in clinics:
            cid = id_map[c]
            codes = set(c.service_codes)
            if c.emergency_flag:
                codes.add("emergency")
            for code in sorted(codes):
                label = SERVICE_LABELS.get(code, code.replace("_", " ").title())
                w.writerow({
                    "clinic_id": cid,
                    "service_code": code,
                    "label": label,
                    "got_ref": "",
                    "price_min_eur": "",
                    "price_max_eur": "",
                    "notes": "",
                    "name": label,
                    "description": "",
                    "icon": "",
                })

    # Specialists
    with open(specialists_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CLINIC_SPECIALISTS_COLUMNS)
        w.writeheader()
        for c in clinics:
            cid = id_map[c]
            for area in sorted(c.specialists):
                w.writerow({
                    "clinic_id": cid,
                    "area": area,
                })

    print(f"✅ Exported clinics CSV → {clinics_csv_path}")
    print(f"✅ Exported clinics JSONL → {clinics_jsonl_path}")
    print(f"✅ Exported clinic_pet_types CSV → {pet_types_path}")
    print(f"✅ Exported clinic_services CSV → {services_path}")
    print(f"✅ Exported clinic_specialists CSV → {specialists_path}")

# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default=CITY_DEFAULT)
    parser.add_argument("--max-results", type=int, default=5000)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--tiles", type=int, default=2)
    parser.add_argument("--no-geocode", action="store_true")
    parser.add_argument("--enrich-websites", action="store_true")
    parser.add_argument("--use-google-places", action="store_true", help="Also fetch clinics from Google Places API")
    args = parser.parse_args()

    city = args.city
    city_slug = re.sub(r"[^a-z0-9]+", "-", city.lower()).strip("-") or "city"
    bbox = REGION_BBOX

    tiles = tile_bbox(bbox, args.tiles)
    print(f"📦 Querying {len(tiles)} tiles from OSM…")

    osm_all: List[Clinic] = []
    for i, tb in enumerate(tiles, start=1):
        print(f"— OSM tile {i}/{len(tiles)}")
        osm_all.extend(fetch_osm_veterinary(tb, args.max_results, city, args.enrich_websites))

    # Deduplicate OSM by (name, lat, lng)
    seen = set()
    osm_uniq: List[Clinic] = []
    for c in osm_all:
        key = (c.name.lower().strip(), round(c.lat or 0, 5), round(c.lng or 0, 5))
        if key in seen:
            continue
        seen.add(key)
        osm_uniq.append(c)

    google_clinics: List[Clinic] = []
    if args.use_google_places:
        google_clinics = fetch_google_clinics(bbox, args.tiles, city)
    else:
        print("ℹ️ Skipping Google Places (flag --use-google-places not set).")

    merged = merge_osm_and_google(osm_uniq, google_clinics)

    if not args.no_geocode:
        merged = fill_missing_address(merged)

    # Final in-bbox filter to be safe
    merged = [c for c in merged if c.lat and c.lng and within_bbox(c.lat, c.lng, bbox)]

    print(f"🧹 Final clinic count after merge/dedupe: {len(merged)}")

    export_all(merged, args.output_dir, city_slug)

if __name__ == "__main__":
    main()
