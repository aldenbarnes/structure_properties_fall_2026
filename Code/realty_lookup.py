"""RealtyAPI lookup for the rent app: find a listing, fetch its details, and pull out units,
location and amenities in the form rent_model.predict_rent takes.

Used by rent_lookup_app.py (Streamlit). Standard library only.

What one lookup costs
  An apartments.com URL or listing key: 1 request (/details/byid).
  A street address: 1 search request (/search/bylocation) to find the listing,
  then 1 /details/byid request for the property you pick. Client caches
  responses, so asking for the same thing twice never calls the API again.

The /details/byid schema has not been checked against a live response from this
code, so units are found by looking for objects with bed/bath/sqft/rent fields
anywhere in the response. If a lookup finds no units, look at the raw response
in the app and adjust UNIT_KEYS below to match the field names you see.
"""
import json
import math
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = "https://apartments.realtyapi.io"
MAX_REQUESTS = 20         # per session; a guard against runaway credit use
SEARCH_RESULTS = 10

# Candidate field names, compared case-insensitively. The first one present wins.
UNIT_KEYS = {
    "beds": ["beds", "bedrooms", "bed", "bedCount", "numberOfBedrooms", "bedroomCount"],
    "baths": ["baths", "bathrooms", "bath", "bathCount", "numberOfBathrooms", "bathroomCount"],
    "sqft": ["sqft", "squareFeet", "squareFootage", "sqFt", "squareFeetMin", "minSquareFeet", "area", "size"],
    "rent": ["rent", "price", "monthlyRent", "rentAmount", "rentPrice", "minRent", "rentMin", "minPrice", "priceMin"],
    "unit": ["unitNumber", "unit", "unitName", "apartmentNumber"],
    "unit_key": ["unitKey", "unitId", "rentalKey", "rentalId"],
    "plan": ["modelName", "floorPlanName", "floorPlan", "planName", "name"],
    "available": ["availableDate", "availability", "availabilityDate", "dateAvailable", "available"],
}

# How train_rent_model turns advertised amenity names into model variables: these groups
# (train_rent_model.AMENITY_GROUPS), plus one variable per other name, lowercased with punctuation
# and spaces turned into underscores ("Tennis Court" -> tennis_court).
AMENITY_GROUPS = {
    "in_unit_laundry": ["washer dryer"], "shared_laundry": ["laundry facilities"],
    "parking": ["parking", "garage parking"], "balcony_patio": ["balcony", "patio"],
    "doorman_concierge": ["doorman", "concierge"],
}


class APIError(Exception):
    pass


class Client:
    """Minimal RealtyAPI client (same request format as realtyapi_notebook.ipynb), with a session cache."""

    def __init__(self, key, max_requests=MAX_REQUESTS, timeout=60):
        self.key = key
        self.max_requests = max_requests
        self.timeout = timeout
        self.requests = 0
        self.cache = {}
        self.log = []      # one entry per call: what was asked and what came back (no key)

    def get(self, endpoint, params):
        cache_key = (endpoint, tuple(sorted(params.items())))
        entry = {"endpoint": endpoint, "params": dict(params), "result": ""}
        self.log.append(entry)
        if cache_key in self.cache:
            entry["result"] = "cached: " + _summary(self.cache[cache_key])
            return self.cache[cache_key]
        if self.requests >= self.max_requests:
            entry["result"] = "not sent: request cap reached"
            raise APIError("Request cap ({}) reached for this session.".format(self.max_requests))
        self.requests += 1
        req = Request(BASE_URL + endpoint + "?" + urlencode(params),
                      headers={"x-realtyapi-key": self.key, "Accept": "application/json"})
        try:
            try:
                with urlopen(req, timeout=self.timeout) as response:
                    data = json.load(response)
            except HTTPError as exc:
                # Do not show response bodies or headers that could contain credentials.
                raise APIError("HTTP {} on {}".format(exc.code, endpoint)) from None
            except (URLError, TimeoutError, OSError, ValueError):
                raise APIError("Network timeout, connection failure, or invalid JSON on " + endpoint) from None
            if not isinstance(data, dict):
                raise APIError("Unexpected response format on " + endpoint)
            message = str(data.get("message", ""))
            if message and not message.lower().startswith("success"):
                raise APIError("API did not report success on {}: {}".format(endpoint, message[:200]))
        except APIError as exc:
            entry["result"] = "error: " + str(exc)
            raise
        entry["result"] = "ok: " + _summary(data)
        time.sleep(0.15)
        self.cache[cache_key] = data
        return data


def _summary(data):
    """One line describing a response: its top-level keys and how many listings it holds."""
    listings = _listing_list(data)
    keys = ", ".join(list(data)[:8]) if isinstance(data, dict) else type(data).__name__
    return "keys [{}]{}".format(keys, "; {} listing(s)".format(len(listings)) if listings is not None else "")


def _listing_list(node, depth=0):
    """The first list of objects with a listingKey anywhere in a response, or None."""
    if depth > 4:
        return None
    if isinstance(node, list):
        if node and all(isinstance(x, dict) and x.get("listingKey") for x in node[:3]):
            return node
        items = node
    elif isinstance(node, dict):
        items = node.values()
    else:
        return None
    for v in items:
        found = _listing_list(v, depth + 1)
        if found is not None:
            return found
    return None


def distance_m(lat1, lon1, lat2, lon2):
    """Approximate distance in meters (fine at city scale)."""
    if None in (lat1, lon1, lat2, lon2):
        return None
    dy = (lat1 - lat2) * 111_000
    dx = (lon1 - lon2) * 111_000 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dx, dy)


# ---------- parsing ----------

def listing_key_from_input(text):
    """Return a listing key if `text` is an apartments.com URL or a bare key, else None."""
    text = text.strip()
    if "apartments.com" in text:
        parts = [p for p in re.split(r"[/?#]", text.split("apartments.com", 1)[1]) if p]
        return parts[-1] if parts else None
    if re.fullmatch(r"[0-9a-z]{6,8}", text) and re.search(r"\d", text):
        return text
    return None


def _num(value, beds=False):
    """Parse 5046, "5046.0", "$5,046", "611 sq ft", "1.5 Baths", "Studio". Ranges give the low end."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, dict):
        for k in ("min", "low", "value", "amount"):
            if k in value:
                return _num(value[k], beds)
        return None
    if not isinstance(value, str):
        return None
    s = value.strip().lower()
    if beds and s.startswith("studio"):
        return 0.0
    m = re.search(r"\d[\d,]*(?:\.\d+)?", s)
    return float(m.group(0).replace(",", "")) if m else None


def _pick(d, names):
    lower = {k.lower(): v for k, v in d.items()}
    for n in names:
        v = lower.get(n.lower())
        if v not in (None, "", [], {}):
            return v
    return None


def _fields(d):
    f = {
        "beds": _num(_pick(d, UNIT_KEYS["beds"]), beds=True),
        "baths": _num(_pick(d, UNIT_KEYS["baths"])),
        "sqft": _num(_pick(d, UNIT_KEYS["sqft"])),
        "rent": _num(_pick(d, UNIT_KEYS["rent"])),
    }
    for k in ("unit", "unit_key", "plan", "available"):
        v = _pick(d, UNIT_KEYS[k])
        f[k] = str(v) if isinstance(v, (str, int, float)) and not isinstance(v, bool) else None
    return f


def _unit_like(f, inherited):
    """A floor plan or unit has its own size fields plus a rent or unit id. A unit nested in a
    floor plan may carry only its id (and rent), taking beds/sqft from the plan."""
    has_id = bool(f["unit"] or f["unit_key"])
    if f["beds"] is not None or f["sqft"] is not None:
        return f["rent"] is not None or has_id
    return has_id and (inherited.get("beds") is not None or inherited.get("sqft") is not None)


def find_units(data):
    """Return unit rows found anywhere in a details response.

    An object with bed/sqft and rent/unit fields is a unit or a floor plan. If it contains
    unit-like objects, those are its units and inherit its beds/baths/sqft/name (floor plan);
    otherwise it is returned as a floor-plan-level row.
    """
    def walk(node, inherited):
        if isinstance(node, list):
            return [r for child in node for r in walk(child, inherited)]
        if not isinstance(node, dict):
            return []
        f = _fields(node)
        if not _unit_like(f, inherited):
            return [r for v in node.values() for r in walk(v, inherited)]
        merged = dict(inherited)
        merged.update({k: v for k, v in f.items() if v is not None})
        child_inherited = {k: merged.get(k) for k in ("beds", "baths", "sqft")}
        child_inherited["plan"] = merged.get("plan")
        children = [r for v in node.values() if isinstance(v, (list, dict)) for r in walk(v, child_inherited)]
        if children:
            return children
        merged["level"] = "unit" if (f["unit"] or f["unit_key"]) else "floor_plan"
        return [merged]

    rows, seen = [], set()
    for r in walk(data, {}):
        key = r.get("unit_key") or (r.get("unit"), r.get("plan"), r.get("beds"), r.get("baths"), r.get("sqft"), r.get("rent"))
        if key not in seen:
            seen.add(key)
            rows.append(r)
    rows.sort(key=lambda r: (r.get("beds") if r.get("beds") is not None else 99, r.get("rent") or 0))
    return rows


def _find_first(node, names):
    """Depth-first search for the first scalar value under any of `names`."""
    wanted = {n.lower() for n in names}
    stack = [node]
    while stack:
        cur = stack.pop(0)
        if isinstance(cur, dict):
            for k, v in cur.items():
                if k.lower() in wanted and isinstance(v, (str, int, float)) and not isinstance(v, bool) and v != "":
                    return v
            stack.extend(v for v in cur.values() if isinstance(v, (dict, list)))
        elif isinstance(cur, list):
            stack.extend(cur)
    return None


def amenity_names(*sources):
    """Collect advertised amenity strings from search results and details responses."""
    names = []

    def strings(node):
        if isinstance(node, str):
            names.append(node)
        elif isinstance(node, list):
            for x in node:
                strings(x)
        elif isinstance(node, dict):
            nested = [k for k in ("groups", "items", "amenities", "amenityNames") if k in node]
            for k in nested or [k for k in ("name", "title") if k in node]:
                strings(node[k])

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if "amenit" in k.lower():
                    strings(v)
                else:
                    walk(v)
        elif isinstance(node, list):
            for x in node:
                walk(x)

    for s in sources:
        walk(s)
    return list(dict.fromkeys(n.strip() for n in names if n and n.strip()))


def _norm(s):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def model_amenities(names, model_names):
    """Map advertised amenity names to rent_model amenity inputs: True/False, or None if no amenity data.

    The model was trained with "not listed" coded as no, so when a listing has an amenity list,
    a model amenity missing from it is set to False.
    """
    have = {_norm(n) for n in names}
    return {m: (any(_norm(t) in have for t in AMENITY_GROUPS.get(m, [m.replace("_", " ")])) if names else None)
            for m in model_names}


def property_summary(search_result, details):
    """Address, zip, lat/lon and name from the search result (known schema), falling back to details."""
    sr = search_result or {}
    addr = sr.get("address") if isinstance(sr.get("address"), dict) else {}
    zip_code = addr.get("postalCode") or _find_first(details, ["postalCode", "zipCode", "zip", "postal_code"])
    one_line = sr.get("oneLineAddress") or _find_first(details, ["oneLineAddress", "fullAddress", "streetAddress", "address"])
    if zip_code is None and isinstance(one_line, str):
        m = re.search(r"\b(9\d{4})\b", one_line)
        zip_code = m.group(1) if m else None
    return {
        "name": sr.get("name") or _find_first(details, ["propertyName", "name"]),
        "address": one_line,
        "zip_code": str(zip_code)[:5] if zip_code is not None else None,
        "latitude": _num(sr.get("latitude") if sr.get("latitude") is not None else _find_first(details, ["latitude", "lat"])),
        "longitude": _num(sr.get("longitude") if sr.get("longitude") is not None else _find_first(details, ["longitude", "lng", "lon"])),
    }


def unit_label(r):
    beds = "?" if r.get("beds") is None else ("Studio" if r["beds"] == 0 else "{:g} bd".format(r["beds"]))
    bits = []
    if r.get("unit"):
        bits.append("Unit " + r["unit"])
    elif r.get("level") == "floor_plan":
        bits.append("Floor plan")
    if r.get("plan"):
        bits.append(r["plan"])
    bits.append("{} / {} ba".format(beds, "?" if r.get("baths") is None else "{:g}".format(r["baths"])))
    bits.append("? sqft" if r.get("sqft") is None else "{:,.0f} sqft".format(r["sqft"]))
    bits.append("rent ?" if r.get("rent") is None else "${:,.0f}".format(r["rent"]))
    if r.get("available"):
        bits.append("avail " + r["available"])
    return " | ".join(bits)


def leaves(value, path="$"):
    """Every scalar leaf with its JSON path, for the raw-response view."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield from leaves(v, path + "." + k)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from leaves(v, path + "[{}]".format(i))
    else:
        yield path, value


# ---------- lookup ----------

def search(client, location, result_count=SEARCH_RESULTS):
    """Return search results (property dicts with listingKey) for an address, neighborhood or zip.

    apartments.com resolves `location` with its typeahead and searches that area, so an address
    can return nearby buildings as well as (or instead of) the building itself.
    """
    data = client.get("/search/bylocation", {"location": location, "page": 1, "resultCount": result_count})
    results = data.get("searchResults")
    if not isinstance(results, list):
        results = _listing_list(data)
    if results is None:
        if data.get("searchResults") is None and any(k in data for k in ("total", "nextPage")):
            return []
        raise APIError("Search response has no listings (top-level keys: {})".format(", ".join(list(data)[:10])))
    return [r for r in results if isinstance(r, dict) and r.get("listingKey")]


def details(client, listing_key):
    return client.get("/details/byid", {"listingKey": listing_key})
