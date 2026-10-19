"""Look up an SF apartment on RealtyAPI, edit its features, and estimate rent with rent_model.

Run in Colab or Jupyter (needs ipywidgets, which Colab already has):

    import rent_lookup_tool
    rent_lookup_tool.open_tool(api_key)   # api_key=None: type features in by hand

What one lookup costs
  An apartments.com URL or listing key: 1 request (/details/byid).
  A street address: 1 search request (/search/bylocation) to find the listing,
  then 1 /details/byid request for the property you pick. Responses are cached
  for the session, so changing units or features never calls the API again.

Every feature the lookup fills in can be changed before estimating. A blank box
means "not known"; rent_model then uses its training average and says so.

The /details/byid schema has not been checked against a live response from this
code, so units are found by looking for objects with bed/bath/sqft/rent fields
anywhere in the response. If a lookup finds no units, open "Raw response" and
adjust UNIT_KEYS below to match the field names you see.
"""
import json
import math
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import rent_model

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

# Advertised amenity names -> model amenities, as coded in train_rent_model.AMENITY_GROUPS.
# Names are compared after lowercasing and dropping punctuation ("Washer/Dryer" = "washer dryer").
MODEL_AMENITIES = {
    "in_unit_laundry": ["Washer Dryer", "Washer Dryer In Unit", "In Unit Washer Dryer"],
    "shared_laundry": ["Laundry Facilities"],
    "doorman_concierge": ["Doorman", "Concierge"],
}
AMENITY_LABELS = {"in_unit_laundry": "In-unit laundry", "shared_laundry": "Shared laundry",
                  "doorman_concierge": "Doorman / concierge"}


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

    def get(self, endpoint, params):
        cache_key = (endpoint, tuple(sorted(params.items())))
        if cache_key in self.cache:
            return self.cache[cache_key]
        if self.requests >= self.max_requests:
            raise APIError("Request cap ({}) reached for this session.".format(self.max_requests))
        self.requests += 1
        req = Request(BASE_URL + endpoint + "?" + urlencode(params),
                      headers={"x-realtyapi-key": self.key, "Accept": "application/json"})
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
        time.sleep(0.15)
        self.cache[cache_key] = data
        return data


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


def model_amenities(names):
    """Map advertised amenity names to model inputs: True/False, or None if no amenity data at all.

    The model was trained with "not listed" coded as no, so when a listing has an amenity list,
    a model amenity missing from it is set to False.
    """
    have = {_norm(n) for n in names}
    out = {}
    for model_name, tokens in MODEL_AMENITIES.items():
        out[model_name] = (any(_norm(t) in have for t in tokens)) if names else None
    return out


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

def search(client, address):
    """Return search results (property dicts with listingKey) for an address."""
    data = client.get("/search/bylocation", {"location": address, "page": 1, "resultCount": SEARCH_RESULTS})
    results = data.get("searchResults")
    if not isinstance(results, list):
        raise APIError("Search response is missing searchResults")
    return [r for r in results if isinstance(r, dict) and r.get("listingKey")]


def details(client, listing_key):
    return client.get("/details/byid", {"listingKey": listing_key})


# ---------- form ----------

def _fmt(x):
    if x is None:
        return ""
    return "{:.6f}".format(x).rstrip("0").rstrip(".") if isinstance(x, float) else str(x)


def _parse_box(text, name, errors, lo=None, hi=None):
    text = (text or "").strip().replace(",", "").replace("$", "")
    if not text:
        return None
    try:
        v = float(text)
    except ValueError:
        errors.append("{}: '{}' is not a number".format(name, text))
        return None
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        errors.append("{}: {:g} looks wrong".format(name, v))
        return None
    return v


def open_tool(api_key=None):
    """Build and display the lookup form. Returns the widget box."""
    import html
    import ipywidgets as w
    from IPython.display import display

    client = Client(api_key) if api_key else None
    state = {"results": [], "units": [], "details": None, "search_result": None}
    wide = w.Layout(width="95%")
    label_style = {"description_width": "140px"}

    query = w.Text(placeholder="e.g. 1140 Harrison St, San Francisco, CA 94103  or an apartments.com URL",
                   description="Address", layout=wide, style=label_style)
    lookup_btn = w.Button(description="Look up", button_style="primary", icon="search")
    status = w.HTML()
    prop_pick = w.Dropdown(description="Property", layout=wide, style=label_style)
    prop_btn = w.Button(description="Load this property", icon="download")
    prop_box = w.VBox([prop_pick, prop_btn], layout=w.Layout(display="none"))
    unit_pick = w.Dropdown(description="Unit", layout=wide, style=label_style)
    unit_box = w.VBox([unit_pick], layout=w.Layout(display="none"))

    beds = w.Text(description="Bedrooms (0=studio)", style=label_style)
    baths = w.Text(description="Bathrooms", style=label_style)
    sqft = w.Text(description="Square feet", style=label_style)
    zip_code = w.Text(description="Zip code", style=label_style)
    lat = w.Text(description="Latitude", style=label_style)
    lon = w.Text(description="Longitude", style=label_style)
    amen = {k: w.ToggleButtons(options=["Yes", "No", "Unknown"], value="Unknown",
                               description=AMENITY_LABELS[k], style={"description_width": "140px", "button_width": "80px"})
            for k in MODEL_AMENITIES}
    listed = w.HTML()
    other_amen = w.HTML()
    estimate_btn = w.Button(description="Estimate rent", button_style="success", icon="calculator")
    result = w.HTML()
    raw = w.Output()
    raw_acc = w.Accordion(children=[raw])
    raw_acc.set_title(0, "Raw response (for checking fields)")
    raw_acc.selected_index = None

    def say(msg, kind="info"):
        color = {"info": "inherit", "error": "#b00020", "ok": "#1b7f3b"}[kind]
        status.value = "<span style='color:{}'>{}</span>".format(color, html.escape(msg))

    def set_amenities(flags):
        for k, box in amen.items():
            v = flags.get(k)
            box.value = "Unknown" if v is None else ("Yes" if v else "No")

    def load_property(listing_key, search_result=None):
        say("Fetching details for {} ...".format(listing_key))
        try:
            data = details(client, listing_key)
        except APIError as exc:
            say(str(exc), "error")
            return
        state.update(details=data, search_result=search_result)
        summary = property_summary(search_result, data)
        zip_code.value = summary["zip_code"] or ""
        lat.value = _fmt(summary["latitude"])
        lon.value = _fmt(summary["longitude"])
        names = amenity_names(search_result or {}, data)
        set_amenities(model_amenities(names))
        unused = [n for n in names if not any(_norm(n) == _norm(t) for ts in MODEL_AMENITIES.values() for t in ts)]
        other_amen.value = ("<small><b>Other advertised amenities (not used by the model):</b> {}</small>"
                            .format(html.escape(", ".join(unused))) if unused else "")
        units = find_units(data)
        state["units"] = units
        options = [(unit_label(u), i) for i, u in enumerate(units)] + [("Enter features by hand", -1)]
        unit_pick.options = options
        unit_box.layout.display = None
        unit_pick.value = 0 if units else -1
        fill_unit()
        with raw:
            raw.clear_output()
            for path, value in leaves(data):
                print(path, "=", json.dumps(value)[:200])
        where = summary["address"] or summary["name"] or listing_key
        if units:
            say("{}: {} unit(s) / floor plan(s) found. Requests used this session: {}."
                .format(where, len(units), client.requests), "ok")
        else:
            say("{}: no units found in the response. Enter features by hand, or check the raw response."
                .format(where), "error")

    def fill_unit(*_):
        i = unit_pick.value
        if i is None or i < 0 or i >= len(state["units"]):
            listed.value = ""
            return
        u = state["units"][i]
        beds.value, baths.value, sqft.value = _fmt(u.get("beds")), _fmt(u.get("baths")), _fmt(u.get("sqft"))
        listed.value = ("<small>Listed asking rent: <b>${:,.0f}</b></small>".format(u["rent"])
                        if u.get("rent") is not None else "")
        result.value = ""

    def on_lookup(_):
        text = query.value.strip()
        result.value = ""
        if not text:
            say("Type an address first.", "error")
            return
        if client is None:
            say("No RealtyAPI key given, so lookups are off. Fill in the features by hand.", "error")
            return
        key = listing_key_from_input(text)
        if key:
            prop_box.layout.display = "none"
            load_property(key)
            return
        say("Searching ...")
        try:
            results = search(client, text)
        except APIError as exc:
            say(str(exc), "error")
            return
        state["results"] = results
        if not results:
            say("No listings found for that address. Try adding the city and zip, or paste the apartments.com URL.", "error")
            return
        if len(results) == 1:
            prop_box.layout.display = "none"
            load_property(results[0]["listingKey"], results[0])
            return
        prop_pick.options = [("{} | {}".format(r.get("name") or "", r.get("oneLineAddress") or r["listingKey"]), i)
                             for i, r in enumerate(results)]
        prop_pick.value = 0
        prop_box.layout.display = None
        say("{} listings matched. Pick the right property, then Load.".format(len(results)))

    def on_prop(_):
        r = state["results"][prop_pick.value]
        load_property(r["listingKey"], r)

    def on_estimate(_):
        errors = []
        args = dict(
            bedrooms=_parse_box(beds.value, "Bedrooms", errors, 0, 10),
            bathrooms=_parse_box(baths.value, "Bathrooms", errors, 0, 10),
            sqft=_parse_box(sqft.value, "Square feet", errors, 50, 20000),
            zip_code=zip_code.value.strip() or None,
            latitude=_parse_box(lat.value, "Latitude", errors, 37, 38.5),
            longitude=_parse_box(lon.value, "Longitude", errors, -123.5, -121.5),
            amenities={k: (None if b.value == "Unknown" else b.value == "Yes") for k, b in amen.items()},
        )
        if (args["latitude"] is None) != (args["longitude"] is None):
            errors.append("Give both latitude and longitude, or neither")
        if errors:
            result.value = "<span style='color:#b00020'>{}</span>".format(html.escape("; ".join(errors)))
            return
        r = rent_model.predict_rent(**args)
        rows = ["<h3 style='margin:4px 0'>Estimated asking rent: ${:,} / month</h3>".format(r["estimate"]),
                "<div>Likely range: ${:,} to ${:,} (about 80% of test listings fell in this range)</div>"
                .format(r["low"], r["high"])]
        i = unit_pick.value
        if state["units"] and i is not None and 0 <= i < len(state["units"]) and state["units"][i].get("rent"):
            listed_rent = state["units"][i]["rent"]
            rows.append("<div>Listed asking rent: ${:,.0f} ({:+.0%} vs estimate)</div>"
                        .format(listed_rent, listed_rent / r["estimate"] - 1))
        rows.append("<div><small>Model reliability: {}</small></div>".format(html.escape(r["reliability"])))
        if r["warnings"]:
            rows.append("<div><b>Warnings</b><ul>{}</ul></div>".format(
                "".join("<li>{}</li>".format(html.escape(x)) for x in r["warnings"])))
        result.value = "".join(rows)

    lookup_btn.on_click(on_lookup)
    prop_btn.on_click(on_prop)
    unit_pick.observe(fill_unit, names="value")
    estimate_btn.on_click(on_estimate)
    if client is None:
        say("No RealtyAPI key: lookups are off, but you can type features in and estimate.")

    ui = w.VBox([
        w.HTML("<h3 style='margin:4px 0'>SF rent estimate</h3>"),
        w.HBox([query, lookup_btn]), status, prop_box, unit_box,
        w.HTML("<b>Features</b> <small>(edit anything; leave blank if unknown)</small>"),
        beds, baths, sqft, zip_code, lat, lon, *amen.values(), listed, other_amen,
        estimate_btn, result, raw_acc,
    ])
    display(ui)
    return ui
