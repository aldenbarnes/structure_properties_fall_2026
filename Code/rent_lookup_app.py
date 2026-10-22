"""SF rent lookup: find an apartment on RealtyAPI, pick the unit, edit any feature, get a rent estimate.

The estimate is rent_model's prediction (size, location and the amenities its lasso kept, fit on SF
listings) plus adjustments for other features taken from published studies (feature_adjustments.py).

Run from the repo root:

    pip install -r Code/requirements.txt
    streamlit run Code/rent_lookup_app.py

The RealtyAPI key comes from the REALTYAPI_KEY environment variable, a Streamlit secret
named REALTYAPI_KEY (.streamlit/secrets.toml, or the app's Secrets settings on Streamlit
Community Cloud), or the box in the sidebar. Without a key, features can still be typed in.
"""
import json
import re

import pandas as pd
import streamlit as st

import feature_adjustments as fa
import realty_lookup as rl
import rent_model
import sf_address

try:
    from streamlit_searchbox import st_searchbox
except ImportError:          # fall back to a plain text box + suggestion list
    st_searchbox = None

st.set_page_config(page_title="SF rent lookup", page_icon=":house:", layout="centered")

LAUNDRY = [fa.UNKNOWN, "None", "Shared in building", "In-unit washer and dryer"]
GARAGE = {"Shared garage space", "Assigned garage space", "Two or more spaces"}
MATCH_METERS = 25            # a listing this close to the parcel (same street and side) is the building

# Amenities rent_model prices (whatever its last training run kept), and the form input feeding each.
MODEL_AMENITIES = list(rent_model.PRICE_MODEL["amenity_share_yes"])
MODEL_FROM_CHECKBOX = {"air_conditioning": "Air conditioning", "dishwasher": "Dishwasher", "fireplace": "Fireplace",
                       "furnished": "Furnished", "gym": "Fitness center", "pool": "Pool",
                       "balcony_patio": "Private outdoor space", "elevator": "Elevator",
                       "hardwood_floors": "Hardwood floors", "walk_in_closets": "Walk-in closet",
                       "den": "Home office / den", "car_charging_station": "EV charging",
                       "roof_terrace": "Shared roof deck or yard"}
MODEL_FROM_SELECT = {"parking": "parking", "utilities_included": "utilities"}
MODEL_FROM_LAUNDRY = {"in_unit_laundry": "In-unit washer and dryer", "shared_laundry": "Shared in building"}
# Model amenities with no matching field in the form get a yes/no/unsure question of their own.
EXTRA_MODEL = [m for m in MODEL_AMENITIES
               if m not in MODEL_FROM_CHECKBOX and m not in MODEL_FROM_SELECT and m not in MODEL_FROM_LAUNDRY]
EXTRA_LABELS = {"doorman_concierge": "Doorman / concierge", "washer_dryer_hookup": "Washer/dryer hookups"}
# Form fields the SF model prices; feature_adjustments leaves these alone so nothing counts twice.
MODEL_PRICED = ({MODEL_FROM_CHECKBOX[m] for m in MODEL_AMENITIES if m in MODEL_FROM_CHECKBOX}
                | {MODEL_FROM_SELECT[m] for m in MODEL_AMENITIES if m in MODEL_FROM_SELECT})
CHECKS = fa.UNIT_FEATURES + fa.BUILDING_FEATURES


def ck(feature):
    return "x_" + feature


def mk(model_name):
    return "m_" + model_name


# Each feature is answered Yes, No or Unsure; Unsure means no adjustment (the model assumes a typical building).
YES, NO, UNSURE = "Yes", "No", "Unsure"
ANSWERS = [YES, NO, UNSURE]


def answer(flag):
    """True/False/None -> Yes/No/Unsure."""
    return UNSURE if flag is None else (YES if flag else NO)


# Values a new lookup resets to. f_* are the unit form's fields, s_* its dropdowns, x_*/m_* its yes/no/unsure
# features (x_ for the form's named features, m_ for model amenities with no other field).
FORM_DEFAULTS = {"f_beds": None, "f_baths": None, "f_sqft": None, "f_floor": None, "f_year": None, "f_zip": "",
                 "f_lat": None, "f_lon": None, "f_laundry": fa.UNKNOWN,
                 **{"s_" + k: fa.UNKNOWN for k in fa.SELECTS}, **{ck(f): UNSURE for f in CHECKS},
                 **{mk(m): UNSURE for m in EXTRA_MODEL}}
ss = st.session_state
for k, v in {"zillow": None, "lookup_steps": [], "results": [], "units": [], "details": None, "summary": None, "amen_names": [], "building": None,
             "typed_unit": None, "typed_zip": None, "msg": None, "last_pick": None, "unit_idx": -1,
             "load_id": 0, "result": None, "floor_note": None, **FORM_DEFAULTS}.items():
    ss.setdefault(k, v)


def env_key():
    import os
    key = os.environ.get("REALTYAPI_KEY", "").strip()
    if not key:
        try:
            key = str(st.secrets.get("REALTYAPI_KEY", "")).strip()
        except Exception:   # no secrets file
            key = ""
    return key


def get_client(key):
    """One Client per browser session and key, so its response cache, request cap and log persist."""
    if not key:
        return None
    if ss.get("client_key") != key:
        ss.client, ss.client_key = rl.Client(key), key
    return ss.client


@st.cache_resource
def zip_guesser():
    return sf_address.ZipGuesser.load()


@st.cache_resource(show_spinner="Loading SF property records ...")
def address_index():
    try:
        return sf_address.AddressIndex.load()
    except OSError:
        return None   # no Data/properties.csv: addresses are searched as typed


def as_typed_query(text):
    return text if re.search(r"san\s*francisco|\bsf\b", text, re.I) else text + ", San Francisco, CA"


# ---------- filling the form ----------

def apply_unit():
    """Copy the chosen unit's fields into the form. A new unit means any earlier estimate is stale."""
    ss.result = None
    i = ss.unit_idx
    if i is None or not 0 <= i < len(ss.units):
        return
    u = ss.units[i]
    ss.f_beds = u.get("beds")
    ss.f_baths = u.get("baths")
    ss.f_sqft = int(round(u["sqft"])) if u.get("sqft") is not None else None
    # Unit numbers like 204 or 1203 usually start with the floor.
    m = re.fullmatch(r"\D*(\d{3,4})", (u.get("unit") or "").strip())
    ss.f_floor = int(m.group(1)) // 100 if m and int(m.group(1)) // 100 > 0 else None
    ss.floor_note = "Floor guessed from unit number {}.".format(u["unit"]) if ss.f_floor else None


def on_unit_select(table_key):
    rows = ss[table_key]["selection"]["rows"]
    ss.unit_idx = rows[0] if rows else -1
    apply_unit()


def set_location(building):
    """Location and year built from the SF property records."""
    if building is not None:
        ss.f_lat, ss.f_lon = building.latitude, building.longitude
        ss.f_year = building.year_built
        ss.f_zip = ss.typed_zip or ""


def fill_amenities(names):
    """Tick the form's amenity fields from a listing's advertised amenity names."""
    flags = rl.model_amenities(names, set(MODEL_AMENITIES) | set(MODEL_FROM_LAUNDRY) | set(MODEL_FROM_SELECT))
    if flags["in_unit_laundry"]:
        ss.f_laundry = "In-unit washer and dryer"
    elif flags["shared_laundry"]:
        ss.f_laundry = "Shared in building"
    else:
        ss.f_laundry = "None" if names else fa.UNKNOWN
    if flags["parking"]:
        ss.s_parking = "Shared garage space"          # listings say parking exists, not which kind
    if flags["utilities_included"]:
        ss.s_utilities = "Water, sewer, garbage"      # the smallest bundle; adjust if more is included
    # With an amenity list, anything not on it is No (how the model's training listings were coded);
    # without one, every feature stays Unsure.
    advertised = fa.checks_from_amenities(names)
    for f in CHECKS:
        ss[ck(f)] = answer(f in advertised if names else None)
    for m, f in MODEL_FROM_CHECKBOX.items():
        if flags.get(m):
            ss[ck(f)] = YES
    for m in EXTRA_MODEL:
        ss[mk(m)] = answer(flags.get(m))


# ---------- lookup ----------

def reset_lookup():
    """Start a new lookup: forget the last property and clear its features."""
    ss.results, ss.msg, ss.details, ss.units, ss.amen_names = [], None, None, [], []
    ss.building, ss.typed_unit, ss.typed_zip, ss.lookup_steps, ss.zillow = None, None, None, [], None
    ss.unit_idx, ss.result, ss.floor_note = -1, None, None
    for k, v in FORM_DEFAULTS.items():
        ss[k] = v


def load_property(client, listing_key, search_result=None, data=None):
    """Load a listing's units into the page. `data` is a details record already in hand (no request)."""
    if data is None:
        try:
            data = rl.details(client, listing_key)
        except rl.APIError as exc:
            ss.msg = ("error", str(exc))
            return
    summary = rl.property_summary(search_result, data)
    names = rl.amenity_names(search_result or {}, data)
    ss.details, ss.summary, ss.amen_names, ss.results = data, summary, names, []
    ss.units = rl.find_units(data)
    b = ss.building
    ss.f_zip = summary["zip_code"] or ss.typed_zip or ""
    ss.f_lat = summary["latitude"] if summary["latitude"] is not None else (b.latitude if b else None)
    ss.f_lon = summary["longitude"] if summary["longitude"] is not None else (b.longitude if b else None)
    fill_amenities(names)
    ss.load_id += 1
    typed = [i for i, u in enumerate(ss.units) if sf_address.same_unit(ss.typed_unit, u.get("unit"))]
    ss.unit_idx = typed[0] if typed else (0 if ss.units else -1)
    apply_unit()
    where = summary["address"] or summary["name"] or listing_key
    if ss.units:
        note = " Unit {} selected.".format(ss.typed_unit) if typed else (
            " Unit {} is not listed; pick another unit or enter features by hand.".format(ss.typed_unit)
            if ss.typed_unit else "")
        ss.msg = ("success", "{}: {} unit(s) / floor plan(s) found.{}".format(where, len(ss.units), note))
    else:
        ss.msg = ("error", "{}: no units found in the response. Enter features by hand, "
                           "or check the raw response at the bottom.".format(where))


def rank_results(results, building):
    """Annotate listings with their address, distance from the building, and whether they are it.

    A listing is the building when its address matches (an address range covering the number counts),
    or when it sits within MATCH_METERS of the parcel on the same street and side of the street, which
    catches listings that write the address differently without grabbing the neighbor across the street.
    """
    rows = []
    for r in results:
        addr = rl.listing_address(r)
        lat, lon = rl.listing_coords(r)
        dist, match = None, False
        if building is not None:
            dist = rl.distance_m(building.latitude, building.longitude, lat, lon)
            near = dist is not None and dist <= MATCH_METERS
            street_ok = addr is None or sf_address.street_similarity(addr, building) >= 0.85
            match = (sf_address.same_building(addr, building) >= 0.85
                     or (near and street_ok and sf_address.same_side(addr, building)))
        rows.append({"r": r, "addr": addr, "dist": dist, "match": match})
    rows.sort(key=lambda x: (not x["match"], x["dist"] if x["dist"] is not None else 1e12))
    return rows


def fill_from_zillow(rec):
    """Fill the form from a Zillow property record (a building with no apartments.com listing)."""
    ss.zillow = rec
    ss.f_beds, ss.f_baths = rec["beds"], rec["baths"]
    ss.f_sqft = int(round(rec["sqft"])) if rec["sqft"] else None
    ss.f_year = rec["year_built"] or ss.f_year
    ss.f_zip = rec["zip_code"] or ss.f_zip
    if rec["latitude"] is not None and rec["longitude"] is not None:
        ss.f_lat, ss.f_lon = rec["latitude"], rec["longitude"]
    if rec["home_type"]:
        ss.s_property_type = rec["home_type"]
    if rec["laundry"]:
        ss.f_laundry = rec["laundry"]
    if rec["garage"]:
        ss.s_parking = "Assigned garage space"
    if rec["view"]:
        ss.s_view = rec["view"]
    for feature, field in [("Dishwasher", "dishwasher"), ("Air conditioning", "air_conditioning"),
                           ("Fireplace", "fireplace"), ("Hardwood floors", "hardwood")]:
        if rec[field] is not None:       # Zillow's facts are incomplete: what it doesn't state stays Unsure
            ss[ck(feature)] = answer(rec[field])
    ss.result = None


def find_building(client, building):
    """Find the building with up to three lookups, stopping at the first that finds it:
    1. apartments.com /details/byaddress (skipped for the session once RealtyAPI rejects the call);
    2. an apartments.com search on the address (an area search; 40 results);
    3. Zillow's record for the address, which covers buildings with no apartments.com listing.
    If apartments.com has nothing at the address, its nearby listings are offered as comparables."""
    full = building.query                              # "1503 Masonic Ave, San Francisco, CA"
    steps, seen = [], {}

    def take(rows, label, n):
        for x in rows:
            seen.setdefault(x["r"].get("listingKey"), x)
        matches = [x for x in rows if x["match"]]
        on_street = [x for x in rows if not x["match"] and x["addr"]
                     and sf_address.street_similarity(x["addr"], building) >= 0.85]
        note = "{} at this address".format(len(matches)) if matches else "none at this address"
        if on_street and not matches:
            note += " (on the same street: {})".format(", ".join(
                (x["addr"].split(",")[0] + (" {:,.0f} m".format(x["dist"]) if x["dist"] is not None else ""))
                for x in on_street[:4]))
        steps.append("{}: {} listing(s), {}.".format(label, n, note))
        return matches

    def done(matches):
        ss.lookup_steps = steps
        if len(matches) == 1:
            load_property(client, matches[0]["r"]["listingKey"], matches[0]["r"])
        else:
            ss.results, ss.prop_idx = matches, 0
            ss.msg = ("info", "{} listings at this address. Pick the right one, then Load.".format(len(matches)))

    if not ss.get("byaddress_off"):
        try:
            rec = rl.details_by_address(client, full)
        except rl.APIError as exc:
            if "not indexed" in str(exc):
                steps.append("apartments.com address lookup: address not indexed by apartments.com.")
            else:
                note = ""
                if re.search(r"HTTP 4(00|04|22)", str(exc)):
                    ss.byaddress_off, note = True, " Skipped for the rest of this session."
                steps.append("apartments.com address lookup: {}.{}".format(exc, note))
        else:
            key = rl._find_first(rec, ["listingKey"])
            lat, lon = rl.listing_coords(rec)
            stub = {"listingKey": key, "oneLineAddress": rl.listing_address(rec), "latitude": lat, "longitude": lon,
                    "name": rl._find_first(rec, ["propertyName", "name"])}
            if key or rl.find_units(rec):
                matches = take(rank_results([stub], building), "apartments.com address lookup", 1)
                if matches:
                    ss.lookup_steps = steps
                    load_property(client, key, stub, data=rec if rl.find_units(rec) else None)
                    return
            else:
                steps.append("apartments.com address lookup: no listing for this address.")

    try:
        results = rl.search(client, full)
    except rl.APIError as exc:
        steps.append("apartments.com address search: {}.".format(exc))
    else:
        matches = take(rank_results(results, building), "apartments.com address search", len(results))
        if matches:
            return done(matches)

    try:
        rec = rl.zillow_record(rl.zillow_by_address(client, full))
    except rl.APIError as exc:
        steps.append("Zillow: {}.".format(exc))
    else:
        zaddr = rec["address"] if isinstance(rec["address"], str) else None
        dist = rl.distance_m(building.latitude, building.longitude, rec["latitude"], rec["longitude"])
        if (zaddr and sf_address.same_building(zaddr, building) >= 0.85) or (dist is not None and dist <= MATCH_METERS):
            steps.append("Zillow: found {}.".format(zaddr or building.address))
            ss.lookup_steps = steps
            fill_from_zillow(rec)
            facts = ", ".join(x for x in [
                rec["home_type"],
                "{:g} bd / {:g} ba".format(rec["beds"], rec["baths"]) if rec["beds"] is not None and rec["baths"] is not None else None,
                "{:,.0f} sq ft".format(rec["sqft"]) if rec["sqft"] else None] if x) or "no details"
            ss.msg = ("success", "Found on Zillow: {} ({}). It has no apartments.com listing, so there is no unit list; "
                                 "the form is filled from Zillow's record. Check it, then press Estimate rent."
                      .format(building.address, facts))
            return
        steps.append("Zillow: returned a different address ({}), not used.".format(zaddr or "unknown"))

    ss.lookup_steps = steps
    rows = sorted((x for x in seen.values() if x["r"].get("listingKey")),
                  key=lambda x: x["dist"] if x["dist"] is not None else 1e12)[:20]
    ss.results, ss.prop_idx = rows, 0
    ss.msg = ("warning", "{} wasn't found on apartments.com or Zillow. Enter the features by hand (location is "
                         "filled in from property records){}.".format(
                             building.address, ", or load a nearby listing below as a comparable" if rows else ""))


def search_zip(client):
    """apartments.com listings in the building's zip code, nearest first, as comparables (one request)."""
    b = ss.building
    z = (ss.f_zip or "").strip() or zip_guesser().guess(b.latitude, b.longitude)
    try:
        results = rl.search(client, "San Francisco, CA {}".format(z), result_count=100)
    except rl.APIError as exc:
        ss.msg = ("error", str(exc))
        return
    rows = rank_results(results, b)
    ss.results, ss.prop_idx = rows[:25], 0
    ss.msg = ("info", "{} apartments.com listing(s) in {}; the nearest are listed below. Loading one replaces the "
                      "form with that listing's unit.".format(len(results), z) if results
              else "No apartments.com listings found in {}.".format(z))


def search_typed(client, query):
    """Search text that isn't an SF address in the property records; never auto-load a result."""
    try:
        results = rl.search(client, query)
    except rl.APIError as exc:
        ss.msg = ("error", str(exc))
        return
    ss.results, ss.prop_idx = rank_results(results, None), 0
    ss.msg = (("info", "That address isn't in the SF property records. {} listing(s) found; load one only if "
                       "it's the right building.".format(len(results))) if results else
              ("error", "No listings found for that text. Pick an address from the suggestions instead."))


def run_lookup(client, kind, payload, term):
    reset_lookup()
    p = sf_address.parse(term)
    ss.typed_unit, ss.typed_zip = p.unit, p.zip_code
    if kind == "building":
        ss.building = payload
        set_location(payload)
    if client is None:
        ss.msg = ("error", "No RealtyAPI key, so listings can't be looked up. Add a key in the sidebar, or type "
                           "the features in{}.".format(" (location is filled in from property records)"
                                                       if ss.building else ""))
        return
    if kind == "key":
        load_property(client, payload)
    elif kind == "building":
        find_building(client, payload)
    else:
        search_typed(client, as_typed_query(term))


def on_load(client):
    x = ss.results[ss.prop_idx]
    load_property(client, x["r"]["listingKey"], x["r"])


def suggestions(index, term):
    """Options for the address box: real SF addresses first, then the text exactly as typed."""
    term = (term or "").strip()
    if not term:
        return []
    key = rl.listing_key_from_input(term)
    if key:
        return [("apartments.com listing " + key, ("key", key, term))]
    out = []
    for c in (index.suggest(term) if index else []):
        b = c.building
        label = b.address + (" · " + b.facts if b.facts else "") + ("" if c.exact else " · not in property records")
        out.append((label, ("building", b, term)))
    out.append(("Search exactly as typed: " + term, ("typed", None, term)))
    return out


# ---------- sidebar ----------
with st.sidebar:
    st.header("RealtyAPI")
    key = env_key()
    if key:
        st.caption("Key loaded from settings.")
    else:
        key = st.text_input("API key", type="password", help="Not saved. Leave blank to type features in by hand.").strip()
    client = get_client(key)
    requests_slot = st.empty()       # filled in at the end, after this run's lookups
    st.caption("Picking an address uses up to 3 lookups (apartments.com address lookup, apartments.com search, "
               "Zillow), stopping at the first that finds the building, plus 1 to load apartments.com units. "
               "An apartments.com URL or listing key costs 1. Repeat lookups are cached.")
    st.divider()
    st.caption("SF model: " + rent_model.MODEL_INFO["reliability"])
    st.caption("Trained on {} units in {} buildings, weighted toward SoMa, Mission Bay and downtown; "
               "read the warnings under each estimate.".format(rent_model.MODEL_INFO["n_used"],
                                                               rent_model.MODEL_INFO["n_buildings"]))
    st.caption("Features the SF model doesn't measure (condition, views, pets, ...) are priced from published "
               "studies. See the breakdown under the estimate.")

# ---------- address ----------
st.title("SF rent lookup")
index = address_index()
if st_searchbox is not None:
    # The box clears itself after a pick, so picking the same address again counts as a new pick.
    pick = st_searchbox(lambda term: suggestions(index, term), key="address_box", label="Address",
                        placeholder="Start typing an SF address, e.g. 1010 anza   (or paste an apartments.com URL)",
                        debounce=250, clear_on_submit=True)
    if pick is None:
        ss.last_pick = None
else:
    typed = st.text_input("Address", placeholder="1010 anza st #3   or an apartments.com URL")
    opts = suggestions(index, typed)
    choice = st.selectbox("Matching addresses", range(len(opts)), format_func=lambda i: opts[i][0]) if opts else None
    pick = opts[choice][1] if opts and st.button("Look up", type="primary") else None
    ss.last_pick = None                      # every press of Look up is a new lookup
if pick is not None and pick != ss.last_pick:
    ss.last_pick = pick
    run_lookup(client, *pick)

if ss.msg:
    getattr(st, ss.msg[0])(ss.msg[1])
if ss.building is not None and ss.building.facts:
    st.caption("Property records for {}: {}".format(ss.building.address, ss.building.facts))
if ss.lookup_steps:
    st.caption("How it was looked up: " + " ".join("{}. {}".format(i + 1, t) for i, t in enumerate(ss.lookup_steps)))
if ss.zillow:
    z = ss.zillow
    bits = ["Rent Zestimate ${:,.0f}/mo".format(z["rent_zestimate"]) if z["rent_zestimate"] else None,
            "listed for rent at ${:,.0f}/mo".format(z["list_rent"]) if z["list_rent"] else None,
            "status: " + z["home_status"].replace("_", " ").lower() if z["home_status"] else None,
            "value Zestimate ${:,.0f}".format(z["zestimate"]) if z["zestimate"] else None]
    st.caption("Zillow: " + "; ".join(b for b in bits if b) + ".")

if ss.results:
    def prop_label(i):
        x = ss.results[i]
        r = x["r"]
        bits = (["This address"] if x["match"] else []) + [r.get("name") or "", x["addr"] or r["listingKey"]]
        if x["dist"] is not None:
            bits.append("{:,.0f} m away".format(x["dist"]))
        return " | ".join(b for b in bits if b)
    prop_labels = [prop_label(i) for i in range(len(ss.results))]
    st.selectbox("Listings", range(len(ss.results)), key="prop_idx", format_func=prop_labels.__getitem__)
    st.button("Load this listing", on_click=on_load, args=(client,))
z_known = (ss.f_zip or "").strip() or (ss.building is not None and ss.building.latitude is not None
                                       and zip_guesser().guess(ss.building.latitude, ss.building.longitude))
if client is not None and ss.building is not None and ss.details is None and z_known:
    st.button("Find comparable apartments.com listings in {} (1 request)".format(z_known), on_click=search_zip,
              args=(client,))

if ss.details is not None and ss.units:
    st.subheader("Units")
    st.caption("Tick the box at the left of a unit to use it; click a column header to sort. "
               "Untick it to enter the features by hand.")
    table = pd.DataFrame([{
        "Unit": u.get("unit") or "(floor plan)",
        "Floor plan": u.get("plan") or "",
        "Beds": u.get("beds"),
        "Baths": u.get("baths"),
        "Sq ft": u.get("sqft"),
        "Asking rent": u.get("rent"),
        "Available": u.get("available") or "",
    } for u in ss.units])
    table_key = "unit_table_{}".format(ss.load_id)    # new key per property, so its default selection applies
    st.dataframe(
        table, key=table_key, hide_index=True, width="stretch",
        on_select=lambda: on_unit_select(table_key), selection_mode="single-row",
        selection_default={"selection": {"rows": [ss.unit_idx]}} if ss.unit_idx >= 0 else None,
        column_config={"Beds": st.column_config.NumberColumn(help="0 = studio", format="%g"),
                       "Baths": st.column_config.NumberColumn(format="%g"),
                       "Sq ft": st.column_config.NumberColumn(format="%d"),
                       "Asking rent": st.column_config.NumberColumn(format="dollar")})
    if 0 <= ss.unit_idx < len(ss.units):
        st.caption("Selected: " + rl.unit_label(ss.units[ss.unit_idx]))
    else:
        st.caption("No unit selected: the form below is filled in by hand.")

# ---------- unit form ----------
st.subheader("Describe the unit")
st.caption("Edit anything, then press Estimate rent. Blank or Unknown means not known: no adjustment is made.")
PRICED_BY_MODEL = "Priced by the SF model."
PRICED_BY_STUDIES = "Priced from published studies (see the breakdown under the estimate)."


def select(col, key):
    label, options = fa.SELECTS[key]
    col.selectbox(label, [fa.UNKNOWN] + options, key="s_" + key,
                  help=PRICED_BY_MODEL if key in MODEL_PRICED else PRICED_BY_STUDIES)


def questions(items):
    """items: (session key, label, help) tuples, each a Yes / No / Unsure choice, in three columns."""
    cols = st.columns(3)
    for i, (key, label, help_text) in enumerate(items):
        cols[i % 3].radio(label, ANSWERS, key=key, help=help_text, horizontal=True)


def feature_items(features):
    return [(ck(f), f, PRICED_BY_MODEL if f in MODEL_PRICED else PRICED_BY_STUDIES) for f in features]


with st.form("unit_form"):
    st.markdown("**Unit basics**")
    c1, c2, c3 = st.columns(3)
    c1.number_input("Bedrooms (0 = studio)", min_value=0.0, max_value=10.0, step=1.0, format="%.0f", value=None, key="f_beds")
    c2.number_input("Bathrooms", min_value=0.0, max_value=10.0, step=0.5, format="%.1f", value=None, key="f_baths")
    c3.number_input("Square feet", min_value=50, max_value=20000, step=10, value=None, key="f_sqft")
    c1, c2, c3 = st.columns(3)
    c1.number_input("Floor", min_value=0, max_value=80, step=1, value=None, key="f_floor", help=ss.floor_note)
    c2.number_input("Year built", min_value=1850, max_value=2035, step=1, value=None, key="f_year",
                    help="For reference only; Condition is what gets priced, so age is not counted twice.")
    select(c3, "property_type")

    st.markdown("**Location**")
    c1, c2, c3 = st.columns(3)
    c1.text_input("Zip code", max_chars=5, key="f_zip")
    c2.number_input("Latitude", min_value=37.0, max_value=38.5, step=0.0001, format="%.6f", value=None, key="f_lat")
    c3.number_input("Longitude", min_value=-123.5, max_value=-121.5, step=0.0001, format="%.6f", value=None, key="f_lon")

    st.markdown("**Condition and terms**")
    c1, c2, c3 = st.columns(3)
    select(c1, "condition")
    select(c2, "parking")
    c3.selectbox("Laundry", LAUNDRY, key="f_laundry",
                 help=PRICED_BY_MODEL if set(MODEL_FROM_LAUNDRY) & set(MODEL_AMENITIES) else None)
    c1, c2, c3 = st.columns(3)
    select(c1, "pets")
    select(c2, "utilities")
    select(c3, "lease_term")
    c1, _, _ = st.columns(3)
    select(c1, "view")

    st.markdown("**Unit features**")
    st.caption("Unsure means no adjustment. No lowers the estimate a little for features many training "
               "buildings have. A listing's amenity list fills these in.")
    questions(feature_items(fa.UNIT_FEATURES))
    st.markdown("**Building amenities**")
    questions(feature_items(fa.BUILDING_FEATURES)
               + [(mk(m), EXTRA_LABELS.get(m, m.replace("_", " ").capitalize()), PRICED_BY_MODEL) for m in EXTRA_MODEL])

    estimate_clicked = st.form_submit_button("Estimate rent", type="primary")

mapped = {fa._norm(a) for aliases in fa.AMENITY_NAMES.values() for a in aliases}
mapped |= {fa._norm(t) for m in MODEL_AMENITIES for t in rl.AMENITY_GROUPS.get(m, [m.replace("_", " ")])}
mapped |= {fa._norm(t) for m in MODEL_FROM_LAUNDRY for t in rl.AMENITY_GROUPS[m]}
other = [n for n in ss.amen_names if fa._norm(n) not in mapped]
if other:
    st.caption("Other advertised amenities (not priced): " + ", ".join(other))


def model_amenity_inputs():
    """rent_model amenity inputs from the form: True, False, or None for unknown."""
    def box(key):
        return {YES: True, NO: False}.get(ss[key])
    out = {}
    for m in MODEL_AMENITIES:
        if m in MODEL_FROM_LAUNDRY:
            out[m] = None if ss.f_laundry == fa.UNKNOWN else ss.f_laundry == MODEL_FROM_LAUNDRY[m]
        elif m in MODEL_FROM_SELECT:
            v = ss["s_" + MODEL_FROM_SELECT[m]]
            out[m] = None if v == fa.UNKNOWN else (v in GARAGE if m == "parking" else v != "None")
        elif m in MODEL_FROM_CHECKBOX:
            out[m] = box(ck(MODEL_FROM_CHECKBOX[m]))
        else:
            out[m] = box(mk(m))
    return out


def run_estimate():
    lat, lon = ss.f_lat, ss.f_lon
    notes = []
    if (lat is None) != (lon is None):
        notes.append("Latitude and longitude are only used together; distance to downtown was taken from the zip.")
        lat = lon = None
    amenities = model_amenity_inputs()
    model = rent_model.predict_rent(
        bedrooms=ss.f_beds, bathrooms=ss.f_baths, sqft=ss.f_sqft, zip_code=ss.f_zip.strip() or None,
        latitude=lat, longitude=lon, amenities=amenities)
    choices = {k: ss["s_" + k] for k in fa.SELECTS}
    choices["floor"] = ss.f_floor
    choices["features"] = {f for f in CHECKS if ss[ck(f)] == YES}
    choices["missing"] = {f for f in CHECKS if ss[ck(f)] == NO}
    choices["skip"] = MODEL_PRICED
    adj = fa.adjust(model["estimate"], choices)
    i = ss.unit_idx
    listed = ss.units[i].get("rent") if ss.details is not None and 0 <= i < len(ss.units) else None
    compare = ("Listed asking rent", listed) if listed else None
    if compare is None and ss.zillow:
        if ss.zillow["list_rent"]:
            compare = ("Zillow asking rent", ss.zillow["list_rent"])
        elif ss.zillow["rent_zestimate"]:
            compare = ("Zillow Rent Zestimate", ss.zillow["rent_zestimate"])
    return {"model": model, "adj": adj, "compare": compare, "notes": notes, "amenities": amenities}


if estimate_clicked:
    ss.result = run_estimate()

# ---------- estimate ----------
st.subheader("Estimate")
if ss.result is None:
    st.info("Fill in the unit above, then press Estimate rent.")
else:
    model, adj, compare = ss.result["model"], ss.result["adj"], ss.result["compare"]
    est = adj["estimate"]
    low, high = model["low"] * adj["factor"], model["high"] * adj["factor"]
    # Metric values are markdown, where a pair of "$" starts math, so dollar signs are escaped.
    range_help = ("About 80% of test listings in buildings the model had not seen fell between low and high. "
                  "This covers the model's error only; the study-based adjustments add more uncertainty.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estimated rent / mo", "\\${:,.0f}".format(est))
    c2.metric("Likely low", "\\${:,.0f}".format(low), help=range_help)
    c3.metric("Likely high", "\\${:,.0f}".format(high), help=range_help)
    if compare:
        c4.metric(compare[0], "\\${:,.0f}".format(compare[1]),
                  delta="{:+.0%} vs estimate".format(compare[1] / est - 1), delta_color="off")
    for n in ss.result["notes"]:
        st.info(n)

    st.markdown("**How we got there**")
    money = lambda x, signed=False: ("{}${:,.0f}".format(("+" if x >= 0 else "-") if signed else "", abs(x)))
    amen = ss.result["amenities"]
    has = [m.replace("_", " ") for m, v in amen.items() if v]
    unknown = [m.replace("_", " ") for m, v in amen.items() if v is None]
    why = "Size, location and the amenities the SF model prices. With: {}.".format(", ".join(has) or "none")
    if unknown:
        why += " Unknown (typical share assumed): {}.".format(", ".join(unknown))
    rel = model["reliability"]
    rows = [{"Factor": "SF model estimate", "Amount": money(model["estimate"]),
             "Why": why + " " + rel[:1].upper() + rel[1:] + "."}]
    lacking = [r for r in adj["rows"] if r["detail"].startswith("Doesn't have it")]
    for r in adj["rows"]:
        if r in lacking:
            continue
        pct = " ({:+.1f}%)".format(r["value"]) if r["unit"] == "pct" else ""
        rows.append({"Factor": r["factor"], "Amount": money(r["dollars"], signed=True),
                     "Why": "{}{}. {} Confidence: {}.".format(r["detail"], pct, r["note"], r["confidence"])})
    if lacking:
        rows.append({"Factor": "Missing common features", "Amount": money(sum(r["dollars"] for r in lacking), signed=True),
                     "Why": "Features many training buildings advertise, scaled by how common each is: " + "; ".join(
                         "{} {}".format(r["factor"].lower(), money(r["dollars"], signed=True)) for r in lacking) + "."})
    rows.append({"Factor": "Estimated rent", "Amount": money(est), "Why": "SF model estimate plus adjustments."})
    # Table cells are markdown too: escape "$" so pairs of dollar amounts don't render as math.
    st.table(pd.DataFrame(rows).set_index("Factor").map(lambda t: t.replace("$", "\\$")))
    if not adj["rows"]:
        st.caption("No study-based adjustments: every other feature is Unknown, Unsure, or the same as the "
                   "typical building the model was trained on.")
    sources = {}
    for r in adj["rows"]:
        for src in r["sources"]:
            sources.setdefault(src["url"] or src["title"], src)
    if sources:
        with st.expander("Sources for the study-based adjustments"):
            for src in sources.values():
                title = "[{}]({})".format(src["title"], src["url"]) if src.get("url") else src["title"]
                st.markdown("- {}, {} ({})".format(src["name"], title, src["year"] or "undated"))
            st.caption("Adjustments are sized relative to the typical training building: having a feature most "
                       "training buildings advertise adds less, and lacking it subtracts a little. Most sources are "
                       "listing comparisons or renter surveys, checked against search-engine extracts rather than "
                       "full documents; treat low-confidence adjustments as rough.")
    if model["warnings"]:
        st.warning("\n".join("- " + w for w in model["warnings"]))

# ---------- diagnostics ----------
if client is not None:
    requests_slot.caption("Requests used this session: {} of {}".format(client.requests, client.max_requests))
if client is not None and client.log:
    with st.expander("RealtyAPI requests this session"):
        st.caption("What was sent to RealtyAPI and what came back (newest first). If a lookup finds nothing, "
                   "this shows whether the request failed or the search returned no listings.")
        st.code("\n".join("{} {} -> {}".format(e["endpoint"], json.dumps(e["params"]), e["result"])
                          for e in reversed(client.log[-15:])), language=None)
if ss.details is not None:
    with st.expander("Raw response (for checking fields)"):
        st.code("\n".join("{} = {}".format(p, json.dumps(v)[:200]) for p, v in rl.leaves(ss.details)), language=None)
