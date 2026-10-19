"""SF rent lookup: find an apartment on RealtyAPI, pick the unit, edit any feature, get a rent estimate.

Run from the repo root:

    pip install -r Code/requirements.txt
    streamlit run Code/rent_lookup_app.py

The RealtyAPI key comes from the REALTYAPI_KEY environment variable, a Streamlit secret
named REALTYAPI_KEY (.streamlit/secrets.toml, or the app's Secrets settings on Streamlit
Community Cloud), or the box in the sidebar. Without a key, features can still be typed in.
"""
import json
import re

import streamlit as st

import realty_lookup as rl
import rent_model
import sf_address

st.set_page_config(page_title="SF rent lookup", page_icon=":house:", layout="centered")

FEATURES = ["f_beds", "f_baths", "f_sqft", "f_zip", "f_lat", "f_lon"]
AMEN_KEYS = {k: "a_" + k for k in rl.MODEL_AMENITIES}
ss = st.session_state
for k, v in {"results": [], "cands": [], "units": [], "details": None, "summary": None, "amen_names": [],
             "building": None, "typed_unit": None, "typed_zip": None, "match_scores": [],
             "msg": None, "f_zip": "", **{a: "Unknown" for a in AMEN_KEYS.values()}}.items():
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
    """One Client per browser session and key, so its response cache and request cap persist."""
    if not key:
        return None
    if ss.get("client_key") != key:
        ss.client, ss.client_key = rl.Client(key), key
    return ss.client


def apply_unit():
    """Copy the chosen unit's fields into the feature boxes."""
    i = ss.get("unit_idx", -1)
    if i is None or not 0 <= i < len(ss.units):
        return
    u = ss.units[i]
    ss.f_beds = u.get("beds")
    ss.f_baths = u.get("baths")
    ss.f_sqft = int(round(u["sqft"])) if u.get("sqft") is not None else None


def load_property(client, listing_key, search_result=None):
    try:
        data = rl.details(client, listing_key)
    except rl.APIError as exc:
        ss.msg = ("error", str(exc))
        return
    summary = rl.property_summary(search_result, data)
    names = rl.amenity_names(search_result or {}, data)
    ss.details, ss.summary, ss.amen_names = data, summary, names
    ss.units = rl.find_units(data)
    b = ss.building
    ss.f_zip = summary["zip_code"] or ss.typed_zip or ""
    ss.f_lat = summary["latitude"] if summary["latitude"] is not None else (b.latitude if b else None)
    ss.f_lon = summary["longitude"] if summary["longitude"] is not None else (b.longitude if b else None)
    for k, v in rl.model_amenities(names).items():
        ss[AMEN_KEYS[k]] = "Unknown" if v is None else ("Yes" if v else "No")
    typed = [i for i, u in enumerate(ss.units) if sf_address.same_unit(ss.typed_unit, u.get("unit"))]
    ss.unit_idx = typed[0] if typed else (0 if ss.units else -1)
    if ss.units:
        apply_unit()
    else:
        ss.f_beds = ss.f_baths = ss.f_sqft = None
    where = summary["address"] or summary["name"] or listing_key
    if ss.units:
        note = " Unit {} selected.".format(ss.typed_unit) if typed else (
            " Unit {} is not listed; pick another unit or enter features by hand.".format(ss.typed_unit)
            if ss.typed_unit else "")
        ss.msg = ("success", "{}: {} unit(s) / floor plan(s) found.{}".format(where, len(ss.units), note))
    else:
        ss.msg = ("error", "{}: no units found in the response. Enter features by hand, "
                           "or check the raw response at the bottom.".format(where))


def reset_lookup():
    """Start a new lookup: forget the last property and clear its features."""
    ss.results, ss.cands, ss.msg, ss.details, ss.units, ss.amen_names = [], [], None, None, [], []
    ss.building, ss.typed_unit, ss.typed_zip = None, None, None
    ss.f_beds = ss.f_baths = ss.f_sqft = ss.f_lat = ss.f_lon = None
    ss.f_zip = ""
    for a in AMEN_KEYS.values():
        ss[a] = "Unknown"


def search_building(client, building, query):
    """Search RealtyAPI, put listings at `building` first, and load the one exact match if there is one."""
    ss.building = building
    if building is not None:
        # Fill location from the property records now, so an unlisted building can still be estimated.
        ss.f_lat, ss.f_lon = building.latitude, building.longitude
        ss.f_zip = ss.typed_zip or ""
    try:
        results = rl.search(client, query)
    except rl.APIError as exc:
        ss.msg = ("error", str(exc))
        return
    if building is not None:
        scored = sorted(((sf_address.same_building(r.get("oneLineAddress"), building), r) for r in results),
                        key=lambda x: -x[0])
        results = [r for _, r in scored]
        ss.match_scores = [sc for sc, _ in scored]
        exact = [r for sc, r in scored if sc >= 0.85]
        if len(exact) == 1:
            load_property(client, exact[0]["listingKey"], exact[0])
            return
        if not exact:
            where = building.address
            if results:
                ss.results, ss.prop_idx = results, 0
                ss.msg = ("warning", "No current listing found at {}. {} nearby listing(s) are below if one is "
                                     "the right building; otherwise enter the features by hand (location is "
                                     "already filled in from property records).".format(where, len(results)))
            else:
                ss.msg = ("warning", "No current listing found at {}. Enter the features by hand; location is "
                                     "filled in from property records where available.".format(where))
            return
    else:
        ss.match_scores = [None] * len(results)
    if not results:
        ss.msg = ("error", "No listings found for that address. Try the apartments.com URL instead.")
    elif building is None:
        # The address wasn't recognized, so never assume a result is the right building.
        ss.results, ss.prop_idx = results, 0
        ss.msg = ("info", "That address isn't in the SF property records. {} listing(s) found; "
                          "load one only if it's the right building.".format(len(results)))
    else:
        ss.results, ss.prop_idx = results, 0
        ss.msg = ("info", "{} listings at this address. Pick the right one, then Load.".format(len(results)))


def on_lookup(client, index):
    text = ss.query.strip()
    reset_lookup()
    if not text:
        ss.msg = ("error", "Type an address first.")
        return
    key = rl.listing_key_from_input(text)
    parsed = sf_address.parse(text)
    ss.typed_unit, ss.typed_zip = parsed.unit, parsed.zip_code
    cands = index.match(text) if index and not key else []
    if index and not key and index.is_confident(cands):
        ss.building = cands[0].building
        ss.f_lat, ss.f_lon = ss.building.latitude, ss.building.longitude
        ss.f_zip = ss.typed_zip or ""
    if client is None:
        ss.msg = ("error", "No RealtyAPI key, so listings can't be looked up. Add a key in the sidebar, or "
                           "type the features in{}.".format(
                               " (location is filled in from property records)" if ss.building else ""))
        return
    if key:
        load_property(client, key)
    elif ss.building is not None:
        search_building(client, ss.building, ss.building.query)
    elif cands:
        ss.cands, ss.cand_idx = cands, 0
        ss.msg = ("info", "Not sure which address you mean. Pick one, then Search.")
    else:
        query = text if re.search(r"san\s*francisco|\bsf\b", text, re.I) else text + ", San Francisco, CA"
        search_building(client, None, query)


def on_pick_candidate(client):
    i = ss.cand_idx
    cands, ss.cands = ss.cands, []
    if i < 0:
        text = ss.query.strip()
        query = text if re.search(r"san\s*francisco|\bsf\b", text, re.I) else text + ", San Francisco, CA"
        search_building(client, None, query)
    else:
        search_building(client, cands[i].building, cands[i].building.query)


def on_load(client):
    r = ss.results[ss.prop_idx]
    ss.results = []
    load_property(client, r["listingKey"], r)


@st.cache_resource(show_spinner="Loading SF property records ...")
def address_index():
    try:
        return sf_address.AddressIndex.load()
    except OSError:
        return None   # no Data/properties.csv: addresses are searched as typed


# ---------- sidebar ----------
with st.sidebar:
    st.header("RealtyAPI")
    key = env_key()
    if key:
        st.caption("Key loaded from settings.")
    else:
        key = st.text_input("API key", type="password", help="Not saved. Leave blank to type features in by hand.").strip()
    client = get_client(key)
    if client:
        st.caption("Requests used this session: {} of {}".format(client.requests, client.max_requests))
    st.caption("A URL or listing key costs 1 request. A street address costs 2 "
               "(search, then details). Typos are fixed against SF property records "
               "before searching, at no cost. Repeat lookups are cached.")
    st.divider()
    st.caption("Model: " + rent_model.MODEL_INFO["reliability"])
    st.caption("Trained mostly on large downtown buildings (SoMa, Tenderloin, Mission Bay). "
               "Outer neighborhoods and 3+ bedrooms are rough.")

# ---------- lookup ----------
st.title("SF rent lookup")
index = address_index()
with st.form("lookup", border=False):
    c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
    c1.text_input("Address", key="query", placeholder="1010 anza st #3   or an apartments.com URL")
    c2.form_submit_button("Look up", type="primary", on_click=on_lookup, args=(client, index), use_container_width=True)

if ss.msg:
    getattr(st, ss.msg[0])(ss.msg[1])

if ss.cands:
    cand_labels = {i: c.building.address + (" ({})".format(c.building.facts) if c.building.facts else "")
                   + ("  [{}]".format("; ".join(c.notes)) if c.notes else "") for i, c in enumerate(ss.cands)}
    cand_labels[-1] = "Search exactly as typed: " + ss.query.strip()
    st.selectbox("Did you mean", list(cand_labels), key="cand_idx", format_func=cand_labels.__getitem__)
    st.button("Search", on_click=on_pick_candidate, args=(client,), disabled=client is None)

if ss.building is not None and ss.building.facts:
    st.caption("Property records for {}: {}".format(ss.building.address, ss.building.facts))

if ss.results:
    prop_labels = ["{}{} | {}".format("Address match | " if (sc or 0) >= 0.85 else "", r.get("name") or "",
                                      r.get("oneLineAddress") or r["listingKey"])
                   for r, sc in zip(ss.results, ss.match_scores or [None] * len(ss.results))]
    st.selectbox("Property", range(len(prop_labels)), key="prop_idx", format_func=prop_labels.__getitem__)
    st.button("Load this property", on_click=on_load, args=(client,))

if ss.details is not None:
    unit_labels = {i: rl.unit_label(u) for i, u in enumerate(ss.units)}
    unit_labels[-1] = "Enter features by hand"
    st.selectbox("Unit", list(unit_labels), key="unit_idx", on_change=apply_unit, format_func=unit_labels.__getitem__)

# ---------- features ----------
st.subheader("Features")
st.caption("Edit anything. Leave a box blank if it is unknown; the model then uses its training average.")
c1, c2, c3 = st.columns(3)
c1.number_input("Bedrooms (0 = studio)", min_value=0.0, max_value=10.0, step=1.0, format="%.0f", value=None, key="f_beds")
c2.number_input("Bathrooms", min_value=0.0, max_value=10.0, step=0.5, format="%.1f", value=None, key="f_baths")
c3.number_input("Square feet", min_value=50, max_value=20000, step=10, value=None, key="f_sqft")
c1, c2, c3 = st.columns(3)
c1.text_input("Zip code", max_chars=5, key="f_zip")
c2.number_input("Latitude", min_value=37.0, max_value=38.5, step=0.0001, format="%.6f", value=None, key="f_lat")
c3.number_input("Longitude", min_value=-123.5, max_value=-121.5, step=0.0001, format="%.6f", value=None, key="f_lon")
for col, (k, sk) in zip(st.columns(3), AMEN_KEYS.items()):
    col.radio(rl.AMENITY_LABELS[k], ["Yes", "No", "Unknown"], key=sk, horizontal=True)
used = {rl._norm(t) for ts in rl.MODEL_AMENITIES.values() for t in ts}
other = [n for n in ss.amen_names if rl._norm(n) not in used]
if other:
    st.caption("Other advertised amenities (not used by the model): " + ", ".join(other))

# ---------- estimate ----------
st.subheader("Estimate")
lat, lon = ss.f_lat, ss.f_lon
if (lat is None) != (lon is None):
    st.info("Latitude and longitude are only used together; distance to downtown is taken from the zip.")
    lat = lon = None
r = rent_model.predict_rent(
    bedrooms=ss.f_beds, bathrooms=ss.f_baths, sqft=ss.f_sqft, zip_code=ss.f_zip.strip() or None,
    latitude=lat, longitude=lon,
    amenities={k: (None if ss[sk] == "Unknown" else ss[sk] == "Yes") for k, sk in AMEN_KEYS.items()})
# Metric values are markdown, where a pair of "$" starts math, so dollar signs are escaped.
range_help = "About 80% of test listings in buildings the model had not seen fell between low and high."
c1, c2, c3, c4 = st.columns(4)
c1.metric("Estimated rent / mo", "\\${:,}".format(r["estimate"]))
c2.metric("Likely low", "\\${:,}".format(r["low"]), help=range_help)
c3.metric("Likely high", "\\${:,}".format(r["high"]), help=range_help)
i = ss.get("unit_idx", -1)
if ss.details is not None and i is not None and 0 <= i < len(ss.units) and ss.units[i].get("rent"):
    listed = ss.units[i]["rent"]
    c4.metric("Listed asking rent", "\\${:,.0f}".format(listed),
              delta="{:+.0%} vs estimate".format(listed / r["estimate"] - 1), delta_color="off")
if r["warnings"]:
    st.warning("\n".join("- " + w for w in r["warnings"]))

if ss.details is not None:
    with st.expander("Raw response (for checking fields)"):
        st.code("\n".join("{} = {}".format(p, json.dumps(v)[:200]) for p, v in rl.leaves(ss.details)), language=None)
