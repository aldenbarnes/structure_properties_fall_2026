"""SF rent lookup: find an apartment on RealtyAPI, pick the unit, edit any feature, get a rent estimate.

The estimate is rent_model's prediction (size, location, laundry, doorman, fit on SF listings) plus
adjustments for other features taken from published studies (feature_adjustments.py).

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

st.set_page_config(page_title="SF rent lookup", page_icon=":house:", layout="centered")

LAUNDRY = [fa.UNKNOWN, "None", "Shared in building", "In-unit washer and dryer"]
DOORMAN = "Doorman / concierge"            # priced by rent_model, not by feature_adjustments
CHECKS = fa.UNIT_FEATURES + [DOORMAN] + fa.BUILDING_FEATURES


def ck(feature):
    return "x_" + feature


# Values a new lookup resets to. f_* are the unit form's fields, s_* its dropdowns, x_* its checkboxes.
FORM_DEFAULTS = {"f_beds": None, "f_baths": None, "f_sqft": None, "f_floor": None, "f_year": None, "f_zip": "",
                 "f_lat": None, "f_lon": None, "f_laundry": fa.UNKNOWN,
                 **{"s_" + k: fa.UNKNOWN for k in fa.SELECTS}, **{ck(f): False for f in CHECKS}}
ss = st.session_state
for k, v in {"results": [], "cands": [], "units": [], "details": None, "summary": None, "amen_names": [],
             "building": None, "typed_unit": None, "typed_zip": None, "match_scores": [], "msg": None,
             "unit_idx": -1, "load_id": 0, "result": None, "floor_note": None, **FORM_DEFAULTS}.items():
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
    flags = rl.model_amenities(names)
    if flags["in_unit_laundry"]:
        ss.f_laundry = "In-unit washer and dryer"
    elif flags["shared_laundry"]:
        ss.f_laundry = "Shared in building"
    else:
        ss.f_laundry = "None" if names else fa.UNKNOWN
    advertised = fa.checks_from_amenities(names)
    for f in CHECKS:
        ss[ck(f)] = f in advertised
    ss[ck(DOORMAN)] = bool(flags["doorman_concierge"])
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


def reset_lookup():
    """Start a new lookup: forget the last property and clear its features."""
    ss.results, ss.cands, ss.msg, ss.details, ss.units, ss.amen_names = [], [], None, None, [], []
    ss.building, ss.typed_unit, ss.typed_zip = None, None, None
    ss.unit_idx, ss.result, ss.floor_note = -1, None, None
    for k, v in FORM_DEFAULTS.items():
        ss[k] = v


def search_building(client, building, query):
    """Search RealtyAPI, put listings at `building` first, and load the one exact match if there is one."""
    ss.building = building
    # Fill location from the property records now, so an unlisted building can still be estimated.
    set_location(building)
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
        set_location(ss.building)
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
    st.caption("Other features (condition, parking, views, ...) are priced from published studies, "
               "not from SF data. See the breakdown under the estimate.")

# ---------- lookup ----------
st.title("SF rent lookup")
index = address_index()
with st.form("lookup", border=False):
    c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
    c1.text_input("Address", key="query", placeholder="1010 anza st #3   or an apartments.com URL")
    c2.form_submit_button("Look up", type="primary", on_click=on_lookup, args=(client, index), width="stretch")

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


def select(col, key):
    label, options = fa.SELECTS[key]
    col.selectbox(label, [fa.UNKNOWN] + options, key="s_" + key)


def checkboxes(features):
    cols = st.columns(3)
    for i, f in enumerate(features):
        cols[i % 3].checkbox(f, key=ck(f))


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
    c3.selectbox("Laundry", LAUNDRY, key="f_laundry")
    c1, c2, c3 = st.columns(3)
    select(c1, "pets")
    select(c2, "utilities")
    select(c3, "lease_term")

    st.markdown("**Unit features**")
    checkboxes(fa.UNIT_FEATURES)
    st.markdown("**Building amenities**")
    checkboxes([DOORMAN] + fa.BUILDING_FEATURES)

    estimate_clicked = st.form_submit_button("Estimate rent", type="primary")

mapped = {fa._norm(a) for aliases in fa.AMENITY_NAMES.values() for a in aliases}
mapped |= {fa._norm(t) for ts in rl.MODEL_AMENITIES.values() for t in ts}
other = [n for n in ss.amen_names if fa._norm(n) not in mapped]
if other:
    st.caption("Other advertised amenities (not priced): " + ", ".join(other))


def run_estimate():
    lat, lon = ss.f_lat, ss.f_lon
    notes = []
    if (lat is None) != (lon is None):
        notes.append("Latitude and longitude are only used together; distance to downtown was taken from the zip.")
        lat = lon = None
    laundry = ss.f_laundry
    model = rent_model.predict_rent(
        bedrooms=ss.f_beds, bathrooms=ss.f_baths, sqft=ss.f_sqft, zip_code=ss.f_zip.strip() or None,
        latitude=lat, longitude=lon,
        amenities={"in_unit_laundry": None if laundry == fa.UNKNOWN else laundry == "In-unit washer and dryer",
                   "shared_laundry": None if laundry == fa.UNKNOWN else laundry == "Shared in building",
                   "doorman_concierge": ss[ck(DOORMAN)]})
    choices = {k: ss["s_" + k] for k in fa.SELECTS}
    choices["floor"] = ss.f_floor
    choices["features"] = {f for f in fa.UNIT_FEATURES + fa.BUILDING_FEATURES if ss[ck(f)]}
    adj = fa.adjust(model["estimate"], choices)
    i = ss.unit_idx
    listed = ss.units[i].get("rent") if ss.details is not None and 0 <= i < len(ss.units) else None
    return {"model": model, "adj": adj, "listed": listed, "notes": notes}


if estimate_clicked:
    ss.result = run_estimate()

# ---------- estimate ----------
st.subheader("Estimate")
if ss.result is None:
    st.info("Fill in the unit above, then press Estimate rent.")
else:
    model, adj, listed = ss.result["model"], ss.result["adj"], ss.result["listed"]
    est = adj["estimate"]
    low, high = model["low"] * adj["factor"], model["high"] * adj["factor"]
    # Metric values are markdown, where a pair of "$" starts math, so dollar signs are escaped.
    range_help = ("About 80% of test listings in buildings the model had not seen fell between low and high. "
                  "This covers the model's error only; the study-based adjustments add more uncertainty.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estimated rent / mo", "\\${:,.0f}".format(est))
    c2.metric("Likely low", "\\${:,.0f}".format(low), help=range_help)
    c3.metric("Likely high", "\\${:,.0f}".format(high), help=range_help)
    if listed:
        c4.metric("Listed asking rent", "\\${:,.0f}".format(listed),
                  delta="{:+.0%} vs estimate".format(listed / est - 1), delta_color="off")
    for n in ss.result["notes"]:
        st.info(n)

    st.markdown("**How we got there**")
    money = lambda x, signed=False: ("{}${:,.0f}".format(("+" if x >= 0 else "-") if signed else "", abs(x)))
    rows = [{"Factor": "Model estimate", "Amount": money(model["estimate"]),
             "Why": "Size, location, laundry and doorman, fit on SF listings. " + model["reliability"][:1].upper() + model["reliability"][1:] + "."}]
    for r in adj["rows"]:
        pct = " ({:+.1f}%)".format(r["value"]) if r["unit"] == "pct" else ""
        rows.append({"Factor": r["factor"], "Amount": money(r["dollars"], signed=True),
                     "Why": "{}{}. {} Confidence: {}.".format(r["detail"], pct, r["note"], r["confidence"])})
    rows.append({"Factor": "Estimated rent", "Amount": money(est), "Why": "Model estimate plus adjustments."})
    st.table(pd.DataFrame(rows).set_index("Factor"))
    if not adj["rows"]:
        st.caption("No feature adjustments: every feature is Unknown, unchecked, or the same as the typical "
                   "building the model was trained on.")
    sources = {}
    for r in adj["rows"]:
        for src in r["sources"]:
            sources.setdefault(src["url"] or src["title"], src)
    if sources:
        with st.expander("Sources for the feature adjustments"):
            for src in sources.values():
                title = "[{}]({})".format(src["title"], src["url"]) if src.get("url") else src["title"]
                st.markdown("- {}, {} ({})".format(src["name"], title, src["year"]))
            st.caption("Adjustments are sized relative to the typical training building: having a feature most "
                       "training buildings advertise adds less, and lacking it subtracts a little.")
    if model["warnings"]:
        st.warning("\n".join("- " + w for w in model["warnings"]))

if ss.details is not None:
    with st.expander("Raw response (for checking fields)"):
        st.code("\n".join("{} = {}".format(p, json.dumps(v)[:200]) for p, v in rl.leaves(ss.details)), language=None)
