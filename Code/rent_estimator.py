#!/usr/bin/env python3
"""Rent estimator: the Python version of Structure_Rent_Estimator.html.

The HTML tool describes a unit, prices it from comparables, and shows a
recommended rent, a range, a confidence level, a base rent plus one dollar
adjustment per feature, the comps used, and a running record of estimates vs.
the rents units actually leased at. This script does the same thing, but the
pricing comes from regression coefficients you pass in instead of from an LLM.

Standard library only.

COEFFICIENTS
  Pass them with --coefs (a JSON file, or a rent_model.py written by
  train_rent_model.py), and/or one at a time with --coef NAME=VALUE (which
  overrides the file). The core ones also have their own flags (--intercept,
  --b-bedrooms, --b-bathrooms, --b-log-sqft, --b-sqft, --smearing).

  Coefficient names (one flat namespace):
    intercept                         constant (reference ZIP, reference levels)
    bedrooms, bathrooms, sqft,        numeric terms; age = as-of year - year built
      log_sqft, floor, age
    zip:94110, zip:other              ZIP fixed effects (reference ZIP has none);
      zip:_citywide                   zip:_citywide is used when no ZIP is given
    hood:Marina                       neighborhood effects
    ptype:Condo                       one per non-reference level of each field:
    condition:Fully renovated           ptype, condition, parking, laundry, pets,
    parking:Assigned garage space       utilities, term (levels listed in OPTIONS)
    unit:Dishwasher                   unit features (UNIT_FEATURES)
    building:Elevator                 building amenities (BUILDING_AMENITIES)

  With "form": "log" (the default) the model is
      log(rent) = intercept + sum(coef * x),  rent = exp(.) * smearing
  so a coefficient c on a yes/no feature means about (exp(c) - 1) * 100 percent.
  With "form": "linear" coefficients are dollars per unit of x.

  A JSON coefficient file looks like Code/rent_coefficients_example.json:
    {"form": "log", "smearing": 1.03,
     "interval": {"p10": -0.35, "p90": 0.31},  # out-of-sample residual quantiles
     "train_range": {"sqft": [150, 3708]},     # optional, used for warnings
     "means": {"floor": 2},                    # optional, fills missing inputs
     "placeholders": ["unit:Views"],           # optional, flagged in the output
     "coefficients": {"intercept": 4.7, "bedrooms": 0.12, ...}}

USAGE
  python rent_estimator.py estimate --coefs rent_coefficients_example.json \\
      --address "532 Bancroft Ave" --unit 4 --zip 94123 --beds 1 --baths 1 \\
      --sqft 750 --condition "Fully renovated" --laundry "In-unit washer and dryer" \\
      --feature Dishwasher --feature "Hardwood floors" --comps ../Data/comps.csv \\
      --save estimates.jsonl --summary

  python rent_estimator.py estimate --coefs rent_model.py --zip 94110 --beds 2 \\
      --baths 1 --sqft 900 --coef "unit:Views=0.04"

  python rent_estimator.py actual <estimate id> 3450 --store estimates.jsonl
  python rent_estimator.py history --store estimates.jsonl
  python rent_estimator.py export-coefs rent_model.py -o coefs.json
"""
import argparse
import csv
import datetime as dt
import importlib.util
import json
import math
import os
import statistics
import sys
import uuid

# ---------------------------------------------------------------------------
# Unit form (same fields and choices as the HTML tool)
# ---------------------------------------------------------------------------
OPTIONS = {
    "ptype": ["Apartment in larger building", "Flat in 2 to 4 unit building", "Condo",
              "Townhouse", "Single-family home", "In-law / ADU"],
    "condition": ["Original / dated", "Updated", "Fully renovated", "New construction"],
    "parking": ["None", "Street permit only", "Shared garage space", "Assigned garage space",
                "Two or more spaces"],
    "laundry": ["None", "Shared in building", "In-unit washer and dryer"],
    "pets": ["No pets", "Cats only", "Cats and dogs"],
    "utilities": ["None", "Water, sewer, garbage", "All utilities"],
    "term": ["12 months", "6 months", "Month-to-month"],
}
DEFAULTS = {"ptype": "Apartment in larger building", "condition": "Updated", "parking": "None",
            "laundry": "In-unit washer and dryer", "pets": "Cats and dogs",
            "utilities": "Water, sewer, garbage", "term": "12 months"}
FIELD_LABELS = {"ptype": "Property type", "condition": "Condition", "parking": "Parking",
                "laundry": "Laundry", "pets": "Pets", "utilities": "Utilities included",
                "term": "Lease term"}
UNIT_FEATURES = ["Dishwasher", "Hardwood floors", "Private outdoor space", "Views",
                 "Abundant natural light / corner", "Fireplace", "Walk-in closet",
                 "Air conditioning", "Central heat", "Extra storage", "Furnished",
                 "Home office / den"]
BUILDING_AMENITIES = ["Elevator", "Doorman / concierge", "Fitness center",
                      "Shared roof deck or yard", "Bike storage", "EV charging",
                      "Package lockers", "Pool"]
BATHS = ["1", "1.5", "2", "2.5", "3", "3.5+"]

BASE_NUMERIC = ["bedrooms", "bathrooms", "sqft", "log_sqft"]   # go into the base rent
ADJ_NUMERIC = ["floor", "age"]                                 # priced as adjustments
NUMERIC_LABELS = {"floor": "Floor", "age": "Building age"}
CATEGORY_PREFIXES = {"zip", "hood", "unit", "building", *OPTIONS}

# Amenity names used by train_rent_model.py, mapped onto this script's names.
MODEL_AMENITY_MAP = {
    "washer_dryer": ["laundry:In-unit washer and dryer"],
    "in_unit_laundry": ["laundry:In-unit washer and dryer"],
    "parking": ["parking:Shared garage space", "parking:Assigned garage space",
                "parking:Two or more spaces"],
    "parking_included": ["parking:Shared garage space", "parking:Assigned garage space",
                         "parking:Two or more spaces"],
    "balcony_patio": ["unit:Private outdoor space"],
    "private_outdoor": ["unit:Private outdoor space"],
    "air_conditioning": ["unit:Air conditioning"],
    "pool": ["building:Pool"],
    "utilities_included": ["utilities:Water, sewer, garbage", "utilities:All utilities"],
    "renovated": ["condition:Fully renovated"],
    "dishwasher": ["unit:Dishwasher"],
}
MODEL_NAME_MAP = {"beds": "bedrooms", "baths": "bathrooms"}

Z80 = 1.2816  # normal quantile for an 80% range when only a residual SD is given


def fmt(n):
    return "" if n is None else "$" + format(round(n), ",")


def signed(n):
    return ("+" if n >= 0 else "-") + "$" + format(abs(round(n)), ",")


def round_to(x, step=25):
    return int(step * round(x / step))


def bed_label(b):
    return "Studio" if b == 0 else ("5+ bed" if b >= 5 else f"{b:g} bed")


# ---------------------------------------------------------------------------
# Coefficients
# ---------------------------------------------------------------------------
def validate_name(name):
    if name == "intercept" or name in BASE_NUMERIC or name in ADJ_NUMERIC:
        return
    prefix, sep, level = name.partition(":")
    if not sep or prefix not in CATEGORY_PREFIXES or not level:
        raise ValueError(f"unknown coefficient name {name!r} (see the docstring for valid names)")
    valid = (OPTIONS.get(prefix) or (UNIT_FEATURES if prefix == "unit" else None)
             or (BUILDING_AMENITIES if prefix == "building" else None))
    if valid is not None and level not in valid:
        raise ValueError(f"{name!r}: {level!r} is not one of {valid}")


def model_from_dict(d):
    m = {"form": d.get("form", "log"), "smearing": float(d.get("smearing", 1.0)),
         "interval": d.get("interval"), "train_range": d.get("train_range", {}),
         "means": d.get("means", {}), "placeholders": set(d.get("placeholders", [])),
         "source": d.get("source", ""), "coefficients": {}}
    if m["form"] not in ("log", "linear"):
        raise ValueError('"form" must be "log" or "linear"')
    for k, v in d.get("coefficients", {}).items():
        validate_name(k)
        m["coefficients"][k] = float(v)
    if "intercept" not in m["coefficients"]:
        raise ValueError("coefficients must include an intercept")
    return m


def model_from_rent_model_py(path):
    """Convert a rent_model.py written by train_rent_model.py into a coefficient dict."""
    spec = importlib.util.spec_from_file_location("_rent_model_src", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pm = mod.PRICE_MODEL
    coefs = {"intercept": pm["const"]}
    for name, c in pm["coefficients"].items():
        if name.endswith("_unknown"):
            continue
        value = c["coef"] if isinstance(c, dict) else c
        name = MODEL_NAME_MAP.get(name, name)
        if name in BASE_NUMERIC:
            coefs[name] = value
    for name, value in getattr(mod, "AMENITY_EFFECTS", {}).items():
        for target in MODEL_AMENITY_MAP.get(name, []):
            coefs[target] = value
    zeff = pm.get("zip_effects", {})
    for z, v in zeff.items():
        if v:
            coefs["zip:" + z] = v
    if zeff and "city_intercept" in pm:
        coefs["zip:_citywide"] = round(pm["city_intercept"] - pm["const"], 5)
    tr = dict(pm.get("train_range") or {})
    sup = pm.get("support", {})
    if "beds" in sup:
        tr["bedrooms"] = sup["beds"]
    if "baths" in sup:
        tr["bathrooms"] = sup["baths"]
    if "zips" in sup:
        tr["zips"] = sup["zips"]
    means = {MODEL_NAME_MAP.get(k, k): v for k, v in pm.get("means", {}).items()}
    if "log_sqft" in means:
        means["sqft"] = round(math.exp(means["log_sqft"]))
    q = pm.get("log_residual_quantiles_out_of_sample")
    info = getattr(mod, "MODEL_INFO", {})
    return {"form": "log", "source": f"{os.path.basename(path)} (trained {info.get('trained', '?')})",
            "smearing": pm.get("smearing", 1.0),
            "interval": {"p10": q["p10"], "p90": q["p90"]} if q else None,
            "train_range": tr, "means": means, "coefficients": coefs}


def load_model(path=None, overrides=None):
    if path is None:
        d = {"coefficients": {}}
    elif path.endswith(".py"):
        d = model_from_rent_model_py(path)
    else:
        with open(path) as f:
            d = json.load(f)
    d = dict(d)
    d["coefficients"] = {**d.get("coefficients", {}), **(overrides or {})}
    return model_from_dict(d)


# ---------------------------------------------------------------------------
# Comparables (optional): any CSV with rent, beds, baths and ideally sqft, zip, date
# ---------------------------------------------------------------------------
COMP_COLUMNS = {
    "rent": ["rent", "actual_rent", "lease_rent", "asking_rent_usd_month", "market_rent",
             "Rent Price ($)"],
    "beds": ["beds", "bedrooms", "Bedrooms"],
    "baths": ["baths", "bathrooms", "Bathrooms"],
    "sqft": ["sqft", "square_feet", "Square Footage"],
    "zip": ["zip", "zip_code", "Zip Code"],
    "address": ["address", "property_address", "Formatted Address"],
    "unit": ["unit"],
    "date": ["date", "lease_start", "lease_start_date", "available_date", "retrieved_date"],
    "kind": ["kind", "status", "rent_basis"],
    "note": ["note", "quality_notes"],
}


def _num(s):
    try:
        return float(str(s).replace("$", "").replace(",", "").replace("+", "").strip())
    except ValueError:
        return None


def _date(s):
    s = (s or "").strip()
    for f in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return dt.datetime.strptime(s[:10], f).date()
        except ValueError:
            pass
    return None


def load_comps(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    cols = rows[0].keys()
    pick = {k: next((c for c in names if c in cols), None) for k, names in COMP_COLUMNS.items()}
    if not all(pick[k] for k in ("rent", "beds", "baths")):
        raise ValueError(f"{path}: needs rent, beds and baths columns")
    comps = []
    for r in rows:
        if str(r.get("comp_use", "")).strip().lower() == "exclude":
            continue
        g = lambda k: r.get(pick[k], "") if pick[k] else ""  # noqa: E731
        rent, beds, baths = _num(g("rent")), _num(g("beds")), _num(g("baths"))
        if not rent or beds is None or baths is None:
            continue
        kind = g("kind").lower()
        comps.append({"address": g("address"), "unit": g("unit"), "zip": g("zip")[:5],
                      "beds": beds, "baths": baths, "sqft": _num(g("sqft")), "rent": rent,
                      "date": _date(g("date")), "note": g("note"),
                      "kind": "Signed lease" if ("lease" in kind and "asking" not in kind)
                      or "signed" in kind else "Asking rent"})
    return comps


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------
def numeric_inputs(unit, as_of):
    sqft = unit.get("sqft")
    year = unit.get("year_built")
    return {"bedrooms": unit["beds"], "bathrooms": float(str(unit["baths"]).rstrip("+")),
            "sqft": sqft, "log_sqft": math.log(sqft) if sqft else None,
            "floor": unit.get("floor"), "age": (as_of.year - year) if year else None}


def location_terms(unit, model, warnings):
    """Return [(label, coef)] for ZIP and neighborhood effects."""
    c = model["coefficients"]
    has_zip = any(k.startswith("zip:") for k in c)
    terms = []
    z = (unit.get("zip") or "")[:5]
    if has_zip:
        if not z:
            terms.append(("Citywide average location", c.get("zip:_citywide", 0.0)))
            warnings.append("No ZIP given; using a citywide average location.")
        elif "zip:" + z in c:
            terms.append((f"ZIP {z}", c["zip:" + z]))
        elif "zip:other" in c:
            terms.append((f"ZIP {z} (pooled as 'other')", c["zip:other"]))
            warnings.append(f"ZIP {z} has its own effect only if the model saw enough listings "
                            "there; using the pooled 'other' ZIP effect.")
        else:
            terms.append((f"ZIP {z} (treated as the reference ZIP)", 0.0))
    else:
        warnings.append("The model has no location terms; the estimate reflects the average "
                        "location of its training data.")
    zips = model["train_range"].get("zips")
    if z and zips and z not in zips:
        warnings.append(f"ZIP {z} is outside the area the model was fit on.")
    hood = unit.get("neighborhood")
    if hood and "hood:" + hood in c:
        terms.append((f"Neighborhood {hood}", c["hood:" + hood]))
    return terms


def predict_core(x, loc, model):
    """Model value (log or dollars) for the base unit: intercept + location + size terms."""
    c = model["coefficients"]
    v = c["intercept"] + sum(coef for _, coef in loc)
    for k in BASE_NUMERIC:
        if c.get(k):
            v += c[k] * x[k]
    return v


def to_dollars(v, model):
    return math.exp(v) * model["smearing"] if model["form"] == "log" else v


def adjust_comp(comp, subject_x, subject_loc, model):
    """Comp rent moved to the subject's size and location using the model's own terms."""
    c = model["coefficients"]
    diff = sum(coef for _, coef in subject_loc)
    zc = "zip:" + comp["zip"] if comp["zip"] else None
    if comp["zip"] and any(k.startswith("zip:") for k in c):
        diff -= c.get(zc, c.get("zip:other", 0.0))
    cx = {"bedrooms": comp["beds"], "bathrooms": comp["baths"], "sqft": comp["sqft"],
          "log_sqft": math.log(comp["sqft"]) if comp["sqft"] else None}
    for k in BASE_NUMERIC:
        if c.get(k) and subject_x[k] is not None and cx[k] is not None:
            diff += c[k] * (subject_x[k] - cx[k])
    return comp["rent"] * math.exp(diff) if model["form"] == "log" else comp["rent"] + diff


def select_comps(comps, unit, as_of, limit=8):
    z, b = (unit.get("zip") or "")[:5], unit["beds"]
    baths = float(str(unit["baths"]).rstrip("+"))
    pool = [c for c in comps if abs(c["beds"] - b) <= 1]
    same_zip = [c for c in pool if z and c["zip"] == z]
    if len(same_zip) >= 10:
        pool = same_zip
    scored = []
    for c in pool:
        s = 2 * (c["zip"] == z and bool(z)) + 2 * (c["beds"] == b) + (abs(c["baths"] - baths) <= 0.5)
        if unit.get("sqft") and c["sqft"]:
            s += abs(c["sqft"] / unit["sqft"] - 1) <= 0.15
        age = (as_of - c["date"]).days if c["date"] else None
        if age is not None and age <= 180:
            s += 1
        sim = "High" if s >= 5 else "Medium" if s >= 3 else "Low"
        scored.append((s, -(age if age is not None else 10 ** 6), {**c, "similarity": sim,
                                                                  "age_days": age}))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [c for _, _, c in scored[:limit]]


def estimate(unit, model, comps=None, comp_weight=0.0, as_of=None):
    """Price one unit. Returns a dict shaped like the HTML tool's result."""
    as_of = as_of or dt.date.today()
    c = model["coefficients"]
    log_form = model["form"] == "log"
    warnings = []
    x = numeric_inputs(unit, as_of)
    for k in BASE_NUMERIC + ADJ_NUMERIC:
        if c.get(k) and x[k] is None:
            src = "sqft" if k == "log_sqft" else k
            mean = model["means"].get(k, model["means"].get(src))
            if mean is None and k in BASE_NUMERIC:
                raise ValueError(f"the model uses {k} but no {src} was given and the "
                                 "coefficient file has no mean to fall back on")
            if mean is None:
                warnings.append(f"{NUMERIC_LABELS[k]} not given; that term is left out.")
                continue
            x[k] = mean if k != "log_sqft" or "log_sqft" in model["means"] else math.log(mean)
            warnings.append(f"{src} not given; using the training average ({mean:g}).")
    for k, rng in model["train_range"].items():
        if k in x and x[k] is not None and isinstance(rng, list) and not rng[0] <= x[k] <= rng[1]:
            warnings.append(f"{k} = {x[k]:g} is outside the range the model was fit on "
                            f"({rng[0]:g} to {rng[1]:g}).")

    loc = location_terms(unit, model, warnings)
    base = to_dollars(predict_core(x, loc, model), model)

    # Adjustment terms: (factor, coefficient, reason)
    terms = []
    for field in OPTIONS:
        level = unit.get(field)
        name = f"{field}:{level}"
        if c.get(name):
            terms.append((name, f"{FIELD_LABELS[field]}: {level}", c[name]))
    for prefix, items in (("unit", unit.get("unit_features", [])),
                          ("building", unit.get("building_amenities", []))):
        for f in items:
            if c.get(f"{prefix}:{f}"):
                terms.append((f"{prefix}:{f}", f, c[f"{prefix}:{f}"]))
    for k in ADJ_NUMERIC:
        if c.get(k) and x[k] is not None:
            terms.append((k, f"{NUMERIC_LABELS[k]}: {x[k]:g}", c[k] * x[k]))

    # Dollar value of each adjustment. In log form the effects multiply; splitting the
    # total by each term's share of the log change makes the rows add up exactly and
    # does not depend on the order the features are listed in.
    S = sum(t[2] for t in terms)
    scale = (math.expm1(S) / S if abs(S) > 1e-12 else 1.0) if log_form else 1.0
    adjustments = []
    for name, label, coef in terms:
        amount = base * coef * scale if log_form else coef
        if name in ADJ_NUMERIC:
            reason = (f"{coef:+.3f} log points in total from the regression" if log_form
                      else "regression coefficient times the value")
        else:
            reason = (f"{math.expm1(coef) * 100:+.1f}% vs. the reference level in the regression"
                      if log_form else f"{signed(coef)} vs. the reference level in the regression")
        placeholder = name in model["placeholders"]
        if placeholder:
            reason += " (placeholder, not estimated from data)"
        adjustments.append({"factor": label, "amount": amount, "reason": reason,
                            "placeholder": placeholder})
    model_rent = base + sum(a["amount"] for a in adjustments)

    # Comparables: shown as evidence, optionally blended into the rent.
    used = select_comps(comps, unit, as_of) if comps else []
    for cp in used:
        cp["adjusted_rent"] = adjust_comp(cp, x, loc, model)
    comp_median = statistics.median(cp["adjusted_rent"] for cp in used) if used else None
    if comp_median and comp_weight > 0 and len(used) >= 3:
        blend = comp_weight * (comp_median - model_rent)
        adjustments.append({"factor": f"Comp blend ({comp_weight:.0%} weight)", "amount": blend,
                            "reason": f"moves toward the median comp adjusted to this unit "
                                      f"({fmt(comp_median)})", "placeholder": False})
    elif comp_weight > 0 and used:
        warnings.append("Fewer than 3 comps; comp blend skipped.")
    total = base + sum(a["amount"] for a in adjustments)
    recommended = round_to(total)

    # Confidence
    close = [cp for cp in used if cp["similarity"] == "High"
             and cp["age_days"] is not None and cp["age_days"] <= 180]
    level = 2 if len(close) >= 5 else 1 if used else 0
    why = ([f"{len(close)} close comps from the last 6 months"] if len(close) >= 5 else
           [f"{len(used)} comps, {len(close)} of them close and recent"] if used else
           ["no comparables supplied, so the estimate rests on the regression alone"])
    if comp_median and abs(comp_median / model_rent - 1) > 0.15:
        level -= 1
        why.append(f"comps adjusted to this unit (median {fmt(comp_median)}) differ from the "
                   f"regression by {abs(comp_median / model_rent - 1):.0%}")
    if any("outside" in w for w in warnings):
        level -= 1
        why.append("some inputs are outside what the model was fit on")
    if any(a["placeholder"] for a in adjustments):
        level -= 1
        why.append("some adjustments use placeholder coefficients")
    confidence = ["low", "medium", "high"][max(level, 0)]

    # Range from out-of-sample errors; wider when confidence is low.
    iv = model["interval"] or {}
    widen = 1.25 if confidence == "low" else 1.0
    if "p10" in iv and "p90" in iv:
        if log_form:
            low, high = total * math.exp(iv["p10"] * widen), total * math.exp(iv["p90"] * widen)
        else:
            low, high = total + iv["p10"] * widen, total + iv["p90"] * widen
    elif "sd" in iv:
        d = Z80 * iv["sd"] * widen
        low, high = (total * math.exp(-d), total * math.exp(d)) if log_form else (total - d, total + d)
    else:
        low, high = total * (1 - 0.06 * widen), total * (1 + 0.06 * widen)
        warnings.append("No residual quantiles or SD in the coefficient file; the range is a "
                        "flat +/-6% and is not calibrated.")

    size = ", ".join(p for p in (bed_label(unit["beds"]), f"{unit['baths']} bath",
                                 f"{x['sqft']:,.0f} sq ft" if x["sqft"] else "") if p)
    where = loc[0][0] if loc else "the model's average location"
    return {"recommended": recommended, "low": round_to(low), "high": round_to(high),
            "confidence": confidence, "confidence_reason": "; ".join(why).capitalize() + ".",
            "base": {"amount": base, "method": f"Regression rent for a {size} unit in {where} "
                                              "with reference features."},
            "adjustments": adjustments, "unrounded": total, "model_rent": model_rent,
            "comps": used, "comp_median_adjusted": comp_median, "warnings": warnings,
            "model_source": model["source"], "as_of": as_of.isoformat()}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def subject_line(u):
    parts = [u.get("address", "") + (f", Unit {u['unit']}" if u.get("unit") else ""),
             " ".join(p for p in (u.get("city"), u.get("zip")) if p),
             f"{bed_label(u['beds'])} / {u['baths']} bath",
             f"{u['sqft']:,} sq ft" if u.get("sqft") else ""]
    return " · ".join(p for p in parts if p)


def range_bar(r, width=60):
    vals = [r["low"], r["high"], r["recommended"]] + [c["rent"] for c in r["comps"]]
    lo, hi = min(vals), max(vals)
    pad = max((hi - lo) * 0.08, 100)
    lo, hi = lo - pad, hi + pad
    pos = lambda v: int(round((v - lo) / (hi - lo) * (width - 1)))  # noqa: E731
    bar = ["-"] * width
    for i in range(pos(r["low"]), pos(r["high"]) + 1):
        bar[i] = "="
    for c in r["comps"]:
        bar[pos(c["rent"])] = "o"
    bar[pos(r["recommended"])] = "|"
    return (f"  {fmt(lo)} [{''.join(bar)}] {fmt(hi)}\n"
            "  = recommended range, | recommended rent, o comp rents (as listed)")


def render_text(unit, r):
    out = [subject_line(unit), ""]
    sqft = unit.get("sqft")
    out.append(f"Recommended rent: {fmt(r['recommended'])} per month"
               + (f" · ${r['recommended'] / sqft:.2f} / sq ft" if sqft else "")
               + f"   [{r['confidence'].upper()} confidence]")
    out.append(f"Range: {fmt(r['low'])} to {fmt(r['high'])}")
    out.append(r["confidence_reason"])
    out += ["", range_bar(r), "", "How we got there"]
    rows = [("Base rent from regression", r["base"]["amount"], r["base"]["method"], True)]
    rows += [(a["factor"], a["amount"], a["reason"], False) for a in r["adjustments"]]
    w = max([len(f) for f, *_ in rows] + [len("Base plus adjustments (rent is rounded to $25)")]) + 2
    for f, amt, why, is_base in rows:
        out.append(f"  {f:<{w}}{fmt(amt) if is_base else signed(amt):>10}")
        out.append(f"      {why}")
    note = " (rent is rounded to $25)" if abs(r["unrounded"] - r["recommended"]) > 1 else ""
    out.append(f"  {'Base plus adjustments' + note:<{w}}{fmt(r['unrounded']):>10}")
    out += ["", f"Comparables used ({len(r['comps'])})"]
    if r["comps"]:
        out.append(f"  {'Address':<30}{'Bed/bath':>9}{'Sq ft':>7}{'Rent':>9}{'Adj. rent':>11}"
                   f"  {'Type':<13}{'Date':<11}Match")
        for cp in r["comps"]:
            addr = (cp["address"] + (f", Unit {cp['unit']}" if cp["unit"] else ""))[:29]
            out.append(f"  {addr:<30}{cp['beds']:>4g}/{cp['baths']:<4g}"
                       f"{(format(cp['sqft'], ',.0f') if cp['sqft'] else ''):>7}"
                       f"{fmt(cp['rent']):>9}{fmt(cp['adjusted_rent']):>11}  {cp['kind']:<13}"
                       f"{(cp['date'].isoformat() if cp['date'] else ''):<11}{cp['similarity']}")
        out.append("  Adj. rent moves each comp to this unit's beds, baths, size and ZIP with the "
                   "regression's own coefficients.")
    else:
        out.append("  None supplied (--comps). The estimate rests on the regression alone; check it "
                   "against current listings before using it.")
    if r["warnings"]:
        out += ["", "What could move this number"] + [f"  - {w}" for w in r["warnings"]]
    out += ["", f"Source: regression coefficients{' from ' + r['model_source'] if r['model_source'] else ''}"
                f", as of {r['as_of']}."]
    return "\n".join(out)


def summary_text(unit, r):
    """Plain-text summary for an owner (the HTML tool's 'Copy summary' button)."""
    addr = unit.get("address", "the unit") + (f", Unit {unit['unit']}" if unit.get("unit") else "")
    lines = [f"Recommended rent for {addr}: {fmt(r['recommended'])} per month (range "
             f"{fmt(r['low'])} to {fmt(r['high'])}, {r['confidence']} confidence).", "",
             f"Base rent from our pricing model: {fmt(r['base']['amount'])}. {r['base']['method']}"]
    lines += [f"{a['factor']}: {signed(a['amount'])}." for a in r["adjustments"]]
    lines += ["", f"Checked against {len(r['comps'])} comparable rentals." if r["comps"] else
              "No comparable rentals were attached to this estimate."]
    return "\n".join(lines)


def to_jsonable(r):
    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, list):
            return [conv(v) for v in o]
        if isinstance(o, dt.date):
            return o.isoformat()
        if isinstance(o, float):
            return round(o, 2)
        return o
    return conv(r)


# ---------------------------------------------------------------------------
# Saved estimates (the HTML tool's "Track record")
# ---------------------------------------------------------------------------
def read_store(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_store(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def save_estimate(path, unit, r):
    rec = {"id": uuid.uuid4().hex[:8], "created_at": dt.datetime.now().isoformat(timespec="seconds"),
           "subject": unit, "result": to_jsonable(r), "actual_rent": None}
    rows = read_store(path)
    rows.append(rec)
    write_store(path, rows)
    return rec["id"]


def record_actual(path, est_id, rent):
    rows = read_store(path)
    for row in rows:
        if row["id"] == est_id:
            row["actual_rent"] = rent
            row["actual_at"] = dt.datetime.now().isoformat(timespec="seconds")
            write_store(path, rows)
            return row
    raise KeyError(f"no saved estimate with id {est_id} in {path}")


def accuracy(rows):
    scored = [r for r in rows if r.get("actual_rent") and r["result"].get("recommended")]
    if not scored:
        return {"estimates": len(rows), "with_actual": 0}
    errs = [abs(r["actual_rent"] - r["result"]["recommended"]) / r["result"]["recommended"] * 100
            for r in scored]
    return {"estimates": len(rows), "with_actual": len(scored),
            "median_miss_pct": round(statistics.median(errs), 1),
            "within_5_pct": round(sum(e <= 5 for e in errs) / len(errs) * 100)}


def render_history(rows):
    if not rows:
        return "No saved estimates yet."
    out = [f"{'ID':<10}{'Date':<12}{'Unit':<34}{'Size':<14}{'Estimate':>10}{'Actual':>10}{'Variance':>10}"]
    for r in sorted(rows, key=lambda r: r["created_at"], reverse=True):
        s, est, act = r["subject"], r["result"]["recommended"], r.get("actual_rent")
        var = f"{(act - est) / est * 100:+.1f}%" if act else ""
        unit = (s.get("address", "") + (f", Unit {s['unit']}" if s.get("unit") else ""))[:33]
        out.append(f"{r['id']:<10}{r['created_at'][:10]:<12}{unit:<34}"
                   f"{bed_label(s['beds']) + ' / ' + str(s['baths']) + ' ba':<14}{fmt(est):>10}"
                   f"{fmt(act) if act else '':>10}{var:>10}")
    a = accuracy(rows)
    out.append("")
    out.append(f"{a['estimates']} estimates · {a['with_actual']} with actual rent"
               + (f" · {a['median_miss_pct']}% median miss · {a['within_5_pct']}% within 5%"
                  if a["with_actual"] else ""))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def parse_coef(s):
    name, sep, value = s.rpartition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"--coef expects NAME=VALUE, got {s!r}")
    try:
        return name.strip(), float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--coef {s!r}: {value!r} is not a number")


def add_estimate_args(p):
    m = p.add_argument_group("regression coefficients")
    m.add_argument("--coefs", help="coefficient JSON file, or a rent_model.py from train_rent_model.py")
    m.add_argument("--coef", action="append", type=parse_coef, default=[], metavar="NAME=VALUE",
                   help='set or override one coefficient, e.g. --coef "unit:Views=0.04" (repeatable)')
    for flag, name in (("--intercept", "intercept"), ("--b-bedrooms", "bedrooms"),
                       ("--b-bathrooms", "bathrooms"), ("--b-log-sqft", "log_sqft"),
                       ("--b-sqft", "sqft"), ("--b-floor", "floor"), ("--b-age", "age")):
        m.add_argument(flag, type=float, dest="b_" + name, help=f"coefficient on {name}")
    m.add_argument("--form", choices=["log", "linear"], help="override the model form")
    m.add_argument("--smearing", type=float, help="retransformation factor for log models")
    m.add_argument("--resid-p10", type=float, help="10th percentile of out-of-sample residuals")
    m.add_argument("--resid-p90", type=float, help="90th percentile of out-of-sample residuals")
    m.add_argument("--resid-sd", type=float, help="residual SD (used if no quantiles)")

    u = p.add_argument_group("unit")
    u.add_argument("--unit-json", help="read the unit from a JSON file (flags below override it)")
    u.add_argument("--address")
    u.add_argument("--unit")
    u.add_argument("--city")
    u.add_argument("--zip")
    u.add_argument("--hood", dest="neighborhood")
    u.add_argument("--beds", type=int, choices=range(0, 6), help="0 = studio, 5 = 5+")
    u.add_argument("--baths", choices=BATHS)
    u.add_argument("--sqft", type=int)
    u.add_argument("--floor", type=int)
    u.add_argument("--year", type=int, dest="year_built")
    for field in OPTIONS:
        u.add_argument("--" + field, choices=OPTIONS[field])
    u.add_argument("--feature", action="append", dest="unit_features", choices=UNIT_FEATURES,
                   help="unit feature (repeatable)")
    u.add_argument("--amenity", action="append", dest="building_amenities",
                   choices=BUILDING_AMENITIES, help="building amenity (repeatable)")
    u.add_argument("--notes")

    o = p.add_argument_group("comps and output")
    o.add_argument("--comps", help="CSV of comparable leases or listings")
    o.add_argument("--comp-weight", type=float, default=0.0,
                   help="0 to 1: how far to move the rent toward the median adjusted comp "
                        "(default 0 = regression only; needs 3+ comps)")
    o.add_argument("--as-of", type=dt.date.fromisoformat, help="pricing date, YYYY-MM-DD")
    o.add_argument("--save", metavar="STORE", help="append the estimate to this .jsonl file")
    o.add_argument("--summary", action="store_true", help="also print a summary for an owner")
    o.add_argument("--json", action="store_true", help="print the result as JSON")


def unit_from_args(a):
    unit = {"city": "San Francisco", "baths": "1", **DEFAULTS,
            "unit_features": [], "building_amenities": []}
    if a.unit_json:
        with open(a.unit_json) as f:
            unit.update(json.load(f))
    for k in ("address", "unit", "city", "zip", "neighborhood", "beds", "baths", "sqft", "floor",
              "year_built", "notes", "unit_features", "building_amenities", *OPTIONS):
        v = getattr(a, k)
        if v is not None:
            unit[k] = v
    if "beds" not in unit:
        raise SystemExit("--beds is required (0 for a studio)")
    return unit


def model_from_args(a):
    over = dict(a.coef)
    for name in ("intercept", "bedrooms", "bathrooms", "log_sqft", "sqft", "floor", "age"):
        v = getattr(a, "b_" + name)
        if v is not None:
            over[name] = v
    if not a.coefs and "intercept" not in over:
        raise SystemExit("give --coefs FILE, or at least --intercept plus the other coefficients")
    model = load_model(a.coefs, over)
    if a.form:
        model["form"] = a.form
    if a.smearing is not None:
        model["smearing"] = a.smearing
    if a.resid_p10 is not None and a.resid_p90 is not None:
        model["interval"] = {"p10": a.resid_p10, "p90": a.resid_p90}
    elif a.resid_sd is not None:
        model["interval"] = {"sd": a.resid_sd}
    return model


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_estimate_args(sub.add_parser("estimate", help="price a unit"))
    p = sub.add_parser("actual", help="record the rent a saved estimate actually leased at")
    p.add_argument("id")
    p.add_argument("rent", type=float)
    p.add_argument("--store", default="estimates.jsonl")
    p = sub.add_parser("history", help="list saved estimates and accuracy")
    p.add_argument("--store", default="estimates.jsonl")
    p = sub.add_parser("export-coefs", help="convert a rent_model.py into a coefficient JSON file")
    p.add_argument("model_py")
    p.add_argument("-o", "--out", default="-")
    a = ap.parse_args(argv)

    try:
        if a.cmd == "estimate":
            model = model_from_args(a)
            unit = unit_from_args(a)
            comps = load_comps(a.comps) if a.comps else None
            r = estimate(unit, model, comps, a.comp_weight, a.as_of)
            print(json.dumps(to_jsonable(r), indent=2) if a.json else render_text(unit, r))
            if a.summary:
                print("\n--- Summary for an owner ---\n" + summary_text(unit, r))
            if a.save:
                print(f"\nSaved as {save_estimate(a.save, unit, r)} in {a.save}. Record the leased "
                      f"rent later with: actual <id> <rent> --store {a.save}", file=sys.stderr)
        elif a.cmd == "actual":
            row = record_actual(a.store, a.id, a.rent)
            est = row["result"]["recommended"]
            print(f"Saved. Estimate {fmt(est)}, leased at {fmt(a.rent)} "
                  f"({(a.rent - est) / est * 100:+.1f}%).")
        elif a.cmd == "history":
            print(render_history(read_store(a.store)))
        elif a.cmd == "export-coefs":
            text = json.dumps(model_from_rent_model_py(a.model_py), indent=2)
            if a.out == "-":
                print(text)
            else:
                with open(a.out, "w") as f:
                    f.write(text + "\n")
    except (ValueError, KeyError, OSError) as e:
        raise SystemExit(f"error: {e}")


if __name__ == "__main__":
    main()
