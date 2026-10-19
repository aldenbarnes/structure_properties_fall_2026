"""Rent adjustments for features rent_model does not estimate, using premiums from published studies.

rent_model is fit on SF listings and only keeps size, location, laundry and doorman. Everything
else here comes from outside research (see EFFECTS: each entry has its sources and a caveat), so
these adjustments are assumptions layered on top of the model, not something the model measured.

    adj = feature_adjustments.adjust(model_estimate, choices)
    adj["estimate"], adj["low_factor"], adj["rows"]

How an adjustment is sized
  A premium is applied relative to the typical building the model was trained on, so the model's
  baseline is not counted twice. For a yes/no feature advertised by a share p of training buildings,
  having it adds premium * (1 - p) and lacking it subtracts premium * p. Features the training
  listings never mention (views, storage, ...) have p = 0: having one adds the full premium and
  lacking one changes nothing. Multi-option features (parking, condition, ...) are priced relative
  to a BASELINE option that describes the typical training listing.

Percent premiums multiply the model estimate; dollar premiums are added after.
"""
import csv
from pathlib import Path

TRAINING_PATH = Path(__file__).resolve().parent.parent / "Data" / "realty_training.csv"

# ---------- what the user picks ----------
UNKNOWN = "Unknown"
SELECTS = {   # key: (label, options); the first option listed after Unknown is not special
    "property_type": ("Property type", ["Apartment in larger building", "Flat in 2 to 4 unit building", "Condo",
                                        "Townhouse", "Single-family home", "In-law / ADU"]),
    "condition": ("Condition", ["Original / dated", "Updated", "Fully renovated", "New construction"]),
    "parking": ("Parking included in rent", ["None", "Street permit only", "Shared garage space",
                                             "Assigned garage space", "Two or more spaces"]),
    "pets": ("Pets", ["No pets", "Cats only", "Cats and dogs"]),
    "utilities": ("Utilities included", ["None", "Water, sewer, garbage", "All utilities"]),
    "lease_term": ("Lease term", ["12 months", "6 months", "Month-to-month"]),
}
# Option that matches the typical training listing (large, professionally managed SF buildings).
BASELINE = {
    "property_type": "Apartment in larger building",
    "condition": "Updated",
    "parking": "None",            # parking in these buildings is usually rented separately
    "pets": "Cats and dogs",      # most large managed SF buildings allow both
    "utilities": "None",          # 8% of training buildings advertise utilities included
    "lease_term": "12 months",
}
UNIT_FEATURES = ["Dishwasher", "Hardwood floors", "Private outdoor space", "Views",
                 "Abundant natural light / corner", "Fireplace", "Walk-in closet", "Air conditioning",
                 "Central heat", "Extra storage", "Furnished", "Home office / den"]
BUILDING_FEATURES = ["Elevator", "Fitness center", "Shared roof deck or yard", "Bike storage",
                     "EV charging", "Package lockers", "Pool"]
FLOOR_BASELINE = 3       # floor premiums are measured from a typical mid-rise floor
FLOOR_CAP = 20           # don't extrapolate the per-floor premium past this

# Advertised amenity names (normalized: lowercase, punctuation removed) that mean each checkbox.
# Used both to pre-check boxes from a RealtyAPI listing and to measure training prevalence.
AMENITY_NAMES = {
    "Dishwasher": ["dishwasher"],
    "Hardwood floors": ["hardwood floors", "hardwood flooring", "hardwood"],
    "Private outdoor space": ["balcony", "patio", "deck", "private outdoor space", "terrace"],
    "Fireplace": ["fireplace"],
    "Walk-in closet": ["walk in closets", "walk in closet"],
    "Air conditioning": ["air conditioning", "ac", "central air"],
    "Central heat": ["heating", "central heat", "central heating"],
    "Furnished": ["furnished"],
    "Home office / den": ["den", "office", "home office"],
    "Views": ["view", "views", "city view", "bay view", "water view"],
    "Extra storage": ["storage", "storage units", "extra storage"],
    "Elevator": ["elevator"],
    "Fitness center": ["gym", "fitness center", "fitness centre"],
    "Shared roof deck or yard": ["roof terrace", "roof deck", "rooftop deck", "yard", "courtyard"],
    "Bike storage": ["bike storage", "bicycle storage"],
    "EV charging": ["car charging station", "ev charging", "electric vehicle charging"],
    "Package lockers": ["package service", "package lockers", "package receiving"],
    "Pool": ["pool"],
}

# ---------- premiums from published research ----------
# value/low/high are percent of rent ("pct", 5 = +5%) or dollars per month ("usd").
EFFECTS = {}   # filled in below


def _norm(s):
    import re
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def training_prevalence(path=TRAINING_PATH):
    """Share of training buildings advertising each checkbox feature (each building counted once)."""
    buildings = {}
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                buildings[r["listing_key"]] = {_norm(t) for t in (r.get("amenities") or "").split(";") if t.strip()}
    except OSError:
        return {}
    n = len(buildings) or 1
    return {feat: sum(1 for toks in buildings.values() if any(_norm(x) in toks for x in names)) / n
            for feat, names in AMENITY_NAMES.items()}


PREVALENCE = training_prevalence()


def checks_from_amenities(names):
    """Checkbox features advertised in a list of amenity names (e.g. from RealtyAPI)."""
    have = {_norm(n) for n in names}
    return {feat for feat, aliases in AMENITY_NAMES.items() if any(_norm(a) in have for a in aliases)}


def _row(label, effect, delta_share, detail):
    """One adjustment: `delta_share` of the full premium (1 = full, -p = lacking a common feature)."""
    return {"factor": label, "detail": detail, "unit": effect["unit"],
            "value": effect["value"] * delta_share, "low": effect["low"] * delta_share,
            "high": effect["high"] * delta_share, "confidence": effect["confidence"],
            "note": effect["note"], "sources": effect["sources"]}


def adjustments(choices):
    """List of adjustment rows for the user's choices.

    choices: {"property_type": "...", "condition": "...", ..., "floor": 4 or None,
              "features": set of checked UNIT_FEATURES + BUILDING_FEATURES}
    """
    rows = []
    for key, (label, _) in SELECTS.items():
        picked = choices.get(key)
        if not picked or picked == UNKNOWN or picked == BASELINE[key]:
            continue
        mine, base = EFFECTS.get((key, picked)), EFFECTS.get((key, BASELINE[key]))
        if not mine:
            continue
        if base and base["value"]:
            mine = dict(mine, value=mine["value"] - base["value"], low=mine["low"] - base["low"],
                        high=mine["high"] - base["high"])
        if mine["value"] == 0:
            continue
        rows.append(_row(label, mine, 1.0, "{} (vs. typical: {})".format(picked, BASELINE[key])))
    floor = choices.get("floor")
    eff = EFFECTS.get(("floor", "per_floor"))
    if floor is not None and eff and floor != FLOOR_BASELINE:
        steps = min(floor, FLOOR_CAP) - FLOOR_BASELINE
        if steps:
            rows.append(_row("Floor", eff, steps, "Floor {} ({:+d} floors vs. floor {})".format(
                int(floor), steps, FLOOR_BASELINE)))
    checked = choices.get("features") or set()
    for feat in UNIT_FEATURES + BUILDING_FEATURES:
        eff = EFFECTS.get(("feature", feat))
        if not eff or not eff["value"]:
            continue
        p = PREVALENCE.get(feat, 0.0)
        if feat in checked:
            share, detail = 1.0 - p, "Has it"
        elif p > 0:
            share, detail = -p, "Doesn't have it"
        else:
            continue
        if p > 0:
            detail += " ({:.0%} of training buildings advertise it)".format(p)
        rows.append(_row(feat, eff, share, detail))
    return rows


def adjust(model_estimate, choices):
    """Apply adjustments to a model estimate.

    Returns {"estimate", "pct_total", "usd_total", "low_factor", "high_factor", "rows"}; each row gets
    a "dollars" field. low/high_factor rescale the model's range for the combined adjustment.
    """
    rows = adjustments(choices)
    pct = 1.0
    for r in rows:
        if r["unit"] == "pct":
            pct *= 1 + r["value"] / 100
    usd = sum(r["value"] for r in rows if r["unit"] == "usd")
    for r in rows:
        r["dollars"] = model_estimate * r["value"] / 100 if r["unit"] == "pct" else r["value"]
    est = model_estimate * pct + usd
    return {"estimate": est, "pct_total": (pct - 1) * 100, "usd_total": usd,
            "factor": est / model_estimate if model_estimate else 1.0, "rows": rows}
