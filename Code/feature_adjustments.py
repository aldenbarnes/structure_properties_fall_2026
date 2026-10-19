"""Rent adjustments for features rent_model does not estimate, using premiums from published studies.

rent_model is fit on SF listings and only keeps size, location, laundry and doorman. Everything
else here comes from outside research (see EFFECTS: each entry has its sources and a caveat), so
these adjustments are assumptions layered on top of the model, not something the model measured.

    adj = feature_adjustments.adjust(model_estimate, choices)
    adj["estimate"], adj["factor"], adj["rows"]

How an adjustment is sized
  A premium is applied relative to the typical building the model was trained on, so the model's
  baseline is not counted twice. For a yes/no feature advertised by a share p of training buildings,
  having it adds premium * (1 - p) and lacking it subtracts premium * p. Features the training
  listings never mention (views, storage, ...) have p = 0: having one adds the full premium and
  lacking one changes nothing. Multi-option features (parking, condition, ...) are priced relative
  to a BASELINE option that describes the typical training listing.

Overlaps (from the research): most building amenities show up mainly in newer, better buildings,
so their premiums are halved when Condition is New construction or Fully renovated; the floor
premium is mostly a view premium, so it is halved when a view is also priced, and floors below the
baseline get no discount (the one rent study found none at street level).

Percent premiums multiply the model estimate; dollar premiums are added after.

Research caveat: the sources were checked against search-engine extracts, not full documents, and
most are raw listing comparisons or renter surveys rather than size-controlled studies. Values were
set below the headline figures, and most are low confidence.
"""
import csv
from pathlib import Path

TRAINING_PATH = Path(__file__).resolve().parent.parent / "Data" / "realty_training.csv"

# ---------- what the user picks ----------
UNKNOWN = "Unknown"
SELECTS = {   # key: (label, options)
    "property_type": ("Property type", ["Apartment in larger building", "Flat in 2 to 4 unit building", "Condo",
                                        "Townhouse", "Single-family home", "In-law / ADU"]),
    "condition": ("Condition", ["Original / dated", "Updated", "Fully renovated", "New construction"]),
    "parking": ("Parking included in rent", ["None", "Street permit only", "Shared garage space",
                                             "Assigned garage space", "Two or more spaces"]),
    "pets": ("Pets", ["No pets", "Cats only", "Cats and dogs"]),
    "utilities": ("Utilities included", ["None", "Water, sewer, garbage", "All utilities"]),
    "lease_term": ("Lease term", ["12 months", "6 months", "Month-to-month"]),
    "view": ("View", ["None", "City or partial view", "Water or bay view"]),
}
# Option that matches the typical training listing (large, professionally managed SF buildings).
BASELINE = {
    "property_type": "Apartment in larger building",
    "condition": "Updated",
    "parking": "None",            # parking in these buildings is usually rented separately
    "pets": "Cats and dogs",      # most large managed SF buildings allow both
    "utilities": "None",          # 8% of training buildings advertise utilities included
    "lease_term": "12 months",
    "view": "None",
}
UNIT_FEATURES = ["Dishwasher", "Hardwood floors", "Private outdoor space",
                 "Abundant natural light / corner", "Fireplace", "Walk-in closet", "Air conditioning",
                 "Central heat", "Extra storage", "Furnished", "Home office / den"]
BUILDING_FEATURES = ["Elevator", "Fitness center", "Shared roof deck or yard", "Bike storage",
                     "EV charging", "Package lockers", "Pool"]
FLOOR_BASELINE = 3       # floor premiums are measured from a typical mid-rise floor
FLOOR_CAP = 19           # 16 floors x 0.5% = the +8% cap the research suggests
# Building amenities whose premium mostly reflects newer, higher-end buildings.
OVERLAPS_CONDITION = {"Elevator", "Fitness center", "Pool", "Shared roof deck or yard", "Package lockers",
                      "EV charging", "Bike storage", "Air conditioning", "Central heat"}
CONDITION_OVERLAP = {"New construction", "Fully renovated"}

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
# value/low/high are percent of rent ("pct", 5 = +5%) or dollars per month ("usd"). Collected Oct 2026.
def _src(name, year, title, url):
    return {"name": name, "year": year, "title": title, "url": url}


def _e(value, unit, low, high, confidence, note, *sources):
    return {"value": value, "unit": unit, "low": low, "high": high, "confidence": confidence,
            "note": note, "sources": list(sources)}


APT_ADVISOR_21 = _src("ApartmentAdvisor", 2021, "ApartmentAdvisor Study Reveals the Apartment Amenities That Raise Rents Most",
                      "https://www.accessnewswire.com/newsroom/en/business-and-professional-services/apartmentadvisor-study-reveals-the-apartment-amenities-that-raise-675079")
APT_ADVISOR_24 = _src("ApartmentAdvisor", 2024, "Which Apartment Amenities Impact Rent Prices Most?",
                      "https://www.apartmentadvisor.com/owners/post/which-apartment-amenities-impact-rent-prices-most")
JVM = _src("Multifamily Executive (JVM Realty quoted)", None, "Renovations Done Right",
           "https://www.multifamilyexecutive.com/design-development/renovations-done-right_o")
GALLIN = _src("Gallin and Verbrugge, Federal Reserve Bank of Cleveland WP 16-30", 2016,
              "Panel Data Estimates of Age-Rent Profiles for Rental Housing",
              "https://www.clevelandfed.org/publications/working-paper/2016/wp-1630-age-rent-profiles")
DANTON = _src("Danton and Himbert, Journal of Urban Economics 107", 2018, "Residential vertical rent curves",
              "https://ideas.repec.org/a/eee/juecon/v107y2018icp89-100.html")
NEIGHBOR = _src("Neighbor.com", None, "San Francisco Multifamily Parking Guide: How Much to Charge for Parking in San Francisco, CA",
                "https://www.neighbor.com/storage-blog/?p=15886")
PETS = _src("Multi-Housing News (citing Carlisle-Frank, Frank and Nielsen, Anthrozoos)", None,
            "Pet-Friendly Apartments: What You Need to Know",
            "https://www.multihousingnews.com/pet-friendly-apartments-what-you-need-to-know/")
SFPUC = _src("SFPUC (Rate Fairness Board rates presentation)", 2026, "Rates Presentation (6a)",
             "https://www.sfpuc.gov/zh-TW/sites/default/files/agendas-minutes-rate-fairness-board/6a-Rates-Presentation.pdf")
RENTHOP_FEATURES = _src("RentHop", 2017, "The Features That Are Increasing Your Rent",
                        "https://www.renthop.com/studies/national/the-features-that-are-increasing-your-rent")
NO_STUDY = "Rule of thumb, no direct evidence found."

EFFECTS = {
    ("condition", "Original / dated"): _e(-6, "pct", -12, -3, "low",
        "Operators report renovated units rent about 10 to 15% above unrenovated ones; operator reports, not controlled studies.",
        JVM, _src("Wealth Management (Avanath quoted)", None,
                  "How to Implement Value-Add Strategies at Apartment Buildings Without Making Them Too Expensive",
                  "https://www.wealthmanagement.com/real-estate/how-to-implement-value-add-strategies-at-apartment-buildings-without-making-them-too-expensive")),
    ("condition", "Updated"): _e(0, "pct", 0, 0, "high", "Reference level."),
    ("condition", "Fully renovated"): _e(5, "pct", 3, 10, "low",
        "A full renovation typically earns a mid-single-digit premium over a merely updated unit.", JVM),
    ("condition", "New construction"): _e(12, "pct", 6, 20, "medium",
        "Rents fall quickly in a building's first years, then level off; NYC prewar elevator 1-beds rent about 10 to 12% below newer ones.",
        GALLIN, _src("RentHop", 2017, "Building Ages and Rents in New York",
                     "https://www.renthop.com/research/building-age-and-rents-in-new-york/")),
    ("floor", "per_floor"): _e(0.5, "pct", 0.3, 1.0, "low",
        "Rent rises with each floor above the ground, mostly because of better views (study sample not confirmed as US).", DANTON),
    ("view", "None"): _e(0, "pct", 0, 0, "high", "Reference level."),
    ("view", "City or partial view"): _e(3, "pct", 1, 6, "low",
        "High-view Manhattan office space rents about 6% above low-view space; residential city views are likely worth less.",
        _src("Sidewalk Labs (Sidewalk Talk)", None, "The view from your office window: what's it worth?",
             "https://medium.com/sidewalk-talk/the-view-from-your-office-window-whats-it-worth-97dc01189d52")),
    ("view", "Water or bay view"): _e(7, "pct", 3, 12, "low",
        "US apartment data shows water-view units renting about 10% higher; depends on how good and permanent the view is.",
        _src("Wealth Management (CoStar and Bainbridge data)", None, "Waterfront Properties Earn Extra Rent",
             "https://www.wealthmanagement.com/real-estate/waterfront-properties-earn-extra-rent")),
    ("parking", "None"): _e(0, "usd", 0, 0, "high", "Reference level."),
    ("parking", "Street permit only"): _e(0, "usd", 0, 0, "medium",
        "A residential permit is paid to SFMTA separately (about $224 a year), so it doesn't change the rent.",
        _src("SFMTA", 2026, "Renew Your Annual Residential Area Permit", "https://www.sfmta.com/es/node/1396")),
    ("parking", "Shared garage space"): _e(175, "usd", 100, 275, "low",
        "SF buildings charge roughly $150 a month or more for covered parking; a shared spot is worth less than an assigned one.", NEIGHBOR),
    ("parking", "Assigned garage space"): _e(250, "usd", 150, 400, "medium",
        "Separate SF parking fees run about $154 (covered average) to $362 a month; nationally a bundled space adds about 17% to rent.",
        NEIGHBOR, _src("Gabbe and Pierce, ACCESS Magazine", 2017, "The Hidden Cost of Bundled Parking",
                       "https://accessmagazine.org/spring-2017/the-hidden-cost-of-bundled-parking/")),
    ("parking", "Two or more spaces"): _e(450, "usd", 275, 700, "low",
        "About twice one space, a bit less since a second space is worth less to most households; an extrapolation.", NEIGHBOR),
    ("pets", "No pets"): _e(0, "pct", 0, 0, "high", "Reference level."),
    ("pets", "Cats only"): _e(1.5, "pct", 0, 4, "low",
        "Pet-friendly landlords earn somewhat higher rents; limited policies earn less than fully open ones.", PETS),
    ("pets", "Cats and dogs"): _e(3, "pct", 0, 8, "low",
        "Old survey data reports about 12% higher rents with no pet limits, likely including pet rent; a much smaller figure is used.",
        PETS, _src("Best Friends Animal Society (citing a 2011 Journal of Real Estate Finance and Economics study)", None,
                   "Pet-Friendly Rentals Increase Profitability",
                   "https://bestfriends.org/pet-care-resources/pet-friendly-rentals-increase-profitability-landlords")),
    ("utilities", "None"): _e(0, "usd", 0, 0, "high", "Reference level: tenant pays all utilities."),
    ("utilities", "Water, sewer, garbage"): _e(120, "usd", 85, 170, "medium",
        "Average SF single-family water and sewer is about $171 a month (apartments use less); garbage is about $45 per unit in small buildings.",
        SFPUC, _src("City and County of San Francisco", 2025, "San Francisco Refuse Rates October 2025 to September 2026",
                    "https://www.sf.gov/es/reports--october-2025--san-francisco-refuse-rates-october-2025-september-2026")),
    ("utilities", "All utilities"): _e(250, "usd", 180, 350, "low",
        "Water, sewer and garbage plus typical PG&E electricity and gas (roughly $100 to $180, a rule of thumb); grows with unit size.", SFPUC),
    ("lease_term", "12 months"): _e(0, "pct", 0, 0, "high", "Reference level."),
    ("lease_term", "6 months"): _e(5, "pct", 2, 10, "low",
        NO_STUDY + " Pricing software charges more for shorter terms.",
        _src("Multifamily Executive", None, "Flexible Lease Strategies Via Revenue Management Software",
             "https://www.multifamilyexecutive.com/technology/flexible-lease-strategies-via-revenue-management-software_o")),
    ("lease_term", "Month-to-month"): _e(10, "pct", 5, 20, "low",
        "Industry sources claim 20 to 40%; a conservative 10% is used, for new month-to-month listings only.",
        _src("Avail (industry blog, no data cited)", None, "Month-to-Month Leases",
             "https://www.avail.com/education/articles/month-to-month-leases")),
    ("property_type", "Apartment in larger building"): _e(0, "pct", 0, 0, "high", "Reference level."),
    ("property_type", "Flat in 2 to 4 unit building"): _e(0, "pct", -3, 3, "low",
        "No credible evidence of a size-adjusted difference."),
    ("property_type", "Condo"): _e(5, "pct", 0, 12, "low",
        "Rented condos rented 20 to 25% above purpose-built rentals in Vancouver, mostly from newer buildings and finishes.",
        _src("City of Vancouver", 2009, "The Housing Role of the Rented Condo Stock",
             "https://vancouver.ca/docs/policy/housing-role-rented-condo-stock.pdf")),
    ("property_type", "Townhouse"): _e(5, "pct", 0, 10, "low",
        NO_STUDY + " More privacy and no unit above."),
    ("property_type", "Single-family home"): _e(8, "pct", 3, 15, "low",
        "Single-family homes ask about 20% more than apartments nationally, not adjusted for size, so a smaller premium is used.",
        _src("Zillow (via rethinking65)", 2025, "Single-family Home Rents a Record 20% Above Apartments",
             "https://rethinking65.com/single-family-home-rents-a-record-20-above-apartments")),
    ("property_type", "In-law / ADU"): _e(-5, "pct", -12, 0, "low",
        "Most surveyed ADUs rent below market, partly because many go to family or friends.",
        _src("Terner Center (via HCD)", 2017, "Accessory Dwelling Unit Update",
             "https://www.hcd.ca.gov/policy-research/docs/Accessory-Dwelling-Unit-Update_Terner-Center_December-2017.pdf"),
        _src("Portland State University", None, "Many accessory dwellings in Portland rent for less than apartments, PSU study finds",
             "https://www.pdx.edu/news/many-accessory-dwellings-portland-rent-less-apartments-psu-study-finds")),
    ("feature", "Dishwasher"): _e(1.5, "pct", 1, 3, "medium",
        "Listing data puts the dishwasher premium at about 1 to 2%; overlaps with an updated kitchen.", APT_ADVISOR_21, APT_ADVISOR_24),
    ("feature", "Hardwood floors"): _e(1, "pct", 0, 3, "low",
        "Observed rents show under 1%, though many renters say they would pay $75 more; largely part of condition.", APT_ADVISOR_21,
        _src("Multifamily Executive (J Turner Research renter survey)", None,
             "Cost Conscious: What Renters Want in the Next-Gen Apartment",
             "https://www.multifamilyexecutive.com/property-management/apartment-trends/cost-conscious-what-renters-want-in-the-next-gen-apartment_o")),
    ("feature", "Private outdoor space"): _e(4, "pct", 2, 7, "medium",
        "National listing data: balconies about +7%, patios about +5%; trimmed because they are more common in newer buildings.",
        APT_ADVISOR_21),
    ("feature", "Abundant natural light / corner"): _e(2, "pct", 0, 5, "low",
        NO_STUDY + " Managers say renters prefer corner units.",
        _src("Multifamily Dive (NAA conference panel)", None, "How the right unit amenities can boost rent",
             "https://www.multifamilydive.com/news/how-the-right-unit-amenities-can-boost-rent/722979/")),
    ("feature", "Fireplace"): _e(1, "pct", 0, 2, "low",
        "An older Dallas study found about +2%; many SF fireplaces are decorative, so less is used.",
        _src("Christopher A. Thomson, MIT MSRED thesis", 1999,
             "Pricing Apartment Attributes: A Hedonic Analysis of the Dallas/Fort Worth Market",
             "https://dspace.mit.edu/handle/1721.1/70326")),
    ("feature", "Walk-in closet"): _e(0.5, "pct", 0, 2, "low", NO_STUDY + " Most of the value is already in square footage."),
    ("feature", "Air conditioning"): _e(1, "pct", 0, 3, "low",
        "About +2% nationally in listing data; SF's mild climate makes it worth less.", APT_ADVISOR_21),
    ("feature", "Central heat"): _e(0, "pct", 0, 2, "low", "No credible evidence; mostly signals a newer building."),
    ("feature", "Extra storage"): _e(1, "pct", 0, 2, "low", NO_STUDY + " Storage is scarce in SF."),
    ("feature", "Furnished"): _e(10, "pct", 5, 20, "low",
        "In 2017, furnished single-person SF listings asked about $300 a month more than unfurnished ones.",
        _src("Nestpick Furnished Apartment Index (via BXJ Magazine)", 2017, "Comparing the Costs of Apartment Rentals",
             "https://bxjmag.com/comparing-the-costs-of-apartment-rentals")),
    ("feature", "Home office / den"): _e(2, "pct", 0, 4, "low", NO_STUDY + " Mostly captured by square footage."),
    ("feature", "Elevator"): _e(2, "pct", 0, 5, "low",
        "Raw listing comparisons show about +7%, mostly reflecting newer, larger buildings.", RENTHOP_FEATURES, APT_ADVISOR_21),
    ("feature", "Fitness center"): _e(2.5, "pct", 1, 5, "low",
        "Buildings with gyms rent about 5% higher, largely because they are newer Class A buildings; about half is used.",
        APT_ADVISOR_24),
    ("feature", "Shared roof deck or yard"): _e(1.5, "pct", 0, 4, "low",
        "National listing data puts a yard at about 5%; a shared space is worth well under a private one.", APT_ADVISOR_24),
    ("feature", "Bike storage"): _e(0, "pct", 0, 1, "low", "No credible evidence; at most a few dollars a month."),
    ("feature", "EV charging"): _e(20, "usd", 0, 50, "low",
        "Interested renters say they would pay about $28 a month (survey, not observed rents).",
        _src("Smart Cities Dive (NMHC/Grace Hill 2022 renter survey)", 2022, "Demand soars for EV charging at apartments",
             "https://www.smartcitiesdive.com/news/demand-soars-ev-charging-apartments-electric-vehicles/638624/")),
    ("feature", "Package lockers"): _e(10, "usd", 0, 37, "low",
        "Interested renters say they would pay about $37 a month, but most would pay nothing (surveys).",
        _src("Swiftlane (summary of NMHC renter survey)", None, "NMHC Renter Preferences Survey Report Overview",
             "https://swiftlane.com/blog/renter-preferences-survey-report-summary/")),
    ("feature", "Pool"): _e(2, "pct", 0, 5, "low",
        "Raw comparisons show +8% nationally, but pool buildings are mostly new luxury towers and SF weather limits use.",
        APT_ADVISOR_21, _src("Brick Underground (Nestio data)", 2017, "How much does it cost to rent in a building with a pool",
                             "https://www.brickunderground.com/rent/how-much-does-it-cost-to-rent-in-a-building-with-a-pool")),
}


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
# Training listings tag "Furnished" when furnished units are offered, not when the unit is furnished,
# so a typical unfurnished unit should not be marked down for lacking it.
PREVALENCE["Furnished"] = 0.0


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
              "features": set of checked UNIT_FEATURES + BUILDING_FEATURES,
              "unticked_means_no": True if an unchecked feature is known to be missing
                                   (False: unchecked = unknown, so nothing is subtracted)}
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
    steps = max(0, min(floor, FLOOR_CAP) - FLOOR_BASELINE) if floor is not None else 0
    if eff and steps:
        detail = "Floor {} ({} floors above floor {})".format(int(floor), int(steps), FLOOR_BASELINE)
        view = choices.get("view")
        if view and view not in (UNKNOWN, "None"):
            steps, detail = steps / 2, detail + ", halved because the view is priced separately"
        rows.append(_row("Floor", eff, steps, detail))
    checked = choices.get("features") or set()
    for feat in UNIT_FEATURES + BUILDING_FEATURES:
        eff = EFFECTS.get(("feature", feat))
        if not eff or not eff["value"]:
            continue
        p = PREVALENCE.get(feat, 0.0)
        if feat in checked:
            share, detail = 1.0 - p, "Has it"
        elif p > 0 and choices.get("unticked_means_no", True):
            share, detail = -p, "Doesn't have it"
        else:
            continue
        if p > 0:
            detail += " ({:.0%} of training buildings advertise it)".format(p)
        if feat in OVERLAPS_CONDITION and choices.get("condition") in CONDITION_OVERLAP:
            share, detail = share / 2, detail + ", halved because it overlaps with {}".format(choices["condition"].lower())
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
