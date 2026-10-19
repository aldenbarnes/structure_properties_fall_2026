"""SF rent model fit on available-unit listings (lasso, penalty chosen by cross-validation).

Trained 2026-10-19 on 847 units in 80 buildings.
Standard library only. Drop this file next to app.py and:

    import rent_model
    r = rent_model.predict_rent(bedrooms=1, bathrooms=1, sqft=700, zip_code="94107",
                                amenities={"in_unit_laundry": True, "gym": False})
    r["estimate"], r["low"], r["high"], r["warnings"]

READ THIS BEFORE USING THE NUMBERS
  Size, bedrooms and location explain most of the rent. Building amenities add
  little that can be measured reliably (see MODEL_INFO["validation"]): the
  lasso keeps only the ones that survive cross-validation, and each kept effect
  has a wide interval (PRICE_MODEL["coefficients"][name]["ci95"]). Amenity and
  zip effects are associations, not proven prices. The estimate is a comp-based
  asking-rent range, not a recommended rent. The vacancy curve
  (predict_vacancy) is still a placeholder.

The model is log(asking monthly rent) = const + sum(coef * variable), and
rent = exp(that) * smearing. For a yes/no variable, exp(coef) - 1 is the percent
rent difference. Beds, baths, log(sqft) and distance to downtown are never
penalized; zip and amenity effects are shrunk toward zero by the lasso, and those
set exactly to zero are listed in MODEL_INFO["set_to_zero_by_lasso"].
"""
import math

MODEL_INFO = {'trained': '2026-10-19',
 'source': '201370d2-realty_training.csv (RealtyAPI available units; advertised asking rents)',
 'method': 'lasso (L1), penalty by grouped cross-validation',
 'penalty_fraction_of_max': 0.3919,
 'n_raw': 886,
 'n_used': 847,
 'n_buildings': 80,
 'cleaning': ['dropped duplicate unit (same unit_key): 0 rows',
              'dropped missing rent, bedrooms, bathrooms or zip code: 0 rows',
              'dropped missing square footage (a size-controlled model cannot use them): 29 rows',
              'dropped under 200 sq ft (shared-living rooms and other non-standard units): 10 '
              'rows'],
 'weights': "each building counts as 1 in total (file 'weight' column), so large buildings do not "
            'dominate',
 'predictors': ['beds',
                'baths',
                'log_sqft',
                'dist_km',
                'zip_94102',
                'zip_94105',
                'zip_94107',
                'zip_94109',
                'zip_94158',
                'in_unit_laundry',
                'shared_laundry',
                'doorman_concierge'],
 'amenities_kept': ['in_unit_laundry', 'shared_laundry', 'doorman_concierge'],
 'set_to_zero_by_lasso': ['zip_94103',
                          'zip_94111',
                          'zip_94115',
                          'zip_94130',
                          'zip_other',
                          'parking',
                          'balcony_patio',
                          'air_conditioning',
                          'business_center',
                          'car_charging_station',
                          'clubhouse',
                          'controlled_access',
                          'den',
                          'dishwasher',
                          'elevator',
                          'fireplace',
                          'furnished',
                          'gated',
                          'gym',
                          'hardwood_floors',
                          'heating',
                          'high_speed_internet',
                          'lofts',
                          'pool',
                          'roof_terrace',
                          'smoke_free',
                          'utilities_included',
                          'walk_in_closets'],
 'r2_in_sample': 0.702,
 'r2_cross_validated': 0.565,
 'validation': {'design': '10-fold cross-validation holding out whole buildings, repeated 3x; the '
                          'penalty is re-chosen (5-fold, buildings held out) inside every fold',
                'lasso': {'mae_usd': 781,
                          'mape_pct': 15.3,
                          'building_weighted_mae_usd': 898,
                          'building_weighted_mape_pct': 19.2},
                'beds_baths_sqft_distance_only': {'mae_usd': 768, 'mape_pct': 15.2},
                'plain_median': {'mae_usd': 1222, 'mape_pct': 24.9},
                'mae_improvement_over_median_pct': 36.1},
 'reliability': 'moderate: typical out-of-building error about 15% per listing (plain median: '
                '25%); cross-validated R2 on log rent 0.56',
 'caveats': ['Fit on 847 units in 80 buildings. Units in one building share their amenities and '
             'location, so the effective sample for amenity and zip effects is the number of '
             'buildings, not units.',
             'Validation holds out whole buildings, so the error quoted is for a unit in a '
             'building the model has not seen.',
             'Amenities are building-level advertised features. A building that does not list an '
             "amenity is coded 'no', but listings often omit amenities they have, which pulls "
             'amenity effects toward zero.',
             'Amenity and zip effects are associations. Buildings with more amenities tend to be '
             'newer and better located, and the lasso picks among highly overlapping amenities '
             'somewhat arbitrarily.',
             'Asking rents from available units, not signed leases; fees and concessions are not '
             'modeled.',
             'Very few 3-bed, 4-bed units in the training data: estimates for those sizes are not '
             'reliable.',
             'Amenities too rare or too common to estimate: basketball_court (4 buildings with, 76 '
             'without); office (4 buildings with, 76 without); playground (3 buildings with, 77 '
             'without); tennis_court (1 buildings with, 79 without); washer_dryer_hookup (3 '
             'buildings with, 77 without); wheelchair_access (3 buildings with, 77 without); yard '
             '(2 buildings with, 78 without).']}

PRICE_MODEL = {
    "target": "log(advertised asking monthly rent, USD)",
    "const": 4.93522,
    "coefficients": {'beds': {'coef': 0.08806, 'ci95': [-0.0036, 0.17313]},
 'baths': {'coef': -0.01598, 'ci95': [-0.12944, 0.07231]},
 'log_sqft': {'coef': 0.54161, 'ci95': [0.37056, 0.74688]},
 'dist_km': {'coef': -0.03163, 'ci95': [-0.05834, 0.00014]},
 'zip_94102': {'coef': -0.08093, 'ci95': [-0.14352, -0.0], 'selected_share': 0.733},
 'zip_94105': {'coef': 0.04029, 'ci95': [0.0, 0.10817], 'selected_share': 0.64},
 'zip_94107': {'coef': 0.00581, 'ci95': [0.0, 0.08206], 'selected_share': 0.377},
 'zip_94109': {'coef': -0.17317, 'ci95': [-0.37546, -0.0], 'selected_share': 0.787},
 'zip_94158': {'coef': 0.03656, 'ci95': [0.0, 0.21425], 'selected_share': 0.483},
 'in_unit_laundry': {'coef': 0.02201, 'ci95': [0.0, 0.09076], 'selected_share': 0.517},
 'shared_laundry': {'coef': -0.02924, 'ci95': [-0.12398, -0.0], 'selected_share': 0.473},
 'doorman_concierge': {'coef': 0.0316, 'ci95': [0.0, 0.09238], 'selected_share': 0.48}},
    "amenity_share_yes": {'in_unit_laundry': 0.613, 'shared_laundry': 0.225, 'doorman_concierge': 0.325},  # share of buildings with the amenity (kept amenities only)
    "means": {'beds': 1.14986, 'baths': 1.27637, 'log_sqft': 6.53298, 'dist_km': 2.36679},
    "smearing": 1.01844,  # corrects exp() bias when converting log-rent back to dollars
    "log_residual_quantiles_out_of_sample": {'p10': -0.2219, 'p25': -0.1265, 'p75': 0.1186, 'p90': 0.2521},
    "support": {'beds': [0.0, 4.0],
 'beds_counts': {'0': 174, '1': 375, '2': 285, '3': 11, '4': 2},
 'baths': [1.0, 2.5],
 'sqft': [221.0, 1850.0],
 'zips': ['94102',
          '94103',
          '94105',
          '94107',
          '94109',
          '94110',
          '94111',
          '94112',
          '94114',
          '94115',
          '94122',
          '94124',
          '94129',
          '94130',
          '94131',
          '94132',
          '94134',
          '94158'],
 'rent': [1795.0, 13837.0]},
    "zip_groups": ['zip_94102',
 'zip_94103',
 'zip_94105',
 'zip_94107',
 'zip_94109',
 'zip_94111',
 'zip_94115',
 'zip_94130',
 'zip_94158'],  # zips with their own effect; other training zips share "zip_other"
    "zip_distance_km": {'94102': 1.672,
 '94103': 1.549,
 '94105': 1.201,
 '94107': 1.998,
 '94109': 1.112,
 '94110': 3.037,
 '94111': 1.212,
 '94112': 7.672,
 '94114': 3.449,
 '94115': 2.392,
 '94122': 5.858,
 '94124': 7.062,
 '94129': 4.433,
 '94130': 4.643,
 '94131': 5.552,
 '94132': 11.228,
 '94134': 8.637,
 '94158': 2.128},  # average distance to Union Square, used when no lat/lon is given
}

VACANCY_MODEL = {
    "is_placeholder": True,
    "note": "Not estimated from data. Shape and numbers are illustrative only.",
    "base_days": 30.0,        # placeholder: days to lease at market rent
    "price_sensitivity": 3.0, # placeholder: how fast days grow as price exceeds market
}


def _dist_km(lat, lon):
    return math.hypot((lat - 37.788) * 111.0, (lon + 122.407) * 111.0 * math.cos(math.radians(37.8)))


def predict_rent(bedrooms=None, bathrooms=None, sqft=None, zip_code=None, amenities=None,
                 latitude=None, longitude=None):
    """Return {"estimate", "low", "high", "effects", "reliability", "warnings"}.

    bedrooms: 0 = studio. amenities: a dict like {"in_unit_laundry": True, "gym": None}
    (None = not known) or a list of amenity names the building has. Give latitude and
    longitude if you have them; otherwise distance to downtown is taken from the zip.
    low/high are the 10th/90th percentile of out-of-building errors, so about 80% of
    held-out listings fell inside that range.
    """
    pm = PRICE_MODEL
    coefs = pm["coefficients"]
    sup = pm["support"]
    warnings = []
    if bedrooms is not None:
        lo, hi = sup["beds"]
        if not lo <= bedrooms <= hi:
            warnings.append(f"bedrooms outside the {lo:g}-{hi:g} range of the training data")
        elif sup["beds_counts"].get(str(int(round(bedrooms))), 0) < 30:
            warnings.append(f"only {sup['beds_counts'].get(str(int(round(bedrooms))), 0)} training units have "
                            f"{bedrooms:g} bedrooms; the estimate is unreliable")
    if bathrooms is not None and not sup["baths"][0] <= bathrooms <= sup["baths"][1]:
        warnings.append("bathrooms are outside the range of the training data")
    if sqft is not None and not sup["sqft"][0] <= sqft <= sup["sqft"][1]:
        warnings.append("square footage is outside the range of the training data")

    z = str(zip_code)[:5] if zip_code is not None else None
    zip_name = None
    if z is not None:
        if "zip_" + z in pm["zip_groups"]:
            zip_name = "zip_" + z
        elif z in sup["zips"]:
            zip_name = "zip_other"
        else:
            warnings.append("zip code is not in the training data; no zip adjustment applied")
    if latitude is not None and longitude is not None:
        dist = _dist_km(latitude, longitude)
    elif z in pm["zip_distance_km"]:
        dist = pm["zip_distance_km"][z]
    else:
        dist = None
    inputs = {"beds": bedrooms, "baths": bathrooms, "log_sqft": math.log(sqft) if sqft else None,
              "dist_km": dist}
    if isinstance(amenities, dict):
        flags = {k: (None if v is None else bool(v)) for k, v in amenities.items()}
    else:
        flags = {k: True for k in (amenities or [])}

    log_rent = pm["const"]
    used = set()
    for name, c in coefs.items():
        if name in inputs:
            x = inputs[name]
            if x is None:
                x = pm["means"][name]
                warnings.append(f"{name} not given; using the training average")
            log_rent += c["coef"] * x
        elif name.startswith("zip_"):
            if name == zip_name:
                log_rent += c["coef"]
        else:
            used.add(name)
            v = flags.get(name)
            if v is None:
                share = pm["amenity_share_yes"][name]
                warnings.append(f"{name} not specified; assuming the typical share "
                                f"({share:.0%}) of buildings have it")
                log_rent += c["coef"] * share
            else:
                log_rent += c["coef"] * (1.0 if v else 0.0)
    if zip_name is None and z is None:
        warnings.append("no zip code given; no zip adjustment applied")
    for name in flags:
        if name not in used:
            warnings.append(f"{name} is not used by this model")

    est = math.exp(log_rent) * pm["smearing"]
    q = pm["log_residual_quantiles_out_of_sample"]
    effects = {}
    for name, c in coefs.items():
        if name == "log_sqft":
            effects["sqft_pct_per_10pct_larger"] = round((1.1 ** c["coef"] - 1) * 100, 1)
        elif name == "dist_km":
            effects["dist_pct_per_km_from_downtown"] = round((math.exp(c["coef"]) - 1) * 100, 1)
        else:
            effects[name + "_pct"] = round((math.exp(c["coef"]) - 1) * 100, 1)
    return {
        "estimate": round(est),
        "low": round(est * math.exp(q["p10"])),
        "high": round(est * math.exp(q["p90"])),
        "effects": effects,
        "reliability": MODEL_INFO["reliability"],
        "warnings": warnings,
    }


def predict_vacancy(price, market_rent):
    """PLACEHOLDER. Illustrative days-to-lease at `price` given `market_rent`
    (use predict_rent(...)["estimate"]). Not estimated from data."""
    v = VACANCY_MODEL
    return v["base_days"] * (price / market_rent) ** v["price_sensitivity"]
