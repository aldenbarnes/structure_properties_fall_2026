"""SF rent model fit on a small set of hand-reviewed comparable listings.

Trained 2026-10-14 on 23 comps (of 28 rows in the file).
Standard library only. Drop this file next to app.py and:

    import rent_model
    r = rent_model.predict_rent(bedrooms=2, bathrooms=1, zip_code="94121",
                                amenities={"in_unit_laundry": True})
    r["estimate"], r["low"], r["high"], r["warnings"]

READ THIS BEFORE USING THE NUMBERS
  The sample is small, so every coefficient has a wide confidence interval
  (PRICE_MODEL["coefficients"][name]["ci95"]) and MODEL_INFO["caveats"] lists the
  limits. Amenity effects are associations in these comps, not proven prices:
  units with an amenity also tend to be newer, larger or better located. Treat
  the estimate as a rough comp-based range, not a recommended rent.
  The vacancy curve (predict_vacancy) is still a placeholder.

The model is log(asking monthly rent) = const + sum(coef * variable), and
rent = exp(that) * smearing. For a yes/no variable, exp(coef) - 1 is the percent
rent difference. The "_unknown" terms only absorb listings whose status was not
reported; they are never used for pricing.
"""
import math

MODEL_INFO = {'trained': '2026-10-14',
 'source': 'anza_comps.csv (RealtyAPI comps; advertised asking rents, hand-reviewed)',
 'n_raw': 28,
 'n_used': 23,
 'cleaning': ['dropped marked "Exclude" in comp_use (furnished short stays, not a true match, bad '
              'dates): 5 rows',
              'dropped missing rent, bedrooms or bathrooms: 0 rows',
              'dropped duplicate unit (same listing_key and unit_key): 0 rows'],
 'predictors': ['baths', 'in_unit_laundry'],
 'r2_in_sample': 0.647,
 'leave_one_out': {'model_mae_usd': 1108,
                   'model_mape_pct': 16.1,
                   'baseline': 'median rent of the other comps',
                   'baseline_mae_usd': 1594,
                   'baseline_mape_pct': 22.6,
                   'beats_baseline': True},
 'reliability': 'low: fit on 23 comps; typical out-of-sample error about 16% (plain median of the '
                'comps: 23%)',
 'caveats': ['Only 23 comps: every coefficient has a wide confidence interval (see ci95).',
             'All comps are 2-bedroom, so the bedroom effect cannot be estimated and the model is '
             'valid for 2-bedroom units only.',
             'Amenity effects are associations. Units with an amenity also tend to be newer, '
             'larger or better located, so they overstate what adding the amenity would earn.',
             "No location effect (too few comps per zip); the estimate reflects the comp set's "
             'average location.',
             'Predictors were picked from 10 candidate models by leave-one-out error, so the '
             'reported error is optimistic.',
             'Square footage is not used: missing for 7 of 23 comps.',
             'Amenities that could not be estimated: private_outdoor (needs at least 5 "yes" and 5 '
             '"no" listings; has 9 yes / 4 no); renovated (needs at least 5 "yes" and 5 "no" '
             'listings; has 17 yes / 0 no); dishwasher (needs at least 5 "yes" and 5 "no" '
             'listings; has 19 yes / 0 no).']}

PRICE_MODEL = {
    "target": "log(advertised asking monthly rent, USD)",
    "const": 8.19411,
    "coefficients": {'baths': {'coef': 0.22366, 'se': 0.07624, 'p': 0.00335, 'ci95': [0.07422, 0.3731]},
 'in_unit_laundry': {'coef': 0.357, 'se': 0.09202, 'p': 0.0001, 'ci95': [0.17664, 0.53735]},
 'in_unit_laundry_unknown': {'coef': 0.01973,
                             'se': 0.0746,
                             'p': 0.7914,
                             'ci95': [-0.12648, 0.16594]}},
    "amenity_share_yes": {'in_unit_laundry': 0.6},  # share of comps with the amenity, among those reporting it
    "means": {'beds': 2.0, 'baths': 1.41304, 'log_sqft': 7.00083},
    "smearing": 1.01782,  # corrects exp() bias when converting log-rent back to dollars
    "log_residual_quantiles_out_of_sample": {'p10': -0.2443, 'p25': -0.1662, 'p75': 0.085, 'p90': 0.1712},
    "support": {'beds': [2.0, 2.0],
 'baths': [1.0, 3.0],
 'zips': ['94114', '94115', '94117', '94118', '94121', '94122'],
 'rent': [3700.0, 13995.0]},
}

# Log-rent effect of each amenity used for pricing (same numbers as in coefficients).
AMENITY_EFFECTS = {'in_unit_laundry': 0.357}

# Every amenity the data could code, including those too sparse to estimate.
AMENITY_SCREEN = {'in_unit_laundry': {'n_yes': 12,
                     'n_no': 8,
                     'n_unknown': 3,
                     'estimable': True,
                     'effect_pct': 42.3,
                     'ci95_pct': [18.4, 71.1],
                     'p': 0.0002},
 'parking_included': {'n_yes': 8,
                      'n_no': 8,
                      'n_unknown': 7,
                      'estimable': True,
                      'effect_pct': -3.9,
                      'ci95_pct': [-28.8, 29.8],
                      'p': 0.7974},
 'private_outdoor': {'n_yes': 9,
                     'n_no': 4,
                     'n_unknown': 10,
                     'estimable': False,
                     'effect_pct': None,
                     'ci95_pct': None,
                     'p': None,
                     'reason': 'needs at least 5 "yes" and 5 "no" listings; has 9 yes / 4 no'},
 'utilities_included': {'n_yes': 8,
                        'n_no': 7,
                        'n_unknown': 8,
                        'estimable': True,
                        'effect_pct': -4.5,
                        'ci95_pct': [-25.1, 21.8],
                        'p': 0.7108},
 'renovated': {'n_yes': 17,
               'n_no': 0,
               'n_unknown': 6,
               'estimable': False,
               'effect_pct': None,
               'ci95_pct': None,
               'p': None,
               'reason': 'needs at least 5 "yes" and 5 "no" listings; has 17 yes / 0 no'},
 'dishwasher': {'n_yes': 19,
                'n_no': 0,
                'n_unknown': 4,
                'estimable': False,
                'effect_pct': None,
                'ci95_pct': None,
                'p': None,
                'reason': 'needs at least 5 "yes" and 5 "no" listings; has 19 yes / 0 no'}}

VACANCY_MODEL = {
    "is_placeholder": True,
    "note": "Not estimated from data. Shape and numbers are illustrative only.",
    "base_days": 30.0,        # placeholder: days to lease at market rent
    "price_sensitivity": 3.0, # placeholder: how fast days grow as price exceeds market
}


def predict_rent(bedrooms=None, bathrooms=None, sqft=None, zip_code=None, amenities=None):
    """Return {"estimate", "low", "high", "effects", "reliability", "warnings"}.

    amenities: a dict like {"in_unit_laundry": True, "parking_included": False}
    (None = not known) or a list of amenity names that the unit has. low/high are
    the 10th/90th percentile of leave-one-out errors, so about 80% of comps fell
    inside that range.
    """
    pm = PRICE_MODEL
    coefs = pm["coefficients"]
    sup = pm["support"]
    warnings = []
    if bedrooms is not None and not sup["beds"][0] <= bedrooms <= sup["beds"][1]:
        lo, hi = sup["beds"]
        label = f"{lo:g}" if lo == hi else f"{lo:g}-{hi:g}"
        warnings.append(f"model was fit on {label}-bedroom comps only; "
                        f"estimate is not valid for {bedrooms:g} bedrooms")
    if bathrooms is not None and not sup["baths"][0] <= bathrooms <= sup["baths"][1]:
        warnings.append("bathrooms are outside the range of the comps")
    if zip_code is not None and str(zip_code)[:5] not in sup["zips"]:
        warnings.append("zip code is outside the area covered by the comps")
    warnings.append("location is not modeled; the estimate reflects the comps' average location")

    if sqft and "log_sqft" not in coefs:
        warnings.append("square footage is not used by this model")
    inputs = {"beds": bedrooms, "baths": bathrooms,
              "log_sqft": math.log(sqft) if sqft else None}
    if isinstance(amenities, dict):
        flags = {k: (None if v is None else bool(v)) for k, v in amenities.items()}
    else:
        flags = {k: True for k in (amenities or [])}

    log_rent = pm["const"]
    used = set()
    for name, c in coefs.items():
        if name.endswith("_unknown"):
            continue
        if name in pm["amenity_share_yes"]:
            used.add(name)
            v = flags.get(name)
            if v is None:
                share = pm["amenity_share_yes"][name]
                warnings.append(f"{name} not specified; assuming the typical share "
                                f"({share:.0%}) of comps have it")
                log_rent += c["coef"] * share
            else:
                log_rent += c["coef"] * (1.0 if v else 0.0)
        else:
            x = inputs.get(name)
            if x is None:
                x = pm["means"][name]
                warnings.append(f"{name} not given; using the comps' average")
            log_rent += c["coef"] * x
    for name in flags:
        if name not in used:
            warnings.append(f"{name} is not used by this model")

    est = math.exp(log_rent) * pm["smearing"]
    q = pm["log_residual_quantiles_out_of_sample"]
    effects = {}
    for name, c in coefs.items():
        if name.endswith("_unknown"):
            continue
        if name == "log_sqft":
            effects["sqft_pct_per_10pct_larger"] = round((1.1 ** c["coef"] - 1) * 100, 1)
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
