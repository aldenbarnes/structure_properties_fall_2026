"""SF rent model fit on available-unit listings (lasso, penalty chosen by cross-validation).

Trained 2026-10-21 on 1228 units in 373 buildings.
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

MODEL_INFO = {'trained': '2026-10-21',
 'source': '4 files pooled and de-duplicated: 577d54a8-realty_training.csv, '
           '201370d2-realty_training.csv, 45ad8b6e-RealtyAPI_1010_Anza_Comps.csv, '
           'san_francisco_rentals_by_neighborhood__2_.xlsx',
 'method': 'lasso (L1), penalty by grouped cross-validation',
 'penalty_fraction_of_max': 0.1366,
 'n_raw': 1505,
 'n_used': 1228,
 'n_buildings': 373,
 'files': [{'file': '577d54a8-realty_training.csv',
            'kind': 'units',
            'rows_read': 1324,
            'rows_after_dedup': 1324},
           {'file': '201370d2-realty_training.csv',
            'kind': 'units',
            'rows_read': 886,
            'rows_after_dedup': 56},
           {'file': '45ad8b6e-RealtyAPI_1010_Anza_Comps.csv',
            'kind': 'comps',
            'rows_read': 28,
            'rows_after_dedup': 11},
           {'file': 'san_francisco_rentals_by_neighborhood__2_.xlsx',
            'kind': 'listings',
            'rows_read': 500,
            'rows_after_dedup': 114}],
 'cleaning': ['dropped 45ad8b6e-RealtyAPI_1010_Anza_Comps.csv: rows excluded or repeated inside '
              "the file (comp_use 'Exclude' or an identical repeated listing): 5 rows",
              'dropped san_francisco_rentals_by_neighborhood__2_.xlsx: listings at a building '
              'already covered by the unit-level files (same street number, street and zip, so '
              'counting them again would double count): 112 rows',
              'dropped san_francisco_rentals_by_neighborhood__2_.xlsx: listings with the same zip, '
              'beds, baths, sqft and rent as a unit already in the unit-level files (probable '
              'repeat under a different address): 274 rows',
              'dropped 201370d2-realty_training.csv: the same unit (listing_key + unit_key) '
              'already in an earlier file: 745 rows',
              'dropped 45ad8b6e-RealtyAPI_1010_Anza_Comps.csv: the same unit (listing_key + '
              'unit_key) already in an earlier file: 12 rows',
              'dropped 201370d2-realty_training.csv: the same floor plan (building + plan + beds + '
              'baths + sqft) already in an earlier file: 85 rows',
              'dropped duplicate unit (same unit_key): 0 rows',
              'dropped missing rent, bedrooms, bathrooms or zip code: 17 rows',
              'dropped outside San Francisco (zip code not 941xx): 4 rows',
              'dropped houses and townhouses (single-family market, not multifamily apartments): '
              '78 rows',
              'dropped more than 4 bedrooms (too few to model; includes room-by-room listings): 4 '
              'rows',
              'dropped missing square footage (a size-controlled model cannot use them): 155 rows',
              'dropped under 200 sq ft (shared-living rooms and other non-standard units): 19 '
              'rows'],
 'weights': 'every building counts as 1 in total, so large buildings do not dominate the fit',
 'predictors': ['beds',
                'baths',
                'log_sqft',
                'dist_km',
                'zip_94102',
                'zip_94103',
                'zip_94104',
                'zip_94105',
                'zip_94108',
                'zip_94109',
                'zip_94115',
                'zip_94117',
                'zip_94123',
                'zip_94124',
                'zip_94133',
                'zip_94134',
                'zip_94158',
                'shared_laundry',
                'parking',
                'air_conditioning',
                'clubhouse',
                'dishwasher',
                'fireplace',
                'furnished',
                'gym',
                'lofts',
                'playground',
                'pool',
                'tennis_court',
                'utilities_included',
                'washer_dryer_hookup'],
 'amenities_kept': ['shared_laundry',
                    'parking',
                    'air_conditioning',
                    'clubhouse',
                    'dishwasher',
                    'fireplace',
                    'furnished',
                    'gym',
                    'lofts',
                    'playground',
                    'pool',
                    'tennis_court',
                    'utilities_included',
                    'washer_dryer_hookup'],
 'set_to_zero_by_lasso': ['zip_94107',
                          'zip_94110',
                          'zip_94111',
                          'zip_94114',
                          'zip_94116',
                          'zip_94118',
                          'zip_94121',
                          'zip_94122',
                          'zip_94130',
                          'zip_94131',
                          'zip_other',
                          'in_unit_laundry',
                          'balcony_patio',
                          'doorman_concierge',
                          'basketball_court',
                          'business_center',
                          'car_charging_station',
                          'controlled_access',
                          'den',
                          'elevator',
                          'gated',
                          'hardwood_floors',
                          'heating',
                          'high_speed_internet',
                          'office',
                          'roof_terrace',
                          'smoke_free',
                          'walk_in_closets',
                          'wheelchair_access',
                          'yard'],
 'source_check': {'201370d2-realty_training.csv': {'units': 51,
                                                   'buildings': 30,
                                                   'mean_log_residual': -0.0234},
                  '45ad8b6e-RealtyAPI_1010_Anza_Comps.csv': {'units': 7,
                                                             'buildings': 7,
                                                             'mean_log_residual': 0.0932},
                  '577d54a8-realty_training.csv': {'units': 1170,
                                                   'buildings': 363,
                                                   'mean_log_residual': 0.0134}},
 'r2_in_sample': 0.776,
 'r2_in_sample_unit_weighted': 0.698,
 'r2_cross_validated': 0.642,
 'validation': {'design': '10-fold cross-validation holding out whole buildings, repeated 3x; the '
                          'penalty is re-chosen (5-fold, buildings held out) inside every fold',
                'cross_validated': {'lasso': {'rmse': 1422.0,
                                              'mae': 919.7,
                                              'mape': 16.8,
                                              'r2': 0.643,
                                              'rmse_log': 0.213,
                                              'r2_log': 0.642,
                                              'w_rmse': 2051.2,
                                              'w_mae': 1201.0,
                                              'w_mape': 19.4},
                                    'beds_baths_sqft_distance_only': {'rmse': 1504.7,
                                                                      'mae': 1022.9,
                                                                      'mape': 19.4,
                                                                      'r2': 0.6,
                                                                      'rmse_log': 0.231,
                                                                      'r2_log': 0.578,
                                                                      'w_rmse': 2154.8,
                                                                      'w_mae': 1341.5,
                                                                      'w_mape': 22.6},
                                    'plain_median': {'rmse': 2412.2,
                                                     'mae': 1521.4,
                                                     'mape': 28.8,
                                                     'r2': -0.027,
                                                     'rmse_log': 0.358,
                                                     'r2_log': -0.017,
                                                     'w_rmse': 3710.3,
                                                     'w_mae': 2215.6,
                                                     'w_mape': 34.9}},
                'holdout_test': {'design': '20% of buildings (random, seed 42) set aside; model '
                                           'and penalty fit on the other buildings only; test '
                                           'buildings scored once',
                                 'n_train_units': 1059,
                                 'n_test_units': 169,
                                 'n_train_buildings': 298,
                                 'n_test_buildings': 75,
                                 'penalty_fraction_of_max': 0.1536,
                                 'lasso': {'rmse': 1833.8,
                                           'mae': 983.6,
                                           'mape': 14.8,
                                           'r2': 0.635,
                                           'rmse_log': 0.194,
                                           'r2_log': 0.706,
                                           'w_rmse': 2552.8,
                                           'w_mae': 1303.9,
                                           'w_mape': 17.2},
                                 'core': {'rmse': 2044.3,
                                          'mae': 1205.1,
                                          'mape': 18.8,
                                          'r2': 0.547,
                                          'rmse_log': 0.224,
                                          'r2_log': 0.608,
                                          'w_rmse': 2739.6,
                                          'w_mae': 1469.3,
                                          'w_mape': 20.5},
                                 'median': {'rmse': 3156.9,
                                            'mae': 1679.7,
                                            'mape': 25.2,
                                            'r2': -0.081,
                                            'rmse_log': 0.366,
                                            'r2_log': -0.049,
                                            'w_rmse': 4494.3,
                                            'w_mae': 2459.2,
                                            'w_mape': 33.3}},
                'mae_improvement_over_median_pct': 39.5},
 'reliability': 'moderate: typical out-of-building error about 17% per listing (plain median: '
                '29%); cross-validated R2 on log rent 0.64',
 'caveats': ['Fit on 1228 units in 373 buildings. Units in one building share their amenities and '
             'location, so the effective sample for amenity and zip effects is the number of '
             'buildings, not units.',
             'Validation holds out whole buildings, so the error quoted is for a unit in a '
             'building the model has not seen.',
             'Amenities are building-level advertised features. A building that does not list an '
             "amenity is coded 'no', but listings often omit amenities they have, which pulls "
             'amenity effects toward zero. 24 units had no amenity list at all and are treated as '
             "a typical building, not as 'no'.",
             'Amenity and zip effects are associations. Buildings with more amenities tend to be '
             'newer and better located, and the lasso picks among highly overlapping amenities '
             'somewhat arbitrarily.',
             'Asking rents from available units, not signed leases; fees and concessions are not '
             'modeled. Scope: apartments and condos with up to 4 bedrooms inside San Francisco.',
             'Very few 4-bed units in the training data: estimates for those sizes are not '
             'reliable.',
             'Amenities too rare or too common to estimate: basement (4 buildings with, 347 '
             'without); rent_specials (1 buildings with, 350 without).']}

PRICE_MODEL = {
    "target": "log(advertised asking monthly rent, USD)",
    "const": 5.33223,
    "coefficients": {'beds': {'coef': 0.19358, 'ci95': [0.1226, 0.25775]},
 'baths': {'coef': -0.00398, 'ci95': [-0.08528, 0.07577]},
 'log_sqft': {'coef': 0.45637, 'ci95': [0.28704, 0.60401]},
 'dist_km': {'coef': -0.04478, 'ci95': [-0.06189, -0.03177]},
 'zip_94102': {'coef': -0.00035, 'ci95': [-0.10011, 0.0], 'selected_share': 0.463},
 'zip_94103': {'coef': -0.01197, 'ci95': [-0.07962, -0.0], 'selected_share': 0.67},
 'zip_94104': {'coef': 0.14828, 'ci95': [0.0, 0.40637], 'selected_share': 0.81},
 'zip_94105': {'coef': 0.15515, 'ci95': [0.06681, 0.21827], 'selected_share': 1.0},
 'zip_94108': {'coef': 0.064, 'ci95': [0.0, 0.17994], 'selected_share': 0.683},
 'zip_94109': {'coef': -0.10801, 'ci95': [-0.23605, 0.0], 'selected_share': 0.97},
 'zip_94115': {'coef': 0.11672, 'ci95': [0.0, 0.20103], 'selected_share': 0.963},
 'zip_94117': {'coef': 0.02006, 'ci95': [0.0, 0.12064], 'selected_share': 0.467},
 'zip_94123': {'coef': 0.18602, 'ci95': [0.0, 0.37151], 'selected_share': 0.933},
 'zip_94124': {'coef': -0.20837, 'ci95': [-0.37284, 0.0], 'selected_share': 0.913},
 'zip_94133': {'coef': -0.13878, 'ci95': [-0.40204, 0.0], 'selected_share': 0.843},
 'zip_94134': {'coef': -0.23734, 'ci95': [-0.36698, -0.05672], 'selected_share': 0.993},
 'zip_94158': {'coef': 0.16792, 'ci95': [0.07529, 0.21977], 'selected_share': 1.0},
 'shared_laundry': {'coef': -0.07723, 'ci95': [-0.17085, 0.0], 'selected_share': 0.913},
 'parking': {'coef': 0.0003, 'ci95': [0.0, 0.05522], 'selected_share': 0.483},
 'air_conditioning': {'coef': 0.06196, 'ci95': [0.00115, 0.11091], 'selected_share': 0.98},
 'clubhouse': {'coef': -0.00911, 'ci95': [-0.07007, -0.0], 'selected_share': 0.43},
 'dishwasher': {'coef': 0.00336, 'ci95': [0.0, 0.06177], 'selected_share': 0.47},
 'fireplace': {'coef': 0.01004, 'ci95': [0.0, 0.09444], 'selected_share': 0.553},
 'furnished': {'coef': 0.08921, 'ci95': [0.0, 0.12966], 'selected_share': 0.967},
 'gym': {'coef': 0.03219, 'ci95': [0.0, 0.09208], 'selected_share': 0.823},
 'lofts': {'coef': -0.07844, 'ci95': [-0.16557, -0.0], 'selected_share': 0.767},
 'playground': {'coef': -0.02423, 'ci95': [-0.11551, -0.0], 'selected_share': 0.623},
 'pool': {'coef': 0.00752, 'ci95': [0.0, 0.07228], 'selected_share': 0.567},
 'tennis_court': {'coef': 0.09654, 'ci95': [0.0, 0.15307], 'selected_share': 0.837},
 'utilities_included': {'coef': -0.04757, 'ci95': [-0.13924, -0.0], 'selected_share': 0.747},
 'washer_dryer_hookup': {'coef': -0.12176, 'ci95': [-0.3929, -0.0], 'selected_share': 0.803}},
    "amenity_share_yes": {'shared_laundry': 0.117,
 'parking': 0.581,
 'air_conditioning': 0.413,
 'clubhouse': 0.068,
 'dishwasher': 0.593,
 'fireplace': 0.085,
 'furnished': 0.444,
 'gym': 0.484,
 'lofts': 0.017,
 'playground': 0.048,
 'pool': 0.16,
 'tennis_court': 0.017,
 'utilities_included': 0.071,
 'washer_dryer_hookup': 0.02},  # share of buildings with the amenity (kept amenities only)
    "means": {'beds': 1.41364, 'baths': 1.40289, 'log_sqft': 6.7216, 'dist_km': 2.55977},
    "smearing": 1.0236,  # corrects exp() bias when converting log-rent back to dollars
    "log_residual_quantiles_out_of_sample": {'p10': -0.2454, 'p25': -0.1411, 'p75': 0.1182, 'p90': 0.2561},
    "support": {'beds': [0.0, 4.0],
 'beds_counts': {'0': 230, '1': 529, '2': 429, '3': 32, '4': 8},
 'baths': [1.0, 3.0],
 'sqft': [200.0, 6000.0],
 'zips': ['94102',
          '94103',
          '94104',
          '94105',
          '94107',
          '94108',
          '94109',
          '94110',
          '94111',
          '94112',
          '94114',
          '94115',
          '94116',
          '94117',
          '94118',
          '94121',
          '94122',
          '94123',
          '94124',
          '94127',
          '94129',
          '94130',
          '94131',
          '94132',
          '94133',
          '94134',
          '94158'],
 'rent': [1400.0, 30000.0]},
    "zip_groups": ['zip_94102',
 'zip_94103',
 'zip_94104',
 'zip_94105',
 'zip_94107',
 'zip_94108',
 'zip_94109',
 'zip_94110',
 'zip_94111',
 'zip_94114',
 'zip_94115',
 'zip_94116',
 'zip_94117',
 'zip_94118',
 'zip_94121',
 'zip_94122',
 'zip_94123',
 'zip_94124',
 'zip_94130',
 'zip_94131',
 'zip_94133',
 'zip_94134',
 'zip_94158'],  # zips with their own effect; other training zips share "zip_other"
    "zip_distance_km": {'94102': 1.671,
 '94103': 1.582,
 '94104': 0.391,
 '94105': 1.202,
 '94107': 1.966,
 '94108': 0.537,
 '94109': 1.134,
 '94110': 3.442,
 '94111': 1.222,
 '94112': 7.838,
 '94114': 3.559,
 '94115': 2.478,
 '94116': 9.568,
 '94117': 4.09,
 '94118': 5.073,
 '94121': 7.418,
 '94122': 6.716,
 '94123': 3.097,
 '94124': 7.085,
 '94127': 7.427,
 '94129': 4.433,
 '94130': 4.642,
 '94131': 5.44,
 '94132': 11.158,
 '94133': 1.845,
 '94134': 8.027,
 '94158': 2.121},  # average distance to Union Square, used when no lat/lon is given
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
