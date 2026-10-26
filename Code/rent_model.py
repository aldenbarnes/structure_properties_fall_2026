"""SF rent model fit on available-unit listings (lasso, penalty chosen by cross-validation).

Trained 2026-10-26 on 1228 units in 373 buildings.
Standard library only. Drop this file next to app.py and:

    import rent_model
    r = rent_model.predict_rent(bedrooms=1, bathrooms=1, sqft=700, zip_code="94107",
                                neighborhood="South of Market",
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

MODEL_INFO = {'trained': '2026-10-26',
 'source': '4 files pooled and de-duplicated: 577d54a8-realty_training.csv, '
           '201370d2-realty_training.csv, 45ad8b6e-RealtyAPI_1010_Anza_Comps.csv, '
           'san_francisco_rentals_by_neighborhood__2_.xlsx',
 'method': 'lasso (L1), penalty by grouped cross-validation',
 'penalty_fraction_of_max': 0.1215,
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
                'crime_violent',
                'crime_property',
                'zip_94104',
                'zip_94105',
                'zip_94107',
                'zip_94109',
                'zip_94110',
                'zip_94118',
                'zip_94123',
                'zip_94124',
                'zip_94131',
                'zip_94133',
                'zip_94134',
                'zip_94158',
                'shared_laundry',
                'parking',
                'air_conditioning',
                'basketball_court',
                'clubhouse',
                'controlled_access',
                'furnished',
                'gym',
                'lofts',
                'playground',
                'pool',
                'smoke_free',
                'tennis_court',
                'utilities_included',
                'washer_dryer_hookup'],
 'amenities_kept': ['shared_laundry',
                    'parking',
                    'air_conditioning',
                    'basketball_court',
                    'clubhouse',
                    'controlled_access',
                    'furnished',
                    'gym',
                    'lofts',
                    'playground',
                    'pool',
                    'smoke_free',
                    'tennis_court',
                    'utilities_included',
                    'washer_dryer_hookup'],
 'set_to_zero_by_lasso': ['zip_94102',
                          'zip_94103',
                          'zip_94108',
                          'zip_94111',
                          'zip_94114',
                          'zip_94115',
                          'zip_94116',
                          'zip_94117',
                          'zip_94121',
                          'zip_94122',
                          'zip_94130',
                          'zip_other',
                          'in_unit_laundry',
                          'balcony_patio',
                          'doorman_concierge',
                          'business_center',
                          'car_charging_station',
                          'den',
                          'dishwasher',
                          'elevator',
                          'fireplace',
                          'gated',
                          'hardwood_floors',
                          'heating',
                          'high_speed_internet',
                          'office',
                          'roof_terrace',
                          'walk_in_closets',
                          'wheelchair_access',
                          'yard'],
 'source_check': {'201370d2-realty_training.csv': {'units': 51,
                                                   'buildings': 30,
                                                   'mean_log_residual': 0.0007},
                  '45ad8b6e-RealtyAPI_1010_Anza_Comps.csv': {'units': 7,
                                                             'buildings': 7,
                                                             'mean_log_residual': 0.0906},
                  '577d54a8-realty_training.csv': {'units': 1170,
                                                   'buildings': 363,
                                                   'mean_log_residual': 0.0209}},
 'r2_in_sample': 0.792,
 'r2_in_sample_unit_weighted': 0.713,
 'r2_cross_validated': 0.661,
 'validation': {'design': '10-fold cross-validation holding out whole buildings, repeated 3x; the '
                          'penalty is re-chosen (5-fold, buildings held out) inside every fold',
                'core_variables': 'beds, baths, sqft, distance, crime',
                'cross_validated': {'lasso': {'rmse': 1385.0,
                                              'mae': 886.3,
                                              'mape': 16.2,
                                              'rmse_pct': 24.7,
                                              'rmspe': 22.5,
                                              'r2': 0.661,
                                              'rmse_log': 0.207,
                                              'r2_log': 0.661,
                                              'w_rmse': 2016.3,
                                              'w_mae': 1173.0,
                                              'w_mape': 18.8},
                                    'core_variables_only': {'rmse': 1452.5,
                                                            'mae': 962.3,
                                                            'mape': 18.0,
                                                            'rmse_pct': 25.9,
                                                            'rmspe': 25.3,
                                                            'r2': 0.628,
                                                            'rmse_log': 0.22,
                                                            'r2_log': 0.618,
                                                            'w_rmse': 2131.4,
                                                            'w_mae': 1325.1,
                                                            'w_mape': 22.1},
                                    'plain_median': {'rmse': 2412.2,
                                                     'mae': 1521.4,
                                                     'mape': 28.8,
                                                     'rmse_pct': 43.1,
                                                     'rmspe': 40.9,
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
                                 'penalty_fraction_of_max': 0.2454,
                                 'lasso': {'rmse': 1769.1,
                                           'mae': 883.2,
                                           'mape': 13.2,
                                           'rmse_pct': 29.1,
                                           'rmspe': 17.3,
                                           'r2': 0.661,
                                           'rmse_log': 0.175,
                                           'r2_log': 0.761,
                                           'w_rmse': 2510.8,
                                           'w_mae': 1249.7,
                                           'w_mape': 16.5},
                                 'core': {'rmse': 1934.0,
                                          'mae': 1057.7,
                                          'mape': 16.4,
                                          'rmse_pct': 31.8,
                                          'rmspe': 21.1,
                                          'r2': 0.594,
                                          'rmse_log': 0.2,
                                          'r2_log': 0.688,
                                          'w_rmse': 2662.8,
                                          'w_mae': 1398.3,
                                          'w_mape': 19.4},
                                 'median': {'rmse': 3156.9,
                                            'mae': 1679.7,
                                            'mape': 25.2,
                                            'rmse_pct': 51.9,
                                            'rmspe': 34.6,
                                            'r2': -0.081,
                                            'rmse_log': 0.366,
                                            'r2_log': -0.049,
                                            'w_rmse': 4494.3,
                                            'w_mae': 2459.2,
                                            'w_mape': 33.3}},
                'mae_improvement_over_median_pct': 41.7},
 'reliability': 'moderate: typical out-of-building error about 16% per listing (plain median: '
                '29%); cross-validated R2 on log rent 0.66',
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
             'Crime enters as two neighborhood-level rates (violent and property incidents per '
             '1,000 residential units). They are associations measured across a few dozen '
             'neighborhoods, not the price of crime at an address, and a building with no '
             "neighborhood input gets its zip's average. The violent and property rates move "
             'together across neighborhoods (correlation about 0.8), so their coefficients have '
             'opposite signs and only the net effect should be read '
             "(effects['crime_net_pct_if_both_rates_double']). Neighborhood crime can act as a "
             'stand-in for who lives in an area: have compliance / legal review before using it in '
             'pricing.',
             'Very few 4-bed units in the training data: estimates for those sizes are not '
             'reliable.',
             'Amenities too rare or too common to estimate: basement (4 buildings with, 347 '
             'without); rent_specials (1 buildings with, 350 without).'],
 'crime': {'features': ['crime_violent', 'crime_property'],
           'period': '2025-10-25 to 2026-10-24',
           'source': 'SFPD incident reports, 89676 geocoded-to-neighborhood incidents in the 12 '
                     'months to 2026-10-24; residential units from the assessor / '
                     'crime_neighborhood table',
           'definition': "log(1 + incidents per 1,000 residential units) in the building's "
                         'neighborhood',
           'violent_categories': ['Assault', 'Homicide', 'Rape', 'Robbery', 'Sex Offense'],
           'property_categories': ['Arson',
                                   'Burglary',
                                   'Larceny Theft',
                                   'Malicious Mischief',
                                   'Motor Vehicle Theft'],
           'buildings_with_neighborhood': 369,
           'buildings_using_zip_average': 4,
           'neighborhoods_without_rate': ['Golden Gate Park (42 units)',
                                          'Lincoln Park (58 units)',
                                          'McLaren Park (85 units)',
                                          'Presidio (31 units)',
                                          'Seacliff (946 units)',
                                          'Treasure Island (105 units)'],
           'cross_validated_without_crime': {'rmse': 1422.0,
                                             'mae': 919.7,
                                             'mape': 16.8,
                                             'rmse_pct': 25.4,
                                             'rmspe': 23.6,
                                             'r2': 0.643,
                                             'rmse_log': 0.213,
                                             'r2_log': 0.642,
                                             'w_rmse': 2051.2,
                                             'w_mae': 1201.0,
                                             'w_mape': 19.4}}}

PRICE_MODEL = {
    "target": "log(advertised asking monthly rent, USD)",
    "const": 5.68748,
    "coefficients": {'beds': {'coef': 0.20261, 'ci95': [0.13417, 0.26677]},
 'baths': {'coef': 0.00691, 'ci95': [-0.07265, 0.08114]},
 'log_sqft': {'coef': 0.42143, 'ci95': [0.27221, 0.56974]},
 'dist_km': {'coef': -0.0688, 'ci95': [-0.08731, -0.05441]},
 'crime_violent': {'coef': -0.15998, 'ci95': [-0.23347, -0.09668]},
 'crime_property': {'coef': 0.10594, 'ci95': [0.00078, 0.20685]},
 'zip_94104': {'coef': 0.09956, 'ci95': [0.0, 0.34617], 'selected_share': 0.71},
 'zip_94105': {'coef': 0.10752, 'ci95': [0.02948, 0.17444], 'selected_share': 1.0},
 'zip_94107': {'coef': -0.02503, 'ci95': [-0.10263, 0.0], 'selected_share': 0.66},
 'zip_94109': {'coef': -0.18521, 'ci95': [-0.30824, -0.06157], 'selected_share': 0.997},
 'zip_94110': {'coef': 0.00444, 'ci95': [-0.01616, 0.10258], 'selected_share': 0.56},
 'zip_94118': {'coef': -0.02458, 'ci95': [-0.09072, 0.0], 'selected_share': 0.427},
 'zip_94123': {'coef': 0.07069, 'ci95': [-0.03248, 0.25226], 'selected_share': 0.723},
 'zip_94124': {'coef': -0.07482, 'ci95': [-0.22324, -0.0], 'selected_share': 0.657},
 'zip_94131': {'coef': -0.09863, 'ci95': [-0.35953, -0.0], 'selected_share': 0.723},
 'zip_94133': {'coef': -0.23332, 'ci95': [-0.45078, 0.0], 'selected_share': 0.917},
 'zip_94134': {'coef': -0.09946, 'ci95': [-0.22196, -0.0], 'selected_share': 0.81},
 'zip_94158': {'coef': 0.16086, 'ci95': [0.08361, 0.22877], 'selected_share': 1.0},
 'shared_laundry': {'coef': -0.09656, 'ci95': [-0.16879, -0.00034], 'selected_share': 0.973},
 'parking': {'coef': 0.00137, 'ci95': [0.0, 0.04929], 'selected_share': 0.467},
 'air_conditioning': {'coef': 0.05436, 'ci95': [0.00331, 0.10556], 'selected_share': 0.983},
 'basketball_court': {'coef': 0.0001, 'ci95': [-0.04702, 0.14636], 'selected_share': 0.543},
 'clubhouse': {'coef': -0.01566, 'ci95': [-0.06964, -0.0], 'selected_share': 0.52},
 'controlled_access': {'coef': -0.00184, 'ci95': [-0.06381, -0.0], 'selected_share': 0.48},
 'furnished': {'coef': 0.09909, 'ci95': [0.01855, 0.14234], 'selected_share': 0.997},
 'gym': {'coef': 0.03787, 'ci95': [0.0, 0.10312], 'selected_share': 0.84},
 'lofts': {'coef': -0.08548, 'ci95': [-0.18845, -0.0], 'selected_share': 0.77},
 'playground': {'coef': -0.04551, 'ci95': [-0.14495, -0.0], 'selected_share': 0.737},
 'pool': {'coef': 0.03298, 'ci95': [0.0, 0.09125], 'selected_share': 0.78},
 'smoke_free': {'coef': -0.00234, 'ci95': [-0.05293, -0.0], 'selected_share': 0.37},
 'tennis_court': {'coef': 0.07455, 'ci95': [0.0, 0.18689], 'selected_share': 0.72},
 'utilities_included': {'coef': -0.05521, 'ci95': [-0.1471, -0.0], 'selected_share': 0.77},
 'washer_dryer_hookup': {'coef': -0.125, 'ci95': [-0.39343, -0.0], 'selected_share': 0.837}},
    "amenity_share_yes": {'shared_laundry': 0.117,
 'parking': 0.581,
 'air_conditioning': 0.413,
 'basketball_court': 0.02,
 'clubhouse': 0.068,
 'controlled_access': 0.128,
 'furnished': 0.444,
 'gym': 0.484,
 'lofts': 0.017,
 'playground': 0.048,
 'pool': 0.16,
 'smoke_free': 0.311,
 'tennis_court': 0.017,
 'utilities_included': 0.071,
 'washer_dryer_hookup': 0.02},  # share of buildings with the amenity (kept amenities only)
    "means": {'beds': 1.41364,
 'baths': 1.40289,
 'log_sqft': 6.7216,
 'dist_km': 2.55977,
 'crime_violent': 3.55685,
 'crime_property': 4.77898},
    "smearing": 1.02209,  # corrects exp() bias when converting log-rent back to dollars
    "log_residual_quantiles_out_of_sample": {'p10': -0.2451, 'p25': -0.1259, 'p75': 0.1213, 'p90': 0.2555},
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
    # Neighborhood crime rates (None when the model was trained without crime data). Neighborhood names are the
    # keys of ["neighborhoods"]; use them for the app's dropdown.
    "crime": {'features': ['crime_violent', 'crime_property'],
 'period': '2025-10-25 to 2026-10-24',
 'source': 'SFPD incident reports, 89676 geocoded-to-neighborhood incidents in the 12 months to '
           '2026-10-24; residential units from the assessor / crime_neighborhood table',
 'neighborhoods': {'Bayview Hunters Point': {'violent_per_1000_units': 64.4,
                                             'property_per_1000_units': 181.1},
                   'Bernal Heights': {'violent_per_1000_units': 21.7,
                                      'property_per_1000_units': 68.2},
                   'Castro/Upper Market': {'violent_per_1000_units': 23.0,
                                           'property_per_1000_units': 97.1},
                   'Chinatown': {'violent_per_1000_units': 42.2, 'property_per_1000_units': 136.5},
                   'Excelsior': {'violent_per_1000_units': 20.7, 'property_per_1000_units': 50.0},
                   'Financial District/South Beach': {'violent_per_1000_units': 40.4,
                                                      'property_per_1000_units': 142.5},
                   'Glen Park': {'violent_per_1000_units': 6.4, 'property_per_1000_units': 39.6},
                   'Haight Ashbury': {'violent_per_1000_units': 16.4,
                                      'property_per_1000_units': 60.3},
                   'Hayes Valley': {'violent_per_1000_units': 18.6,
                                    'property_per_1000_units': 97.5},
                   'Inner Richmond': {'violent_per_1000_units': 7.7,
                                      'property_per_1000_units': 36.1},
                   'Inner Sunset': {'violent_per_1000_units': 4.6, 'property_per_1000_units': 28.9},
                   'Japantown': {'violent_per_1000_units': 14.5, 'property_per_1000_units': 79.2},
                   'Lakeshore': {'violent_per_1000_units': 27.3, 'property_per_1000_units': 183.5},
                   'Lone Mountain/USF': {'violent_per_1000_units': 11.5,
                                         'property_per_1000_units': 72.9},
                   'Marina': {'violent_per_1000_units': 10.5, 'property_per_1000_units': 67.9},
                   'Mission': {'violent_per_1000_units': 64.0, 'property_per_1000_units': 136.4},
                   'Mission Bay': {'violent_per_1000_units': 43.2,
                                   'property_per_1000_units': 121.2},
                   'Nob Hill': {'violent_per_1000_units': 18.3, 'property_per_1000_units': 57.6},
                   'Noe Valley': {'violent_per_1000_units': 3.5, 'property_per_1000_units': 60.6},
                   'North Beach': {'violent_per_1000_units': 32.4, 'property_per_1000_units': 75.4},
                   'Oceanview/Merced/Ingleside': {'violent_per_1000_units': 8.8,
                                                  'property_per_1000_units': 44.0},
                   'Outer Mission': {'violent_per_1000_units': 29.4,
                                     'property_per_1000_units': 56.9},
                   'Outer Richmond': {'violent_per_1000_units': 7.1,
                                      'property_per_1000_units': 35.2},
                   'Pacific Heights': {'violent_per_1000_units': 5.1,
                                       'property_per_1000_units': 66.2},
                   'Portola': {'violent_per_1000_units': 26.1, 'property_per_1000_units': 89.5},
                   'Potrero Hill': {'violent_per_1000_units': 7.8, 'property_per_1000_units': 97.7},
                   'Presidio Heights': {'violent_per_1000_units': 5.2,
                                        'property_per_1000_units': 55.2},
                   'Russian Hill': {'violent_per_1000_units': 7.3, 'property_per_1000_units': 55.8},
                   'South of Market': {'violent_per_1000_units': 114.5,
                                       'property_per_1000_units': 313.8},
                   'Sunset/Parkside': {'violent_per_1000_units': 6.2,
                                       'property_per_1000_units': 31.4},
                   'Tenderloin': {'violent_per_1000_units': 94.3, 'property_per_1000_units': 162.6},
                   'Twin Peaks': {'violent_per_1000_units': 3.1, 'property_per_1000_units': 35.7},
                   'Visitacion Valley': {'violent_per_1000_units': 33.2,
                                         'property_per_1000_units': 55.3},
                   'West of Twin Peaks': {'violent_per_1000_units': 6.2,
                                          'property_per_1000_units': 47.0},
                   'Western Addition': {'violent_per_1000_units': 34.4,
                                        'property_per_1000_units': 102.9}},
 'zip_default': {'94102': {'crime_violent': 4.2526, 'crime_property': 5.0},
                 '94103': {'crime_violent': 4.5456, 'crime_property': 5.4894},
                 '94104': {'crime_violent': 3.7242, 'crime_property': 4.9662},
                 '94105': {'crime_violent': 3.7242, 'crime_property': 4.9662},
                 '94107': {'crime_violent': 3.5974, 'crime_property': 4.9917},
                 '94108': {'crime_violent': 2.962, 'crime_property': 4.0712},
                 '94109': {'crime_violent': 3.1035, 'crime_property': 4.3125},
                 '94110': {'crime_violent': 3.9561, 'crime_property': 4.8356},
                 '94111': {'crime_violent': 3.5635, 'crime_property': 4.4933},
                 '94112': {'crime_violent': 3.415, 'crime_property': 4.0589},
                 '94114': {'crime_violent': 3.1768, 'crime_property': 4.5864},
                 '94115': {'crime_violent': 2.3742, 'crime_property': 4.3287},
                 '94116': {'crime_violent': 1.9746, 'crime_property': 3.4782},
                 '94117': {'crime_violent': 2.788, 'crime_property': 4.2215},
                 '94118': {'crime_violent': 2.0284, 'crime_property': 3.7442},
                 '94121': {'crime_violent': 2.0955, 'crime_property': 3.5893},
                 '94122': {'crime_violent': 1.9041, 'crime_property': 3.4552},
                 '94123': {'crime_violent': 2.4388, 'crime_property': 4.2333},
                 '94124': {'crime_violent': 4.1811, 'crime_property': 5.2047},
                 '94127': {'crime_violent': 1.9708, 'crime_property': 3.8721},
                 '94131': {'crime_violent': 1.5415, 'crime_property': 3.9594},
                 '94132': {'crime_violent': 3.3426, 'crime_property': 5.2178},
                 '94133': {'crime_violent': 2.9542, 'crime_property': 4.2173},
                 '94134': {'crime_violent': 3.4249, 'crime_property': 4.276},
                 '94158': {'crime_violent': 3.7883, 'crime_property': 4.8058}},
 'overall': {'crime_violent': 3.5569, 'crime_property': 4.779}},
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
                 latitude=None, longitude=None, neighborhood=None):
    """Return {"estimate", "low", "high", "effects", "reliability", "warnings"}.

    bedrooms: 0 = studio. amenities: a dict like {"in_unit_laundry": True, "gym": None}
    (None = not known) or a list of amenity names the building has. Give latitude and
    longitude if you have them; otherwise distance to downtown is taken from the zip.
    neighborhood: an SF analysis neighborhood name (see PRICE_MODEL["crime"]["neighborhoods"]); it sets the
    violent / property crime rates. Without it the zip's average rate is used, which carries much less of
    the crime effect.
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
    crime = pm.get("crime")
    hood_used = None
    if crime:
        names = {h.lower(): h for h in crime["neighborhoods"]}
        if neighborhood is not None:
            hood_used = names.get(str(neighborhood).strip().lower())
            if hood_used is None:
                warnings.append(f"neighborhood '{neighborhood}' has no crime rate in this model; using the zip average")
        if hood_used is not None:
            r = crime["neighborhoods"][hood_used]
            inputs["crime_violent"] = math.log1p(r["violent_per_1000_units"])
            inputs["crime_property"] = math.log1p(r["property_per_1000_units"])
        else:
            if neighborhood is None:
                warnings.append("no neighborhood given; using the average crime rate of training buildings in this zip")
            inputs.update(crime["zip_default"].get(z, crime["overall"]))
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
        elif name in ("crime_violent", "crime_property"):
            effects[name + "_pct_if_rate_doubles"] = round((2 ** c["coef"] - 1) * 100, 1)
        else:
            effects[name + "_pct"] = round((math.exp(c["coef"]) - 1) * 100, 1)
    if "crime_violent" in coefs and "crime_property" in coefs:
        # the two rates move together across neighborhoods (correlation about 0.8), so only their net effect is meaningful
        effects["crime_net_pct_if_both_rates_double"] = round(
            (2 ** (coefs["crime_violent"]["coef"] + coefs["crime_property"]["coef"]) - 1) * 100, 1)
    return {
        "estimate": round(est),
        "low": round(est * math.exp(q["p10"])),
        "high": round(est * math.exp(q["p90"])),
        "effects": effects,
        "neighborhood_used": hood_used,
        "reliability": MODEL_INFO["reliability"],
        "warnings": warnings,
    }


def predict_vacancy(price, market_rent):
    """PLACEHOLDER. Illustrative days-to-lease at `price` given `market_rent`
    (use predict_rent(...)["estimate"]). Not estimated from data."""
    v = VACANCY_MODEL
    return v["base_days"] * (price / market_rent) ** v["price_sensitivity"]
