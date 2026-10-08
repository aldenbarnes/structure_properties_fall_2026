"""SF rent model: regression coefficients and prediction functions in ONE file.

Trained 2026-10-08 on 352 cleaned listings (of 500 raw).
Standard library only. Drop this file next to app.py and:

    import rent_model
    result = rent_model.predict_rent(bedrooms=1, bathrooms=1, sqft=650, zip_code="94110")
    result["estimate"], result["low"], result["high"]

WHAT IS REAL AND WHAT IS A PLACEHOLDER
  Real (estimated from data): bedrooms, bathrooms, square footage and zip-code
    effects on asking rent (PRICE_MODEL below).
  Placeholder (NOT estimated): amenity effects (AMENITY_EFFECTS is empty) and the
    vacancy curve (predict_vacancy). Both exist so the app has something to call
    until the RealtyAPI amenity / time-on-market data is available.

The model is log(asking monthly rent) = const + zip effect + b1*bedrooms
  + b2*bathrooms + b3*log(sqft); rent = exp(that) * smearing.
A coefficient c on bedrooms/bathrooms means about (exp(c)-1)*100 % more rent per
extra bedroom/bathroom; the log_sqft coefficient is an elasticity (a 10% larger
unit rents for about 10*c % more).
"""
import math

MODEL_INFO = {'trained': '2026-10-08',
 'source': 'san_francisco_rentals_by_neighborhood.xlsx (RealtyAPI/Apartments.com-style active '
           'listings, snapshot ~2026-10-07; asking rents, not signed leases)',
 'n_raw': 500,
 'n_used': 352,
 'cleaning': ['dropped missing bedrooms, bathrooms or square footage: 135 rows',
              'dropped bedrooms above 5 (data errors / rooming houses): 3 rows',
              'dropped square footage outside 150-4,000: 5 rows',
              'dropped rent under $1,000 (SRO / shared): 0 rows',
              'dropped rent per sq ft outside $2-$12 (typos / hotel rooms): 5 rows'],
 'r2': 0.783,
 'cv': {'method': '10-fold cross-validation repeated 10x',
        'model_mae_usd': 1169,
        'model_mape_pct': 21.5,
        'baseline': 'median rent by zip x bedrooms',
        'baseline_mae_usd': 1446,
        'baseline_mape_pct': 25.5}}

PRICE_MODEL = {
    "target": "log(asking monthly rent, USD)",
    "const": 4.69953,                 # intercept for the reference zip (94109)
    "city_intercept": 4.67409,  # intercept averaged over zips; used when zip is unknown
    "coefficients": {'bedrooms': {'coef': 0.11961, 'se': 0.02863, 'p': 3e-05, 'ci95': [0.06349, 0.17573]},
 'bathrooms': {'coef': 0.05113, 'se': 0.04134, 'p': 0.2162, 'ci95': [-0.0299, 0.13216]},
 'log_sqft': {'coef': 0.54145, 'se': 0.05739, 'p': 0.0, 'ci95': [0.42898, 0.65393]}},
    "zip_effects": {'94109': 0.0,
 '94102': -0.14219,
 '94103': -0.08778,
 '94105': 0.27629,
 '94107': -0.01454,
 '94110': -0.01737,
 '94115': 0.04429,
 '94117': 0.09154,
 '94122': -0.17876,
 '94123': 0.20705,
 '94131': -0.08749,
 'other': -0.10446},
    "zip_listing_counts": {'other': 83,
 '94109': 72,
 '94102': 36,
 '94103': 28,
 '94107': 22,
 '94110': 20,
 '94123': 20,
 '94105': 18,
 '94117': 16,
 '94115': 13,
 '94122': 12,
 '94131': 12},
    "smearing": 1.03116,           # corrects exp() bias when converting log-rent back to dollars
    "log_residual_quantiles_out_of_sample": {'p10': -0.3549, 'p25': -0.1706, 'p75': 0.1713, 'p90': 0.308},
    "train_range": {'bedrooms': [0, 5], 'bathrooms': [1.0, 4.5], 'sqft': [150.0, 3708.0]},
}

# Placeholder: filled in once amenity data exists. Format: {"washer_dryer": 0.04, ...}
# (each value is the log-rent effect; predict_rent adds it when the amenity is passed in).
AMENITY_EFFECTS = {}

VACANCY_MODEL = {
    "is_placeholder": True,
    "note": "Not estimated from data. Shape and numbers are illustrative only.",
    "base_days": 30.0,        # placeholder: days to lease at market rent
    "price_sensitivity": 3.0, # placeholder: how fast days grow as price exceeds market
}


def _zip_effect(zip_code):
    if zip_code is None:
        return None
    z = str(zip_code).strip()[:5]
    effects = PRICE_MODEL["zip_effects"]
    return effects[z] if z in effects else effects["other"]


def predict_rent(bedrooms, bathrooms, sqft, zip_code=None, amenities=None):
    """Return {"estimate", "low", "high", "effects", "warnings"} for monthly asking rent.

    low/high are the 10th/90th percentile range seen in out-of-sample testing, so
    about 80% of comparable listings landed inside it. zip_code=None uses a
    citywide average location effect. amenities (list of names) only matter once
    AMENITY_EFFECTS is populated; unknown names are ignored.
    """
    pm = PRICE_MODEL
    c = pm["coefficients"]
    tr = pm["train_range"]
    warnings = []
    if not tr["bedrooms"][0] <= bedrooms <= tr["bedrooms"][1]:
        warnings.append("bedrooms outside the range the model was fit on")
    if not tr["sqft"][0] <= sqft <= tr["sqft"][1]:
        warnings.append("square footage outside the range the model was fit on")
    if not tr["bathrooms"][0] <= bathrooms <= tr["bathrooms"][1]:
        warnings.append("bathrooms outside the range the model was fit on")
    ze = _zip_effect(zip_code)
    if zip_code is None:
        warnings.append("no zip code given; using a citywide average location")
        base = pm["city_intercept"]
    else:
        base = pm["const"] + ze
    log_rent = (base + c["bedrooms"]["coef"] * bedrooms
                + c["bathrooms"]["coef"] * bathrooms
                + c["log_sqft"]["coef"] * math.log(sqft))
    am = sum(AMENITY_EFFECTS.get(a, 0.0) for a in (amenities or []))
    log_rent += am
    est = math.exp(log_rent) * pm["smearing"]
    q = pm["log_residual_quantiles_out_of_sample"]
    return {
        "estimate": round(est),
        "low": round(est * math.exp(q["p10"])),
        "high": round(est * math.exp(q["p90"])),
        "effects": {
            "bedrooms_pct_per_extra": round((math.exp(c["bedrooms"]["coef"]) - 1) * 100, 1),
            "bathrooms_pct_per_extra": round((math.exp(c["bathrooms"]["coef"]) - 1) * 100, 1),
            "sqft_pct_per_10pct_larger": round((1.1 ** c["log_sqft"]["coef"] - 1) * 100, 1),
        },
        "warnings": warnings,
    }


def predict_vacancy(price, market_rent):
    """PLACEHOLDER. Illustrative days-to-lease at `price` given `market_rent`
    (use predict_rent(...)["estimate"]). Not estimated from data."""
    v = VACANCY_MODEL
    return v["base_days"] * (price / market_rent) ** v["price_sensitivity"]
