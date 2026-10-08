#!/usr/bin/env python3
"""Fit the SF rent regression and write rent_model.py (the single file the app imports).

Usage:
    pip install pandas numpy statsmodels openpyxl
    python train_rent_model.py san_francisco_rentals_by_neighborhood.xlsx
    python train_rent_model.py listings.csv --out rent_model.py

Input: the listings spreadsheet (sheet "All SF Listings") or a CSV with the same
columns: Rent Price ($), Bedrooms, Bathrooms, Square Footage, Zip Code, and
optionally Washer/Dryer, Parking/Garage, Balcony / Patio, Cooling / AC, Pool
(values "Yes"/"No").

What it does:
  1. Cleans the listings (rules printed at run time and saved in the output file).
  2. Fits log(rent) ~ bedrooms + bathrooms + log(sqft) + zip effects
     (+ any amenity that is flagged "Yes" often enough, see MIN_AMENITY_YES).
  3. Scores it with repeated 10-fold cross-validation against a baseline.
  4. Writes rent_model.py with the coefficients, validation numbers and the
     predict functions. The vacancy curve stays a placeholder: it needs
     closed-listing / time-on-market data, which this input does not have.

Rerun it whenever new data arrives; the old rent_model.py stays in git history.
"""
import argparse
import datetime
import pprint
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

RENAME = {"Rent Price ($)": "rent", "Square Footage": "sqft", "Bedrooms": "bd",
          "Bathrooms": "ba", "Zip Code": "zip", "Formatted Address": "addr"}
AMENITY_COLUMNS = {"Washer/Dryer": "washer_dryer", "Parking/Garage": "parking",
                   "Balcony / Patio": "balcony_patio", "Cooling / AC": "air_conditioning",
                   "Pool": "pool"}
MIN_AMENITY_YES = 15      # an amenity needs this many "Yes" rows to be estimated
MAX_AMENITY_SHARE = 0.95  # ...and must not be "Yes" on nearly every row
MIN_ZIP_LISTINGS = 12     # smaller zips are pooled into "other"


def load(path):
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, sheet_name="All SF Listings")
    else:
        df = pd.read_csv(path)
    missing = [c for c in ("Rent Price ($)", "Bedrooms", "Bathrooms", "Square Footage", "Zip Code")
               if c not in df.columns]
    if missing:
        sys.exit("Input is missing required columns: " + ", ".join(missing))
    df = df.rename(columns=RENAME)
    for col, name in AMENITY_COLUMNS.items():
        if col in df.columns:
            df["amen_" + name] = (df[col].astype(str).str.strip().str.lower() == "yes").astype(float)
    return df


def clean(raw):
    d = raw.copy()
    steps = []

    def step(name, keep):
        nonlocal d
        before = len(d)
        d = d[keep(d)].copy()
        steps.append((name, before - len(d)))

    step("missing bedrooms, bathrooms or square footage",
         lambda x: x.bd.notna() & x.ba.notna() & x.sqft.notna())
    step("bedrooms above 5 (data errors / rooming houses)", lambda x: x.bd <= 5)
    step("square footage outside 150-4,000", lambda x: x.sqft.between(150, 4000))
    step("rent under $1,000 (SRO / shared)", lambda x: x.rent >= 1000)
    step("rent per sq ft outside $2-$12 (typos / hotel rooms)",
         lambda x: (x.rent / x.sqft).between(2, 12))
    d["zip"] = d["zip"].astype(str).str.strip().str.slice(0, 5)
    cnt = d["zip"].value_counts()
    keep_zips = [z for z in cnt.index if cnt[z] >= MIN_ZIP_LISTINGS]
    d["zg"] = d["zip"].where(d["zip"].isin(keep_zips), "other")
    amen = [c[5:] for c in d.columns if c.startswith("amen_")
            and d[c].sum() >= MIN_AMENITY_YES and d[c].mean() <= MAX_AMENITY_SHARE]
    return d, steps, cnt.index[0], amen


def design(df, zlevels, amen):
    X = pd.DataFrame({"bedrooms": df.bd, "bathrooms": df.ba, "log_sqft": np.log(df.sqft)},
                     index=df.index)
    for z in zlevels:
        X["zip_" + z] = (df.zg == z).astype(float)
    for a in amen:
        X["amen_" + a] = df["amen_" + a]
    return sm.add_constant(X, has_constant="add")


def cross_validate(d, zl, amen, columns):
    """Repeated 10-fold CV: our model vs. median rent by zip group x bedrooms."""
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(10):
        idx = rng.permutation(len(d))
        pm, pb = np.zeros(len(d)), np.zeros(len(d))
        for fo in np.array_split(idx, 10):
            tr = np.setdiff1d(idx, fo)
            dt, dv = d.iloc[tr], d.iloc[fo]
            m = sm.OLS(np.log(dt.rent), design(dt, zl, amen)).fit()
            smear = np.mean(np.exp(m.resid))
            pm[fo] = np.exp(m.predict(design(dv, zl, amen).reindex(columns=columns, fill_value=0.0))) * smear
            g = dt.groupby(["zg", "bd"]).rent.median()
            gb = dt.groupby("bd").rent.median()
            pb[fo] = [g.get((z, b), gb.get(b, dt.rent.median())) for z, b in zip(dv.zg, dv.bd)]
        err = lambda p: np.abs(p - d.rent.values)  # noqa: E731
        rows.append(((err(pm).mean(), (err(pm) / d.rent.values).mean() * 100),
                     (err(pb).mean(), (err(pb) / d.rent.values).mean() * 100)))
    model = np.mean([r[0] for r in rows], axis=0)
    base = np.mean([r[1] for r in rows], axis=0)
    return model, base


def out_of_sample_quantiles(d, zl, amen, columns):
    rng = np.random.default_rng(1)
    idx = rng.permutation(len(d))
    oof = np.zeros(len(d))
    for fo in np.array_split(idx, 10):
        tr = np.setdiff1d(idx, fo)
        dt, dv = d.iloc[tr], d.iloc[fo]
        m = sm.OLS(np.log(dt.rent), design(dt, zl, amen)).fit()
        pred = m.predict(design(dv, zl, amen).reindex(columns=columns, fill_value=0.0)).values
        oof[fo] = np.log(dv.rent.values) - pred
    return {k: round(float(np.quantile(oof, v)), 4)
            for k, v in {"p10": .1, "p25": .25, "p75": .75, "p90": .9}.items()}


def fit(raw, source_label, n_raw):
    d, steps, ref, amen = clean(raw)
    if len(d) < 100:
        sys.exit(f"Only {len(d)} usable listings after cleaning; too few to fit.")
    zl = [z for z in sorted(d.zg.unique()) if z != ref]
    X = design(d, zl, amen)
    f = sm.OLS(np.log(d.rent), X).fit(cov_type="HC3")
    ci = f.conf_int()
    smear = float(np.mean(np.exp(f.resid)))
    coefs = {k: {"coef": round(float(f.params[k]), 5), "se": round(float(f.bse[k]), 5),
                 "p": round(float(f.pvalues[k]), 5),
                 "ci95": [round(float(ci.loc[k, 0]), 5), round(float(ci.loc[k, 1]), 5)]}
             for k in ["bedrooms", "bathrooms", "log_sqft"]}
    zeff = {ref: 0.0}
    for z in zl:
        zeff[z] = round(float(f.params["zip_" + z]), 5)
    cnts = d.zg.value_counts().to_dict()
    N = len(d)
    city_int = float(f.params["const"]) + sum(cnts[z] / N * zeff[z] for z in zeff)
    amen_effects = {a: round(float(f.params["amen_" + a]), 5) for a in amen}
    cvm, cvb = cross_validate(d, zl, amen, X.columns)
    cleaning = [f"dropped {name}: {n} rows" for name, n in steps]
    if amen:
        cleaning.append("amenities estimated: " + ", ".join(amen))
    return dict(
        info=dict(trained=str(datetime.date.today()), source=source_label, n_raw=n_raw, n_used=N,
                  cleaning=cleaning, r2=round(float(f.rsquared), 3),
                  cv=dict(method="10-fold cross-validation repeated 10x",
                          model_mae_usd=round(float(cvm[0])), model_mape_pct=round(float(cvm[1]), 1),
                          baseline="median rent by zip x bedrooms",
                          baseline_mae_usd=round(float(cvb[0])), baseline_mape_pct=round(float(cvb[1]), 1))),
        coefs=coefs, const=round(float(f.params["const"]), 5), ref_zip=ref, zip_effects=zeff,
        zip_counts={k: int(v) for k, v in cnts.items()}, city_intercept=round(city_int, 5),
        smearing=round(smear, 5), resid_q=out_of_sample_quantiles(d, zl, amen, X.columns),
        train_range=dict(bedrooms=[0, 5], bathrooms=[float(d.ba.min()), float(d.ba.max())],
                         sqft=[float(d.sqft.min()), float(d.sqft.max())]),
        amenities=amen_effects), steps


TEMPLATE = '''"""SF rent model: regression coefficients and prediction functions in ONE file.

Trained %(trained)s on %(n_used)d cleaned listings (of %(n_raw)d raw).
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
A coefficient c on bedrooms/bathrooms means about (exp(c)-1)*100 %% more rent per
extra bedroom/bathroom; the log_sqft coefficient is an elasticity (a 10%% larger
unit rents for about 10*c %% more).
"""
import math

MODEL_INFO = %(info)s

PRICE_MODEL = {
    "target": "log(asking monthly rent, USD)",
    "const": %(const)r,                 # intercept for the reference zip (%(ref_zip)s)
    "city_intercept": %(city_intercept)r,  # intercept averaged over zips; used when zip is unknown
    "coefficients": %(coefs)s,
    "zip_effects": %(zip_effects)s,
    "zip_listing_counts": %(zip_counts)s,
    "smearing": %(smearing)r,           # corrects exp() bias when converting log-rent back to dollars
    "log_residual_quantiles_out_of_sample": %(resid_q)s,
    "train_range": %(train_range)s,
}

# Placeholder: filled in once amenity data exists. Format: {"washer_dryer": 0.04, ...}
# (each value is the log-rent effect; predict_rent adds it when the amenity is passed in).
AMENITY_EFFECTS = %(amenities)s

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
    about 80%% of comparable listings landed inside it. zip_code=None uses a
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
'''


def render(p):
    fmt = lambda o: pprint.pformat(o, width=100, sort_dicts=False)  # noqa: E731
    return TEMPLATE % dict(
        trained=p["info"]["trained"], n_used=p["info"]["n_used"], n_raw=p["info"]["n_raw"],
        info=fmt(p["info"]), const=p["const"], ref_zip=p["ref_zip"], city_intercept=p["city_intercept"],
        coefs=fmt(p["coefs"]), zip_effects=fmt(p["zip_effects"]), zip_counts=fmt(p["zip_counts"]),
        smearing=p["smearing"], resid_q=fmt(p["resid_q"]), train_range=fmt(p["train_range"]),
        amenities=fmt(p["amenities"]))


def main():
    ap = argparse.ArgumentParser(description="Fit the SF rent regression and write rent_model.py")
    ap.add_argument("input", help="listings .xlsx (sheet 'All SF Listings') or .csv")
    ap.add_argument("--out", default="rent_model.py")
    a = ap.parse_args()
    raw = load(a.input)
    source = a.input.replace("\\", "/").split("/")[-1] + \
        " (active listings; asking rents, not signed leases)"
    params, steps = fit(raw, source, len(raw))
    for name, n in steps:
        print(f"dropped {n:>4}  {name}")
    open(a.out, "w", encoding="utf-8").write(render(params))
    i = params["info"]
    print(f"\nused {i['n_used']} of {i['n_raw']} listings   R2 {i['r2']}")
    print(f"cross-validated MAE ${i['cv']['model_mae_usd']:,} ({i['cv']['model_mape_pct']}%)  "
          f"vs baseline ${i['cv']['baseline_mae_usd']:,} ({i['cv']['baseline_mape_pct']}%)")
    for k, v in params["coefs"].items():
        print(f"{k:<10} coef {v['coef']:+.3f}   95% CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]   p={v['p']:.3f}")
    if params["amenities"]:
        print("amenity effects:", params["amenities"])
    else:
        print("no amenity had enough 'Yes' listings; AMENITY_EFFECTS left empty")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
