#!/usr/bin/env python3
"""Fit the SF rent regression and write rent_model.py (the single file the app imports).

Usage:
    pip install pandas numpy statsmodels scikit-learn openpyxl
    python train_rent_model.py san_francisco_rentals_by_neighborhood.xlsx
    python train_rent_model.py listings.csv --out rent_model.py

Two kinds of input are recognised automatically:

  1. The listings spreadsheet (sheet "All SF Listings") or a CSV with the same
     columns: Rent Price ($), Bedrooms, Bathrooms, Square Footage, Zip Code, and
     optionally Washer/Dryer, Parking/Garage, Balcony / Patio, Cooling / AC, Pool
     (values "Yes"/"No").

  2. A RealtyAPI comps CSV (recognised by its asking_rent_usd_month column): one
     row per comparable listing, with hand-reviewed amenity columns
     (laundry_reviewed, parking_reviewed, outdoor_space_reviewed, ...) and a
     comp_use column. Rows marked "Exclude" are dropped. Each amenity is coded
     yes / no / unknown (unknown is never treated as "no"). Because such a file
     is small, the script fits a LASSO (L1-penalized regression on log rent;
     the penalty is chosen by leave-one-out error and weak predictors are set
     to exactly zero). Pass --method subset for the older best-subset OLS
     (n/10 predictors chosen by leave-one-out error). Either way an amenity is
     estimated only if at least 5 comps have it and 5 do not, and the output
     rent_model.py carries wide-interval warnings. Needs scikit-learn.

  3. One or more unit-level files (several are pooled, see below). A RealtyAPI training file with one row per available unit (recognised by
     the columns rent, beds, baths, sqft, zip_code, amenities, listing_key;
     latitude, longitude and weight are used when present). Fits a lasso on
     log rent with beds, baths, log(sqft) and distance to downtown left
     unpenalized and zip effects and building-level amenities penalized. The
     penalty comes from cross-validation that holds out whole buildings
     (units in one building are not independent), and the reported error is
     measured the same way, re-choosing the penalty inside every fold. It also
     sets 20% of the buildings aside as a test set, fits on the rest, and
     reports the test RMSE (the final model is then refit on everything).

     Pooling:  python train_rent_model.py new_units.csv old_units.csv comps.csv listings.xlsx
     Files are read in the order given and an earlier file wins whenever the same
     unit (listing_key + unit_key) or floor plan appears again, so nothing is
     counted twice. Spreadsheet listings at an address already covered by a unit
     file are skipped; the rest are added as extra buildings with an unknown
     amenity list (treated as a typical building, never as 'no'). Amenities named
     only in the comps file are ignored so a naming difference cannot read as
     'not offered'. Everything from here on describes input type 1.

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
import itertools
import math
import pprint
import re
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


def read_table(path):
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path, sheet_name="All SF Listings")
    return pd.read_csv(path, dtype={"zip_code": str})


def prepare_listings(df):
    missing = [c for c in ("Rent Price ($)", "Bedrooms", "Bathrooms", "Square Footage", "Zip Code")
               if c not in df.columns]
    if missing:
        sys.exit("Input is missing required columns: " + ", ".join(missing))
    df = df.rename(columns=RENAME)
    for col, name in AMENITY_COLUMNS.items():
        if col in df.columns:
            df["amen_" + name] = (df[col].astype(str).str.strip().str.lower() == "yes").astype(float)
    return df


def load(path):
    return prepare_listings(read_table(path))


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


# ---------------------------------------------------------------------------
# Input type 2: RealtyAPI comps CSV (a small, hand-reviewed comparable set).
# Detected by the asking_rent_usd_month column. Small-sample safeguards:
#   * at most n/10 predictors, chosen by leave-one-out error
#   * an amenity is estimated only with >= MIN_CELL "yes" and >= MIN_CELL "no"
#   * unknown is its own category and never counts as "no"
# ---------------------------------------------------------------------------
MIN_CELL = 5              # an amenity needs this many yes AND no listings
MAX_MISSING_SHARE = 0.10  # square footage is used only if at least this complete


def _txt(s):
    return "" if pd.isna(s) else str(s).strip().lower()


def code_laundry(s):
    """1 = in-unit washer/dryer; 0 = shared, hookups only or none; NaN = unknown."""
    t = _txt(s)
    if t.startswith("in unit"):
        return 1.0
    if t.startswith("none") or "nearby" in t or "laundromat" in t:
        return 0.0
    if "verify" in t or "unverified" in t:
        return np.nan
    if "shared" in t or "hookup" in t:
        return 0.0
    return np.nan


def code_parking(s):
    """1 = parking included in the rent; 0 = none, street only or at extra cost."""
    t = _txt(s)
    if "included" in t or "private garage" in t:
        return 1.0
    if "no parking" in t or "street" in t or "extra fee" in t or "$" in t or "on request" in t:
        return 0.0
    return np.nan


def code_outdoor(s):
    """1 = private patio / deck / balcony; 0 = shared space only."""
    t = _txt(s)
    if t in ("", "unknown", "nan"):
        return np.nan
    return 1.0 if any(k in t for k in ("private", "patio", "balcony", "deck")) else 0.0


def code_utilities(s):
    """1 = at least one utility included; 0 = tenant pays all."""
    t = _txt(s)
    if t in ("", "unknown"):
        return np.nan
    if "included" in t:
        return 1.0
    if t.startswith("tenant pays"):
        return 0.0
    return np.nan


def code_known_only(s):
    """Columns that only ever say what is present: known -> 1, otherwise unknown."""
    return np.nan if _txt(s) in ("", "unknown", "nan") else 1.0


COMP_AMENITIES = {
    "in_unit_laundry": ("laundry_reviewed", code_laundry),
    "parking_included": ("parking_reviewed", code_parking),
    "private_outdoor": ("outdoor_space_reviewed", code_outdoor),
    "utilities_included": ("utilities_reviewed", code_utilities),
    "renovated": ("renovation_reviewed", code_known_only),
    "dishwasher": ("dishwasher_reviewed", code_known_only),
}


def clean_comps(raw):
    d = raw.copy()
    steps = []
    d["rent"] = pd.to_numeric(d["asking_rent_usd_month"], errors="coerce")
    d["baths"] = pd.to_numeric(d["baths"], errors="coerce")
    d["beds"] = pd.to_numeric(d["beds"], errors="coerce")
    d["sqft"] = pd.to_numeric(d["sqft"], errors="coerce") if "sqft" in d else np.nan

    def step(name, keep):
        nonlocal d
        before = len(d)
        d = d[keep(d)].copy()
        steps.append((name, before - len(d)))

    if "comp_use" in d:
        step('marked "Exclude" in comp_use (furnished short stays, not a true match, bad dates)',
             lambda x: x["comp_use"].astype(str).str.strip().str.lower() != "exclude")
    step("missing rent, bedrooms or bathrooms",
         lambda x: x.rent.notna() & x.baths.notna() & x.beds.notna())
    if {"listing_key", "unit_key"} <= set(d.columns):
        step("duplicate unit (same listing_key and unit_key)",
             lambda x: ~x.duplicated(["listing_key", "unit_key"], keep="first"))
    for name, (col, fn) in COMP_AMENITIES.items():
        d[name] = d[col].map(fn) if col in d else np.nan
    d["zip_code"] = d["zip_code"].astype(str).str.slice(0, 5) if "zip_code" in d else "NA"
    return d.reset_index(drop=True), steps


def _comp_design(df, preds, unk, means):
    X = pd.DataFrame(index=df.index)
    for p in preds:
        if p == "log_sqft":
            X[p] = np.log(df["sqft"]).fillna(means["log_sqft"])
        elif p in ("beds", "baths"):
            X[p] = df[p]
        else:
            X[p] = df[p].fillna(0.0)
            if p in unk:
                X[p + "_unknown"] = df[p].isna().astype(float)
    return sm.add_constant(X, has_constant="add")


def _comp_fit(df, preds, unk, means, robust=False):
    X = _comp_design(df, preds, unk, means)
    keep = [c for c in X.columns if c == "const" or X[c].std() > 0]
    y = np.log(df["rent"])
    m = sm.OLS(y, X[keep]).fit(cov_type="HC3") if robust else sm.OLS(y, X[keep]).fit()
    return m, keep


def _comp_loo(d, preds, unk, means):
    """Leave-one-out: refit without each comp and predict it."""
    res_log, abs_err, ape = [], [], []
    for i in range(len(d)):
        tr, te = d.drop(d.index[i]), d.iloc[[i]]
        m, keep = _comp_fit(tr, preds, unk, means)
        xt = _comp_design(te, preds, unk, means).reindex(columns=keep, fill_value=0.0)
        pl = float(m.predict(xt).iloc[0])
        smear = float(np.mean(np.exp(m.resid)))
        y = float(te["rent"].iloc[0])
        res_log.append(math.log(y) - pl)
        abs_err.append(abs(math.exp(pl) * smear - y))
        ape.append(abs_err[-1] / y)
    return float(np.mean(abs_err)), float(np.mean(ape) * 100), np.array(res_log)


def _median_loo(d):
    p = np.array([d.drop(d.index[i])["rent"].median() for i in range(len(d))])
    e = np.abs(p - d["rent"].values)
    return float(e.mean()), float((e / d["rent"].values).mean() * 100)


# ---- lasso (default for comps files) ---------------------------------------
N_BOOT = 300  # bootstrap resamples for the lasso intervals / selection frequency


def _lasso_X(df, cands, unk, means):
    return _comp_design(df, cands, unk, means).drop(columns="const")


def _lasso_fit(X, y, alpha=None):
    """Lasso on standardized predictors; returns (const, coef Series on the original
    scale, alpha). If alpha is None it is chosen by leave-one-out cross-validation."""
    from sklearn.linear_model import Lasso, LassoCV
    from sklearn.model_selection import LeaveOneOut
    mu, sd = X.mean(), X.std(ddof=0).replace(0, 1.0)
    Z = ((X - mu) / sd).values
    if alpha is None:
        yc = y - y.mean()
        amax = max(float(np.max(np.abs(Z.T @ yc))) / len(y), 1e-6)  # smallest alpha that zeroes everything
        grid = np.geomspace(amax, amax * 1e-3, 60)
        alpha = float(LassoCV(alphas=grid, cv=LeaveOneOut(), max_iter=100000).fit(Z, y).alpha_)
    m = Lasso(alpha=alpha, max_iter=100000).fit(Z, y)
    coef = pd.Series(m.coef_ / sd.values, index=X.columns)
    const = float(m.intercept_ - np.dot(coef.values, mu.values))
    return const, coef, alpha


def _lasso_loo(d, cands, unk, means):
    """Honest leave-one-out: alpha is re-chosen inside every fold, so the error
    includes the cost of the selection step."""
    X, y = _lasso_X(d, cands, unk, means), np.log(d["rent"].values)
    res_log, abs_err, ape = [], [], []
    for i in range(len(d)):
        tr = np.arange(len(d)) != i
        const, coef, _ = _lasso_fit(X[tr], y[tr])
        pl = const + float(X.iloc[i].values @ coef.values)
        smear = float(np.mean(np.exp(y[tr] - (const + X[tr].values @ coef.values))))
        res_log.append(y[i] - pl)
        abs_err.append(abs(math.exp(pl) * smear - math.exp(y[i])))
        ape.append(abs_err[-1] / math.exp(y[i]))
    return float(np.mean(abs_err)), float(np.mean(ape) * 100), np.array(res_log)


def _fit_comps_lasso(d, steps, raw, cands, unk, means, screen, source_label):
    n = len(d)
    X, y = _lasso_X(d, cands, unk, means), np.log(d["rent"].values)
    base_mae, base_mape = _median_loo(d)
    const, coef, alpha = _lasso_fit(X, y)
    mae, mape, res = _lasso_loo(d, cands, unk, means)

    rng = np.random.default_rng(0)
    boot = np.zeros((N_BOOT, X.shape[1]))
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        boot[b] = _lasso_fit(X.iloc[idx], y[idx], alpha)[1].values
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    freq = (boot != 0).mean(axis=0)

    kept = [c for c in X.columns if coef[c] != 0]
    fitted = const + X.values @ coef.values
    resid = y - fitted
    coefs = {c: {"coef": round(float(coef[c]), 5),
                 "ci95": [round(float(lo[j]), 5), round(float(hi[j]), 5)],
                 "selected_share": round(float(freq[j]), 3)}
             for j, c in enumerate(X.columns) if c in kept}
    names = [c for c in kept if not c.endswith("_unknown")]
    q = {k: round(float(np.quantile(res, v)), 4)
         for k, v in {"p10": .1, "p25": .25, "p75": .75, "p90": .9}.items()}
    shares = {a: round(float(d[a].mean()), 3) for a in names if a in COMP_AMENITIES}
    improvement = (base_mae - mae) / base_mae * 100
    beats = improvement >= 10
    dropped = [c for c in X.columns if c not in kept and not c.endswith("_unknown")]
    r2 = 1 - float(np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2))

    only_beds = sorted(d["beds"].unique())
    caveats = [f"Only {n} comps: every coefficient has a wide interval (bootstrap ci95; approximate for a lasso).",
               "Lasso shrinks coefficients toward zero, so the effects are deliberately smaller than "
               "an unpenalized fit would give; the penalty is chosen by leave-one-out error.",
               "Amenity effects are associations. Units with an amenity also tend to be newer, "
               "larger or better located, so they overstate what adding the amenity would earn.",
               "No location effect (too few comps per zip); the estimate reflects the comp set's average location."]
    if len(only_beds) == 1:
        caveats.insert(1, f"All comps are {only_beds[0]:g}-bedroom, so the bedroom effect cannot be "
                          f"estimated and the model is valid for {only_beds[0]:g}-bedroom units only.")
    if d["sqft"].notna().sum() and "log_sqft" not in cands:
        caveats.append(f"Square footage is not used: missing for {int(d['sqft'].isna().sum())} of {n} comps.")
    skipped = [f"{a} ({r['reason']})" for a, r in screen.items() if not r["estimable"]]
    if skipped:
        caveats.append("Amenities that could not be estimated: " + "; ".join(skipped) + ".")
    reliability = (f"low: fit on {n} comps; typical out-of-sample error about {mape:.0f}% "
                   f"(plain median of the comps: {base_mape:.0f}%)")
    if not beats:
        reliability += "; does NOT clearly beat the plain median"

    info = dict(trained=str(datetime.date.today()), source=source_label, method="lasso",
                alpha_standardized=round(alpha, 5), n_raw=int(len(raw)), n_used=int(n),
                cleaning=[f"dropped {name}: {c} rows" for name, c in steps],
                predictors=names, set_to_zero_by_lasso=dropped, r2_in_sample=round(r2, 3),
                leave_one_out=dict(model_mae_usd=round(mae), model_mape_pct=round(mape, 1),
                                   baseline="median rent of the other comps",
                                   baseline_mae_usd=round(base_mae), baseline_mape_pct=round(base_mape, 1),
                                   beats_baseline=bool(beats)),
                reliability=reliability, caveats=caveats)
    params = dict(
        info=info, const=round(const, 5), coefs=coefs,
        amenity_share_yes=shares, means={k: round(v, 5) for k, v in means.items()},
        smearing=round(float(np.mean(np.exp(resid))), 5), resid_q=q,
        support=dict(beds=[float(d["beds"].min()), float(d["beds"].max())],
                     baths=[float(d["baths"].min()), float(d["baths"].max())],
                     zips=sorted(d["zip_code"].unique().tolist()),
                     rent=[float(d["rent"].min()), float(d["rent"].max())]),
        screen=screen,
        _report=dict(table=None, alpha=alpha, base=(base_mae, base_mape), budget=None,
                     improvement=improvement, candidates=list(X.columns)))
    return params, steps


def _amenity_screen(d, control_baths):
    """Per amenity: how many yes / no / unknown, and (if enough of both) its effect
    on its own, controlling for bathrooms."""
    rows = {}
    for a in COMP_AMENITIES:
        n_yes, n_no = int((d[a] == 1).sum()), int((d[a] == 0).sum())
        row = {"n_yes": n_yes, "n_no": n_no, "n_unknown": int(d[a].isna().sum()),
               "estimable": False, "effect_pct": None, "ci95_pct": None, "p": None}
        if n_yes >= MIN_CELL and n_no >= MIN_CELL:
            s = d.dropna(subset=[a])
            X = pd.DataFrame({a: s[a]})
            if control_baths:
                X["baths"] = s["baths"]
            m = sm.OLS(np.log(s["rent"]), sm.add_constant(X)).fit(cov_type="HC3")
            ci = m.conf_int()
            row.update(estimable=True,
                       effect_pct=round(float((math.exp(m.params[a]) - 1) * 100), 1),
                       ci95_pct=[round(float((math.exp(ci.loc[a, 0]) - 1) * 100), 1),
                                 round(float((math.exp(ci.loc[a, 1]) - 1) * 100), 1)],
                       p=round(float(m.pvalues[a]), 4))
        else:
            row["reason"] = (f'needs at least {MIN_CELL} "yes" and {MIN_CELL} "no" listings; '
                             f"has {n_yes} yes / {n_no} no")
        rows[a] = row
    return rows


def fit_comps(raw, source_label, method="lasso"):
    d, steps = clean_comps(raw)
    n = len(d)
    if n < 15:
        sys.exit(f"Only {n} usable comps after cleaning; too few to fit anything.")
    cands = []
    if (d["beds"].value_counts() >= MIN_CELL).sum() >= 2:
        cands.append("beds")
    if d["baths"].nunique() >= 2:
        cands.append("baths")
    if d["sqft"].notna().sum() >= 15 and d["sqft"].isna().mean() <= MAX_MISSING_SHARE:
        cands.append("log_sqft")
    screen = _amenity_screen(d, "baths" in cands)
    cands += [a for a, r in screen.items() if r["estimable"]]
    if not cands:
        sys.exit("Nothing in this file varies enough to estimate.")
    unk = {a for a in COMP_AMENITIES if d[a].isna().any()}
    means = {"beds": float(d["beds"].mean()), "baths": float(d["baths"].mean()),
             "log_sqft": float(np.log(d["sqft"]).mean()) if d["sqft"].notna().any() else 0.0}

    if method == "lasso":
        return _fit_comps_lasso(d, steps, raw, cands, unk, means, screen, source_label)

    budget = max(1, n // 10)
    base_mae, base_mape = _median_loo(d)
    table = []
    for k in range(1, min(budget, len(cands)) + 1):
        for preds in itertools.combinations(cands, k):
            mae, mape, _ = _comp_loo(d, list(preds), unk, means)
            table.append((mae, mape, preds))
    table.sort(key=lambda t: (t[0], len(t[2])))
    best_mae, best_mape, best = table[0]
    best = list(best)

    m, keep = _comp_fit(d, best, unk, means, robust=True)
    ci = m.conf_int()
    coefs = {k: {"coef": round(float(m.params[k]), 5), "se": round(float(m.bse[k]), 5),
                 "p": round(float(m.pvalues[k]), 5),
                 "ci95": [round(float(ci.loc[k, 0]), 5), round(float(ci.loc[k, 1]), 5)]}
             for k in keep if k != "const"}
    _, _, res = _comp_loo(d, best, unk, means)
    q = {k: round(float(np.quantile(res, v)), 4)
         for k, v in {"p10": .1, "p25": .25, "p75": .75, "p90": .9}.items()}
    shares = {a: round(float(d[a].mean()), 3) for a in best if a in COMP_AMENITIES}
    improvement = (base_mae - best_mae) / base_mae * 100
    beats = improvement >= 10

    only_beds = sorted(d["beds"].unique())
    caveats = [f"Only {n} comps: every coefficient has a wide confidence interval (see ci95).",
               "Amenity effects are associations. Units with an amenity also tend to be newer, "
               "larger or better located, so they overstate what adding the amenity would earn.",
               "No location effect (too few comps per zip); the estimate reflects the comp set's average location.",
               f"Predictors were picked from {len(table)} candidate models by leave-one-out error, "
               "so the reported error is optimistic."]
    if len(only_beds) == 1:
        caveats.insert(1, f"All comps are {only_beds[0]:g}-bedroom, so the bedroom effect cannot be "
                          f"estimated and the model is valid for {only_beds[0]:g}-bedroom units only.")
    if d["sqft"].notna().sum() and "log_sqft" not in cands:
        caveats.append(f"Square footage is not used: missing for {int(d['sqft'].isna().sum())} of {n} comps.")
    skipped = [f"{a} ({r['reason']})" for a, r in screen.items() if not r["estimable"]]
    if skipped:
        caveats.append("Amenities that could not be estimated: " + "; ".join(skipped) + ".")
    reliability = (f"low: fit on {n} comps; typical out-of-sample error about {best_mape:.0f}% "
                   f"(plain median of the comps: {base_mape:.0f}%)")
    if not beats:
        reliability += "; does NOT clearly beat the plain median"

    info = dict(trained=str(datetime.date.today()), source=source_label, method="best-subset OLS",
                n_raw=int(len(raw)), n_used=int(n),
                cleaning=[f"dropped {name}: {c} rows" for name, c in steps],
                predictors=[p for p in best], r2_in_sample=round(float(m.rsquared), 3),
                leave_one_out=dict(model_mae_usd=round(best_mae), model_mape_pct=round(best_mape, 1),
                                   baseline="median rent of the other comps",
                                   baseline_mae_usd=round(base_mae), baseline_mape_pct=round(base_mape, 1),
                                   beats_baseline=bool(beats)),
                reliability=reliability, caveats=caveats)
    params = dict(
        info=info, const=round(float(m.params["const"]), 5), coefs=coefs,
        amenity_share_yes=shares, means={k: round(v, 5) for k, v in means.items()},
        smearing=round(float(np.mean(np.exp(m.resid))), 5), resid_q=q,
        support=dict(beds=[float(d["beds"].min()), float(d["beds"].max())],
                     baths=[float(d["baths"].min()), float(d["baths"].max())],
                     zips=sorted(d["zip_code"].unique().tolist()),
                     rent=[float(d["rent"].min()), float(d["rent"].max())]),
        screen=screen,
        _report=dict(table=table, base=(base_mae, base_mape), budget=budget, improvement=improvement))
    return params, steps


TEMPLATE_COMPS = '''"""SF rent model fit on a small set of hand-reviewed comparable listings.

Trained @@TRAINED@@ on @@N_USED@@ comps (of @@N_RAW@@ rows in the file).
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
rent = exp(that) * smearing. MODEL_INFO["method"] says how the coefficients were
fit: "lasso" shrinks weak coefficients toward zero (or exactly to zero, see
MODEL_INFO["set_to_zero_by_lasso"]) so effects are conservative. For a yes/no
variable, exp(coef) - 1 is the percent rent difference. The "_unknown" terms only absorb listings whose status was not
reported; they are never used for pricing.
"""
import math

MODEL_INFO = @@INFO@@

PRICE_MODEL = {
    "target": "log(advertised asking monthly rent, USD)",
    "const": @@CONST@@,
    "coefficients": @@COEFS@@,
    "amenity_share_yes": @@SHARES@@,  # share of comps with the amenity, among those reporting it
    "means": @@MEANS@@,
    "smearing": @@SMEAR@@,  # corrects exp() bias when converting log-rent back to dollars
    "log_residual_quantiles_out_of_sample": @@RESID_Q@@,
    "support": @@SUPPORT@@,
}

# Log-rent effect of each amenity used for pricing (same numbers as in coefficients).
AMENITY_EFFECTS = @@AMEN_EFFECTS@@

# Every amenity the data could code, including those too sparse to estimate.
AMENITY_SCREEN = @@SCREEN@@

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
'''


def render_comps(p):
    fmt = lambda o: pprint.pformat(o, width=100, sort_dicts=False)  # noqa: E731
    amen_eff = {k: v["coef"] for k, v in p["coefs"].items() if k in p["amenity_share_yes"]}
    out = TEMPLATE_COMPS
    for key, val in {
        "@@TRAINED@@": p["info"]["trained"], "@@N_USED@@": str(p["info"]["n_used"]),
        "@@N_RAW@@": str(p["info"]["n_raw"]), "@@INFO@@": fmt(p["info"]),
        "@@CONST@@": repr(p["const"]), "@@COEFS@@": fmt(p["coefs"]),
        "@@SHARES@@": fmt(p["amenity_share_yes"]), "@@MEANS@@": fmt(p["means"]),
        "@@SMEAR@@": repr(p["smearing"]), "@@RESID_Q@@": fmt(p["resid_q"]),
        "@@SUPPORT@@": fmt(p["support"]), "@@AMEN_EFFECTS@@": fmt(amen_eff),
        "@@SCREEN@@": fmt(p["screen"]),
    }.items():
        out = out.replace(key, val)
    return out


def report_comps(p, steps):
    r, i = p["_report"], p["info"]
    for name, n in steps:
        print(f"dropped {n:>3}  {name}")
    print(f"\nused {i['n_used']} of {i['n_raw']} rows")
    print('\namenities (yes / no / unknown) and each one\'s effect on its own, controlling for bathrooms:')
    for a, s in p["screen"].items():
        head = f"  {a:<19} {s['n_yes']:>2} / {s['n_no']:>2} / {s['n_unknown']:>2}"
        if s["estimable"]:
            print(f"{head}   {s['effect_pct']:+.0f}%  95% CI [{s['ci95_pct'][0]:+.0f}%, {s['ci95_pct'][1]:+.0f}%]  p={s['p']:.3f}")
        else:
            print(f"{head}   not estimable: {s['reason']}")
    if r["table"] is None:
        print(f"\nlasso on {', '.join(r['candidates'])}")
        print(f"penalty (standardized alpha) {r['alpha']:.4f}, chosen by leave-one-out cross-validation")
        print(f"set to zero by the lasso: {', '.join(i['set_to_zero_by_lasso']) or 'none'}")
    else:
        print(f"\nleave-one-out error of candidate models (at most {r['budget']} predictors; "
              f"plain median of the other comps: ${r['base'][0]:,.0f}):")
        for mae, mape, preds in r["table"][:6]:
            print(f"  ${mae:>6,.0f} ({mape:4.1f}%)  {' + '.join(preds)}")
    lo = i["leave_one_out"]
    print(f"\nselected: {' + '.join(i['predictors']) or '(nothing: constant only)'}   in-sample R2 {i['r2_in_sample']}")
    print(f"leave-one-out MAE ${lo['model_mae_usd']:,} ({lo['model_mape_pct']}%) vs "
          f"${lo['baseline_mae_usd']:,} ({lo['baseline_mape_pct']}%) for the plain median"
          f"{'' if lo['beats_baseline'] else '   <-- does not clearly beat the median'}")
    for k, v in p["coefs"].items():
        eff = f"{(math.exp(v['coef']) - 1) * 100:+.0f}%" if k != "log_sqft" else "elasticity"
        tail = (f"p={v['p']:.3f}" if "p" in v else f"kept in {v['selected_share']:.0%} of bootstrap fits")
        print(f"  {k:<24} coef {v['coef']:+.3f} ({eff:>10})  95% CI [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]  {tail}")
    print("\ncaveats:")
    for c in i["caveats"]:
        print("  -", c)


# =====================================================================================
# Input type 3: unit-level training data (one row per available unit), from one file or
# several files pooled. A RealtyAPI training file has the columns rent, beds, baths,
# sqft, zip_code, latitude, longitude, amenities, listing_key (and unit_key). Lasso
# with cross-validation that holds out whole buildings, plus a building-grouped
# train/test split that reports RMSE.
# =====================================================================================
UNITS_REQUIRED = {"rent", "beds", "baths", "sqft", "zip_code", "amenities", "listing_key"}
MIN_SQFT = 200            # smaller rows are shared-living rooms / non-standard units
MAX_BEDS = 4              # larger homes are too few to model
OUT_OF_SCOPE_TYPES = {"house", "single family", "single-family home", "townhouse", "multi-family"}
MIN_ZIP_BUILDINGS = 3     # zips with fewer buildings are pooled into "zip_other"
MIN_AMEN_BUILDINGS = 5    # an amenity needs this many buildings with it and without it
CBD_LAT, CBD_LON = 37.788, -122.407   # Union Square, for the distance-to-downtown variable
CORE = ["beds", "baths", "log_sqft", "dist_km"]   # never penalized by the lasso
RATIOS = np.geomspace(1.0, 1e-3, 60)  # penalty as a fraction of the smallest penalty that zeroes everything
CV_REPEATS = 3            # repeats of the outer (error-measuring) cross-validation
SELECT_REPEATS = 20       # repeats of 10-fold CV used to choose the final penalty (the curve is flat and noisy)
INNER_REPEATS = 2         # repeats of 5-fold CV for the penalty inside each outer fold
HOLDOUT_FRACTION = 0.20   # share of buildings set aside as the test set
HOLDOUT_SEED = 42
AMENITY_GROUPS = {
    "in_unit_laundry": ["washer dryer"], "shared_laundry": ["laundry facilities"],
    "parking": ["parking", "garage parking"], "balcony_patio": ["balcony", "patio"],
    "doorman_concierge": ["doorman", "concierge"],
}


def _dist_km(lat, lon):
    return np.hypot((np.asarray(lat, float) - CBD_LAT) * 111.0,
                    (np.asarray(lon, float) - CBD_LON) * 111.0 * math.cos(math.radians(37.8)))


def _norm_addr(a):
    a = str(a).lower().split(", san francisco")[0]
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", a)).strip()


def _bkey(address, zip_code):
    """Building key that survives address-range formatting: first street number, first street
    word and zip ('1080-1084 Folsom St' and '1080 Folsom St' give the same key)."""
    a = _norm_addr(address)
    m = re.match(r"(\d+)(?:\s+\d+)*\s+([a-z]+)", a)
    z = re.search(r"(\d{5})", str(zip_code))
    return (m.group(1), m.group(2), z.group(1) if z else None) if m else None


def _tok(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower().replace("/", " ").replace("-", " "))).strip()


def _tokens(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return set()
    return {t for t in (_tok(x) for x in str(s).split(";")) if t}


# ---- pooling several files into one unit-level table ---------------------------------
POOL_COLS = ["listing_key", "unit_key", "floor_plan", "rent", "beds", "baths", "sqft", "zip_code",
             "latitude", "longitude", "amenities", "property_type", "address"]


def _source_kind(df, path):
    cols = set(df.columns)
    if "asking_rent_usd_month" in cols:
        return "comps"
    if UNITS_REQUIRED <= cols:
        return "units"
    if "Rent Price ($)" in cols:
        return "listings"
    sys.exit(f"{path}: not a recognised training file (needs a rent column and either the unit-level "
             f"columns {sorted(UNITS_REQUIRED)}, the comps columns, or the listings spreadsheet columns).")


def _adapt(df, kind):
    """Map one file to POOL_COLS (+ amen_known). Returns (frame, rows_dropped_inside_file)."""
    inner = 0
    if kind == "units":
        o = pd.DataFrame({c: df[c] if c in df else np.nan for c in POOL_COLS})
    elif kind == "comps":
        d = df
        if "comp_use" in d:
            d = d[d["comp_use"].astype(str).str.strip().str.lower() != "exclude"]
            inner = len(df) - len(d)
        pt = d["property_type_api"].astype(str).str.contains("Single-Family", na=False)
        o = pd.DataFrame({"listing_key": d["listing_key"], "unit_key": d["unit_key"], "floor_plan": np.nan,
                          "rent": d["asking_rent_usd_month"], "beds": d["beds"], "baths": d["baths"],
                          "sqft": d["sqft"], "zip_code": d["zip_code"], "latitude": d["latitude"],
                          "longitude": d["longitude"], "amenities": d["amenities_api"],
                          "property_type": np.where(pt, "House", "Apartment"), "address": d["address"]})
    else:  # listings spreadsheet
        key = ["Formatted Address", "Rent Price ($)", "Bedrooms", "Bathrooms", "Square Footage", "Listed Date"]
        d = df.drop_duplicates([c for c in key if c in df.columns])
        inner = len(df) - len(d)
        o = pd.DataFrame({"listing_key": np.nan, "unit_key": np.nan, "floor_plan": np.nan,
                          "rent": d["Rent Price ($)"], "beds": d["Bedrooms"], "baths": d["Bathrooms"],
                          "sqft": d["Square Footage"], "zip_code": d["Zip Code"], "latitude": np.nan,
                          "longitude": np.nan, "amenities": np.nan, "property_type": d["Property Type"],
                          "address": d["Formatted Address"]})
    o = o.reset_index(drop=True)
    o["amen_known"] = o["amenities"].notna() & (o["amenities"].astype(str).str.strip() != "")
    return o, inner


def load_pool(paths):
    """Read several training files (newest / most trusted first) and pool them without
    double counting. Returns (pooled frame, steps, per-file summary)."""
    items = []
    for p in paths:
        df = read_table(p)
        kind = _source_kind(df, p)
        o, inner = _adapt(df, kind)
        name = p.replace("\\", "/").split("/")[-1]
        o["source"] = name
        items.append(dict(name=name, kind=kind, frame=o, rows=len(df), inner=inner))
    if not any(i["kind"] == "units" for i in items):
        sys.exit("Pooling needs at least one unit-level RealtyAPI training file to define the amenity vocabulary.")
    vocab = set()
    for i in items:
        if i["kind"] == "units":
            for s in i["frame"]["amenities"]:
                vocab |= _tokens(s)
    steps = []
    for i in items:
        if i["inner"]:
            steps.append((f"{i['name']}: rows excluded or repeated inside the file "
                          f"(comp_use 'Exclude' or an identical repeated listing)", i["inner"]))
        if i["kind"] != "units":
            # keep only amenities that the unit-level files also use, so that an amenity named
            # differently in this file can never be mistaken for "not offered" elsewhere
            i["frame"]["amenities"] = i["frame"]["amenities"].map(
                lambda s: "; ".join(sorted(_tokens(s) & vocab)) if isinstance(s, str) else s)
    addr_key, bkey_key, unit_sig = {}, {}, set()
    for i in items:
        f = i["frame"]
        for a, z, k in zip(f["address"], f["zip_code"], f["listing_key"]):
            if isinstance(k, str):
                addr_key.setdefault(_norm_addr(a), k)
                bkey_key.setdefault(_bkey(a, z), k)
        for z, b, ba, sq, r_ in zip(f["zip_code"], f["beds"], f["baths"], f["sqft"], f["rent"]):
            unit_sig.add((str(z)[:5], b, ba, sq, r_))
    pooled = []
    for i in items:
        f = i["frame"].copy()
        if f["listing_key"].isna().all():
            na = f["address"].map(_norm_addr)
            mapped = na.map(addr_key)
            bk = pd.Series([_bkey(a, z) for a, z in zip(f["address"], f["zip_code"])], index=f.index)
            mapped = mapped.where(mapped.notna(), bk.map(lambda x: bkey_key.get(x) if x is not None else None))
            covered = mapped.notna()
            if covered.sum():
                steps.append((f"{i['name']}: listings at a building already covered by the unit-level files "
                              f"(same street number, street and zip, so counting them again would double count)",
                              int(covered.sum())))
            f = f[~covered].copy()
            same = pd.Series([(str(z)[:5], b, ba, sq, r_) in unit_sig for z, b, ba, sq, r_ in
                              zip(f["zip_code"], f["beds"], f["baths"], f["sqft"], f["rent"])], index=f.index)
            if same.sum():
                steps.append((f"{i['name']}: listings with the same zip, beds, baths, sqft and rent as a unit already "
                              f"in the unit-level files (probable repeat under a different address)", int(same.sum())))
            f = f[~same].copy()
            f["listing_key"] = "addr:" + na[f.index]
        pooled.append(f)
    pool = pd.concat(pooled, ignore_index=True)
    before = pool["source"].value_counts()
    has_unit = pool["unit_key"].notna()
    dup_unit = has_unit & pool.duplicated(["listing_key", "unit_key"], keep="first")
    plan = pool["unit_key"].isna() & pool["floor_plan"].notna()
    dup_plan = plan & pool.duplicated(["listing_key", "floor_plan", "beds", "baths", "sqft"], keep="first")
    for mask, label in ((dup_unit, "the same unit (listing_key + unit_key) already in an earlier file"),
                        (dup_plan, "the same floor plan (building + plan + beds + baths + sqft) already in an earlier file")):
        for src, n in pool[mask]["source"].value_counts().items():
            steps.append((f"{src}: {label}", int(n)))
    pool = pool[~(dup_unit | dup_plan)].reset_index(drop=True)
    summary = [dict(file=i["name"], kind=i["kind"], rows_read=i["rows"], rows_after_dedup=int((pool["source"] == i["name"]).sum()))
               for i in items]
    return pool, steps, summary


def clean_units(raw):
    d = raw.copy()
    steps = []
    for c in ("rent", "beds", "baths", "sqft"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if "source" not in d:
        d["source"] = "file"
    if "amen_known" not in d:
        d["amen_known"] = d["amenities"].notna() & (d["amenities"].astype(str).str.strip() != "")

    def step(name, keep):
        nonlocal d
        before = len(d)
        d = d[keep(d)].copy()
        steps.append((name, before - len(d)))

    if "unit_key" in d:
        step("duplicate unit (same unit_key)",
             lambda x: x["unit_key"].isna() | ~x.duplicated(["listing_key", "unit_key"], keep="first"))
    d["zip_code"] = d["zip_code"].astype(str).str.extract(r"(\d{5})", expand=False)
    step("missing rent, bedrooms, bathrooms or zip code",
         lambda x: x.rent.notna() & x.baths.notna() & x.beds.notna() & x.zip_code.notna())
    step("outside San Francisco (zip code not 941xx)", lambda x: x.zip_code.str.startswith("941"))
    if "property_type" in d:
        step("houses and townhouses (single-family market, not multifamily apartments)",
             lambda x: ~x["property_type"].astype(str).str.strip().str.lower().isin(OUT_OF_SCOPE_TYPES))
    step(f"more than {MAX_BEDS} bedrooms (too few to model; includes room-by-room listings)", lambda x: x.beds <= MAX_BEDS)
    step("missing square footage (a size-controlled model cannot use them)", lambda x: x.sqft.notna())
    step(f"under {MIN_SQFT} sq ft (shared-living rooms and other non-standard units)", lambda x: x.sqft >= MIN_SQFT)
    d = d.reset_index(drop=True)
    if "weight" in raw and "source" not in raw:
        w = pd.to_numeric(d["weight"], errors="coerce")
    else:
        w = pd.Series(np.nan, index=d.index)
    w = w.fillna(1.0 / d.groupby("listing_key")["rent"].transform("size"))
    d["w"] = w / w.groupby(d["listing_key"]).transform("sum")   # every building counts as 1 in total
    return d, steps


def _units_design(d):
    """Core variables, penalized variables (zip effects + amenities) and the metadata
    the prediction function needs."""
    toks = d["amenities"].map(_tokens)
    known = d["amen_known"].values.astype(bool)
    grouped = {t for v in AMENITY_GROUPS.values() for t in v}
    A = {}
    for k, v in AMENITY_GROUPS.items():
        A[k] = toks.map(lambda s, v=v: float(bool(s & set(v))))
    for t in sorted({t for s, kn in zip(toks, known) if kn for t in s} - grouped):
        A[t.replace(" ", "_")] = toks.map(lambda s, t=t: float(t in s))
    A = pd.DataFrame(A, index=d.index)
    bld = d["listing_key"]
    keep, skipped = [], []
    for c in A:
        per = A.loc[known, c].groupby(bld[known]).max()
        n_yes, n_no = int((per == 1).sum()), int((per == 0).sum())
        if n_yes >= MIN_AMEN_BUILDINGS and n_no >= MIN_AMEN_BUILDINGS:
            keep.append(c)
        else:
            skipped.append(f"{c} ({n_yes} buildings with, {n_no} without)")
    A = A[keep].copy()
    per_bld = A[known].groupby(bld[known]).max()
    shares = {c: float(per_bld[c].mean()) for c in A.columns}
    for c in A.columns:                      # unknown amenity list: neutral (typical building), never "no"
        A.loc[~known, c] = shares[c]

    zb = d.groupby("zip_code")["listing_key"].nunique()
    zgroups = sorted(str(z) for z in zb[zb >= MIN_ZIP_BUILDINGS].index)
    Z = pd.DataFrame({f"zip_{z}": (d["zip_code"] == z).astype(float) for z in zgroups}, index=d.index)
    Z["zip_other"] = (~d["zip_code"].isin(zgroups)).astype(float)
    if Z["zip_other"].sum() == 0:
        Z = Z.drop(columns="zip_other")

    lat, lon = d["latitude"].astype(float), d["longitude"].astype(float)
    dist = pd.Series(_dist_km(lat, lon), index=d.index)
    dist = dist.fillna(dist.groupby(d["zip_code"]).transform("mean")).fillna(dist.mean())
    core = pd.DataFrame({"beds": d["beds"], "baths": d["baths"], "log_sqft": np.log(d["sqft"]),
                         "dist_km": dist})
    pen = pd.concat([Z, A], axis=1)
    meta = dict(amenities=list(A.columns), skipped=skipped, zip_groups=zgroups, shares=shares,
                zip_dist={str(z): round(float(v), 3) for z, v in dist.groupby(d["zip_code"]).mean().items()},
                beds_counts={str(int(b)): int(n) for b, n in d["beds"].round().value_counts().sort_index().items()},
                n_buildings=int(bld.nunique()), n_amen_unknown=int((~known).sum()))
    return core, pen, meta


def _u_path(C, P, y, w, ratios):
    """Lasso path with `C` (core, unpenalized) and `P` (penalized) columns on log rent `y`.
    Unpenalized columns are handled exactly by projecting them out of y and P first
    (Frisch-Waugh), the lasso runs on the leftovers, and the core coefficients are then
    recovered. Returns beta (1 + n_core, n_penalties) and gamma (n_pen, n_penalties)."""
    from sklearn.linear_model import enet_path
    sw = np.sqrt(w / w.mean())
    Cc = np.column_stack([np.ones(len(C)), C])
    by = np.linalg.lstsq(Cc * sw[:, None], y * sw, rcond=None)[0]
    m = len(ratios)
    if P.shape[1] == 0:
        return np.repeat(by[:, None], m, axis=1), np.zeros((0, m))
    Bx = np.linalg.lstsq(Cc * sw[:, None], P * sw[:, None], rcond=None)[0]
    mu = (w[:, None] * P).sum(0) / w.sum()
    s = np.sqrt((w[:, None] * (P - mu) ** 2).sum(0) / w.sum())
    s[s == 0] = 1.0
    Xs = (P - Cc @ Bx) / s
    yt = y - Cc @ by
    amax = max(float(np.max(np.abs((Xs * sw[:, None]).T @ (yt * sw)))) / len(y), 1e-12)
    _, coefs, _ = enet_path(Xs * sw[:, None], yt * sw, l1_ratio=1.0, alphas=amax * ratios, max_iter=50000)
    gam = coefs / s[:, None]
    return by[:, None] - Bx @ gam, gam


def _u_predict(C, P, beta, gam):
    return np.column_stack([np.ones(len(C)), C]) @ beta + P @ gam


def _u_folds(groups, k, seed):
    u = np.unique(groups)
    np.random.default_rng(seed).shuffle(u)
    fid = {b: i % k for i, b in enumerate(u)}
    return np.array([fid[b] for b in groups])


def _u_cv_curve(C, P, y, w, groups, k, reps, seed):
    """Weighted CV error of log rent at every penalty; whole buildings are held out."""
    err, tot = np.zeros(len(RATIOS)), 0.0
    for r in range(reps):
        f = _u_folds(groups, k, seed + r)
        for i in range(k):
            tr, te = f != i, f == i
            beta, gam = _u_path(C[tr], P[tr], y[tr], w[tr], RATIOS)
            pr = _u_predict(C[te], P[te], beta, gam)
            err += (w[te, None] * (y[te, None] - pr) ** 2).sum(0)
            tot += w[te].sum()
    return err / tot


def _metrics(pred, act, w):
    """Error measures for predicted vs actual rent in dollars (pred, act are arrays)."""
    e = pred - act
    ae = np.abs(e)
    res = np.log(act) - np.log(pred)
    lg = np.log(act)
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(ae.mean()), mape=float((ae / act).mean() * 100),
                r2=float(1 - np.sum(e ** 2) / np.sum((act - act.mean()) ** 2)),
                rmse_log=float(np.sqrt(np.mean(res ** 2))),
                r2_log=float(1 - np.sum(res ** 2) / np.sum((lg - lg.mean()) ** 2)),
                w_rmse=float(np.sqrt(np.average(e ** 2, weights=w))), w_mae=float(np.average(ae, weights=w)),
                w_mape=float(np.average(ae / act, weights=w) * 100))


def _u_fit_predict(C, P, y, w, tr, te, j):
    """Fit on rows `tr` at penalty index j and return dollar predictions for rows `te`."""
    beta, gam = _u_path(C[tr], P[tr], y[tr], w[tr], RATIOS[[j]])
    fit_tr = _u_predict(C[tr], P[tr], beta, gam)[:, 0]
    smear = float(np.average(np.exp(y[tr] - fit_tr), weights=w[tr]))
    return np.exp(_u_predict(C[te], P[te], beta, gam)[:, 0]) * smear


def _u_outer_cv(C, P, y, w, groups, raw_w):
    """Honest out-of-building predictions: in every outer fold the penalty is re-chosen
    on that fold's training buildings only, so the selection step is inside the test."""
    act = np.exp(y)
    out = dict(lasso=[], core=[], median=[], log_res=[], ratios=[])
    P0 = P[:, :0]
    for r in range(CV_REPEATS):
        f = _u_folds(groups, 10, 1000 + r)
        pl, pc, pm = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
        for i in range(10):
            tr, te = f != i, f == i
            e = _u_cv_curve(C[tr], P[tr], y[tr], w[tr], groups[tr], 5, INNER_REPEATS, 10 * r + i)
            j = int(np.argmin(e))
            out["ratios"].append(float(RATIOS[j]))
            pl[te] = _u_fit_predict(C, P, y, w, tr, te, j)
            pc[te] = _u_fit_predict(C, P0, y, w, tr, te, 0)
            pm[te] = np.median(act[tr])
        out["lasso"].append(pl)
        out["core"].append(pc)
        out["median"].append(pm)
        out["log_res"].append(y - np.log(pl))
    avg = lambda L: {k: float(np.mean([_metrics(p, act, raw_w)[k] for p in L])) for k in _metrics(L[0], act, raw_w)}  # noqa: E731
    out["scores"] = {k: avg(out[k]) for k in ("lasso", "core", "median")}
    out["cv_r2"] = out["scores"]["lasso"]["r2_log"]
    return out


def _u_holdout(C, P, y, w, groups, raw_w):
    """One train/test split, by building: the penalty is chosen by cross-validation on the
    training buildings only, and the test buildings are scored once."""
    ub = np.unique(groups)
    np.random.default_rng(HOLDOUT_SEED).shuffle(ub)
    n_test = max(1, int(round(len(ub) * HOLDOUT_FRACTION)))
    te = np.isin(groups, ub[:n_test])
    tr = ~te
    curve = np.mean([_u_cv_curve(C[tr], P[tr], y[tr], w[tr], groups[tr], 10, 1, 700 + s) for s in range(10)], axis=0)
    j = int(np.argmin(curve))
    act = np.exp(y[te])
    res = dict(n_train_units=int(tr.sum()), n_test_units=int(te.sum()), n_train_buildings=int(len(ub) - n_test),
               n_test_buildings=int(n_test), penalty_fraction_of_max=round(float(RATIOS[j]), 4))
    res["lasso"] = _metrics(_u_fit_predict(C, P, y, w, tr, te, j), act, raw_w[te])
    res["core"] = _metrics(_u_fit_predict(C, P[:, :0], y, w, tr, te, 0), act, raw_w[te])
    res["median"] = _metrics(np.full(te.sum(), np.median(np.exp(y[tr]))), act, raw_w[te])
    return res


def fit_units(raw, source_label, penalty_fraction=None, extra_steps=None, files=None):
    d, steps = clean_units(raw)
    steps = list(extra_steps or []) + steps
    n = len(d)
    if n < 100 or d["listing_key"].nunique() < 20:
        sys.exit(f"Only {n} usable rows in {d['listing_key'].nunique()} buildings; too few for this model.")
    core, pen, meta = _units_design(d)
    C, P = core.values, pen.values
    y, w, groups = np.log(d["rent"].values), d["w"].values, d["listing_key"].values

    cv = _u_outer_cv(C, P, y, w, groups, w)
    ho = _u_holdout(C, P, y, w, groups, w)
    curve = np.mean([_u_cv_curve(C, P, y, w, groups, 10, 1, 200 + s) for s in range(SELECT_REPEATS)], axis=0)
    j = int(np.argmin(curve)) if penalty_fraction is None else int(np.argmin(np.abs(RATIOS - penalty_fraction)))
    beta, gam = _u_path(C, P, y, w, RATIOS[[j]])
    beta, gam = beta[:, 0], gam[:, 0]
    fitted = _u_predict(C, P, beta[:, None], gam[:, None])[:, 0]
    resid = y - fitted
    smear = float(np.average(np.exp(resid), weights=w))
    r2 = 1 - float(np.sum(w * resid ** 2) / np.sum(w * (y - np.average(y, weights=w)) ** 2))
    r2_unit = 1 - float(np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2))

    rng = np.random.default_rng(0)
    ub = np.unique(groups)
    idx = {b: np.where(groups == b)[0] for b in ub}
    bb, gg = [], []
    for _ in range(N_BOOT):
        ii = np.concatenate([idx[b] for b in rng.choice(ub, len(ub))])
        b_, g_ = _u_path(C[ii], P[ii], y[ii], w[ii], RATIOS[[j]])
        bb.append(b_[:, 0])
        gg.append(g_[:, 0])
    bb, gg = np.array(bb), np.array(gg)
    blo, bhi = np.percentile(bb, [2.5, 97.5], axis=0)
    glo, ghi = np.percentile(gg, [2.5, 97.5], axis=0)
    freq = (gg != 0).mean(axis=0)

    coefs = {}
    for k, name in enumerate(CORE):
        coefs[name] = {"coef": round(float(beta[k + 1]), 5), "ci95": [round(float(blo[k + 1]), 5), round(float(bhi[k + 1]), 5)]}
    pcols = list(pen.columns)
    for k, name in enumerate(pcols):
        if gam[k] != 0:
            coefs[name] = {"coef": round(float(gam[k]), 5), "ci95": [round(float(glo[k]), 5), round(float(ghi[k]), 5)],
                           "selected_share": round(float(freq[k]), 3)}
    kept_amen = [c for c in pcols if c in meta["amenities"] and gam[pcols.index(c)] != 0]
    zeroed = [c for c in pcols if gam[pcols.index(c)] == 0]
    q = {k: round(float(np.quantile(np.concatenate(cv["log_res"]), v)), 4)
         for k, v in {"p10": .1, "p25": .25, "p75": .75, "p90": .9}.items()}
    sc = cv["scores"]
    improvement = (sc["median"]["mae"] - sc["lasso"]["mae"]) / sc["median"]["mae"] * 100
    means = {c: round(float(np.average(core[c], weights=w)), 5) for c in CORE}
    src = pd.DataFrame({"source": d["source"], "res": resid})
    source_check = {s: dict(units=int(len(g)), buildings=int(d.loc[g.index, "listing_key"].nunique()),
                            mean_log_residual=round(float(g["res"].mean()), 4))
                    for s, g in src.groupby("source")}

    caveats = [
        f"Fit on {n} units in {meta['n_buildings']} buildings. Units in one building share their amenities and "
        "location, so the effective sample for amenity and zip effects is the number of buildings, not units.",
        "Validation holds out whole buildings, so the error quoted is for a unit in a building the model has not seen.",
        "Amenities are building-level advertised features. A building that does not list an amenity is coded "
        "'no', but listings often omit amenities they have, which pulls amenity effects toward zero. "
        f"{meta['n_amen_unknown']} units had no amenity list at all and are treated as a typical building, not as 'no'.",
        "Amenity and zip effects are associations. Buildings with more amenities tend to be newer and better "
        "located, and the lasso picks among highly overlapping amenities somewhat arbitrarily.",
        "Asking rents from available units, not signed leases; fees and concessions are not modeled. "
        "Scope: apartments and condos with up to 4 bedrooms inside San Francisco.",
    ]
    thin = [k + "-bed" for k, v in meta["beds_counts"].items() if v < 30]
    if thin:
        caveats.append(f"Very few {', '.join(thin)} units in the training data: estimates for those sizes are not reliable.")
    if meta["skipped"]:
        caveats.append("Amenities too rare or too common to estimate: " + "; ".join(meta["skipped"]) + ".")
    reliability = (f"moderate: typical out-of-building error about {sc['lasso']['mape']:.0f}% per listing "
                   f"(plain median: {sc['median']['mape']:.0f}%); cross-validated R2 on log rent {sc['lasso']['r2_log']:.2f}")
    rnd = lambda m: {k: round(v, 3 if k.startswith(("r2", "rmse_log")) else 1) for k, v in m.items()}  # noqa: E731
    info = dict(trained=str(datetime.date.today()), source=source_label,
                method="lasso (L1), penalty by grouped cross-validation",
                penalty_fraction_of_max=round(float(RATIOS[j]), 4),
                n_raw=int(len(raw)), n_used=int(n), n_buildings=meta["n_buildings"],
                files=files or [], cleaning=[f"dropped {name}: {c} rows" for name, c in steps],
                weights="every building counts as 1 in total, so large buildings do not dominate the fit",
                predictors=[c for c in coefs], amenities_kept=kept_amen, set_to_zero_by_lasso=zeroed,
                source_check=source_check,
                r2_in_sample=round(r2, 3), r2_in_sample_unit_weighted=round(r2_unit, 3),
                r2_cross_validated=round(sc["lasso"]["r2_log"], 3),
                validation=dict(
                    design=f"10-fold cross-validation holding out whole buildings, repeated {CV_REPEATS}x; "
                           "the penalty is re-chosen (5-fold, buildings held out) inside every fold",
                    cross_validated=dict(lasso=rnd(sc["lasso"]), beds_baths_sqft_distance_only=rnd(sc["core"]),
                                         plain_median=rnd(sc["median"])),
                    holdout_test=dict(
                        design=f"{int(HOLDOUT_FRACTION * 100)}% of buildings (random, seed {HOLDOUT_SEED}) set aside; "
                               "model and penalty fit on the other buildings only; test buildings scored once",
                        **{k: (rnd(v) if isinstance(v, dict) else v) for k, v in ho.items()}),
                    mae_improvement_over_median_pct=round(improvement, 1)),
                reliability=reliability, caveats=caveats)
    params = dict(
        info=info, const=round(float(beta[0]), 5), coefs=coefs,
        amenity_share_yes={c: round(meta["shares"][c], 3) for c in kept_amen},
        means=means, smearing=round(smear, 5), resid_q=q,
        support=dict(beds=[float(d["beds"].min()), float(d["beds"].max())], beds_counts=meta["beds_counts"],
                     baths=[float(d["baths"].min()), float(d["baths"].max())],
                     sqft=[float(d["sqft"].min()), float(d["sqft"].max())],
                     zips=sorted(d["zip_code"].unique().tolist()),
                     rent=[float(d["rent"].min()), float(d["rent"].max())]),
        zip_groups=[f"zip_{z}" for z in meta["zip_groups"]], zip_distance_km=meta["zip_dist"],
        screen=dict(skipped=meta["skipped"]),
        _report=dict(curve=(curve, j), cv=cv, holdout=ho, candidates=pcols, n_amen=len(meta["amenities"])))
    return params, steps


def report_units(p, steps):
    i, r = p["info"], p["_report"]
    for name, n in steps:
        print(f"dropped {n:>4}  {name}")
    print(f"\nused {i['n_used']} units in {i['n_buildings']} buildings (of {i['n_raw']} rows after pooling)")
    for f in i["files"]:
        print(f"  {f['file']}: {f['rows_read']} rows read, {f['rows_after_dedup']} after removing duplicates")
    print(f"lasso over {len(r['candidates'])} penalized variables ({r['n_amen']} amenities + zip effects); "
          f"unpenalized: beds, baths, log(sqft), distance to downtown")
    gain = (1 - r["curve"][0][r["curve"][1]] / r["curve"][0][0]) * 100
    print(f"penalty = {i['penalty_fraction_of_max']:.3f} of the penalty that zeroes everything "
          f"(chosen by grouped cross-validation; versus dropping every zip and amenity term it lowers "
          f"cross-validated squared error by {gain:.1f}%)")
    v = i["validation"]
    h = v["holdout_test"]
    print(f"\nTEST SET ({h['design']}):")
    print(f"  {h['n_train_buildings']} training buildings ({h['n_train_units']} units), "
          f"{h['n_test_buildings']} test buildings ({h['n_test_units']} units)")
    for nm, key in (("lasso", "lasso"), ("beds+baths+sqft+distance only", "core"), ("plain median", "median")):
        m = h[key]
        print(f"  {nm:<31} test RMSE ${m['rmse']:>6,.0f}   MAE ${m['mae']:>6,.0f} ({m['mape']:.1f}%)   R2 {m['r2']:.2f} (log {m['r2_log']:.2f})")
    cvs = v["cross_validated"]
    print(f"\nrepeated cross-validation, whole buildings held out ({v['design']}):")
    for nm, key in (("lasso", "lasso"), ("beds+baths+sqft+distance only", "beds_baths_sqft_distance_only"), ("plain median", "plain_median")):
        m = cvs[key]
        print(f"  {nm:<31} RMSE ${m['rmse']:>6,.0f}   MAE ${m['mae']:>6,.0f} ({m['mape']:.1f}%)   R2 {m['r2']:.2f} (log {m['r2_log']:.2f})   "
              f"building-weighted RMSE ${m['w_rmse']:,.0f}")
    print(f"R2 on log rent: in-sample {i['r2_in_sample']} (building-weighted) / {i['r2_in_sample_unit_weighted']} (per unit); "
          f"cross-validated {i['r2_cross_validated']}")
    print("\nmean in-sample log residual by source file (near 0 = the pooled sources agree):")
    for s, c in i["source_check"].items():
        print(f"  {s:<45} {c['units']:>5} units  {c['buildings']:>4} buildings  {c['mean_log_residual']:+.3f}")
    print("\ncoefficients (95% interval from resampling whole buildings):")
    for k, c in p["coefs"].items():
        eff = {"log_sqft": "elasticity", "dist_km": f"{(math.exp(c['coef']) - 1) * 100:+.1f}%/km"}.get(
            k, f"{(math.exp(c['coef']) - 1) * 100:+.1f}%")
        tail = f"  kept in {c['selected_share']:.0%} of resamples" if "selected_share" in c else ""
        print(f"  {k:<20} {c['coef']:+.4f} ({eff:>10})  [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]{tail}")
    print(f"set to zero: {', '.join(i['set_to_zero_by_lasso']) or 'none'}")
    print("\ncaveats:")
    for c in i["caveats"]:
        print("  -", c)


TEMPLATE_UNITS = '''"""SF rent model fit on available-unit listings (lasso, penalty chosen by cross-validation).

Trained @@TRAINED@@ on @@N_USED@@ units in @@N_BUILDINGS@@ buildings.
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

MODEL_INFO = @@INFO@@

PRICE_MODEL = {
    "target": "log(advertised asking monthly rent, USD)",
    "const": @@CONST@@,
    "coefficients": @@COEFS@@,
    "amenity_share_yes": @@SHARES@@,  # share of buildings with the amenity (kept amenities only)
    "means": @@MEANS@@,
    "smearing": @@SMEAR@@,  # corrects exp() bias when converting log-rent back to dollars
    "log_residual_quantiles_out_of_sample": @@RESID_Q@@,
    "support": @@SUPPORT@@,
    "zip_groups": @@ZIP_GROUPS@@,  # zips with their own effect; other training zips share "zip_other"
    "zip_distance_km": @@ZIP_DIST@@,  # average distance to Union Square, used when no lat/lon is given
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
'''


def render_units(p):
    fmt = lambda o: pprint.pformat(o, width=100, sort_dicts=False)  # noqa: E731
    out = TEMPLATE_UNITS
    for key, val in {
        "@@TRAINED@@": p["info"]["trained"], "@@N_USED@@": str(p["info"]["n_used"]),
        "@@N_BUILDINGS@@": str(p["info"]["n_buildings"]), "@@INFO@@": fmt(p["info"]),
        "@@CONST@@": repr(p["const"]), "@@COEFS@@": fmt(p["coefs"]),
        "@@SHARES@@": fmt(p["amenity_share_yes"]), "@@MEANS@@": fmt(p["means"]),
        "@@SMEAR@@": repr(p["smearing"]), "@@RESID_Q@@": fmt(p["resid_q"]),
        "@@SUPPORT@@": fmt(p["support"]), "@@ZIP_GROUPS@@": fmt(p["zip_groups"]),
        "@@ZIP_DIST@@": fmt(p["zip_distance_km"]),
    }.items():
        out = out.replace(key, val)
    return out


def main():
    ap = argparse.ArgumentParser(description="Fit the SF rent regression and write rent_model.py")
    ap.add_argument("input", nargs="+",
                    help="one training file: listings .xlsx (sheet 'All SF Listings'), listings .csv, RealtyAPI comps "
                         ".csv, or a unit-level RealtyAPI training .csv. Several files are pooled without double "
                         "counting; list the newest / most trusted first (it wins on duplicate units) and include at "
                         "least one unit-level RealtyAPI training file.")
    ap.add_argument("--out", default="rent_model.py")
    ap.add_argument("--penalty-fraction", type=float, default=None,
                    help="unit-level training files only: override the cross-validated lasso penalty with this "
                         "fraction (0-1) of the penalty that zeroes everything. Smaller keeps more zip/amenity "
                         "terms. Not chosen by cross-validation, so say so wherever the numbers are used.")
    ap.add_argument("--method", choices=["lasso", "subset"], default="lasso",
                    help="small hand-reviewed comps files only: lasso (default) or best-subset OLS chosen by "
                         "leave-one-out. Unit-level training files always use the lasso.")
    a = ap.parse_args()
    if len(a.input) > 1:
        pool, pre_steps, files = load_pool(a.input)
        names = ", ".join(f["file"] for f in files)
        params, steps = fit_units(pool, f"{len(files)} files pooled and de-duplicated: {names}",
                                  a.penalty_fraction, pre_steps, files)
        open(a.out, "w", encoding="utf-8").write(render_units(params))
        report_units(params, steps)
        print("\nwrote", a.out)
        return
    path = a.input[0]
    table = read_table(path)
    if "asking_rent_usd_month" not in table.columns and UNITS_REQUIRED <= set(table.columns):
        label = path.replace("\\", "/").split("/")[-1]
        params, steps = fit_units(table, label + " (RealtyAPI available units; advertised asking rents)",
                                  a.penalty_fraction)
        open(a.out, "w", encoding="utf-8").write(render_units(params))
        report_units(params, steps)
        print("\nwrote", a.out)
        return
    label = path.replace("\\", "/").split("/")[-1]
    if "asking_rent_usd_month" in table.columns:
        params, steps = fit_comps(table, label + " (RealtyAPI comps; advertised asking rents, hand-reviewed)",
                                   a.method)
        open(a.out, "w", encoding="utf-8").write(render_comps(params))
        report_comps(params, steps)
        print("\nwrote", a.out)
        return
    raw = prepare_listings(table)
    source = label + " (active listings; asking rents, not signed leases)"
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
