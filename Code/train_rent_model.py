#!/usr/bin/env python3
"""Fit the SF rent regression and write rent_model.py (the single file the app imports).

Usage:
    pip install pandas numpy statsmodels openpyxl
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
     is small, the script limits the model to n/10 predictors chosen by
     leave-one-out error, estimates an amenity only if at least 5 comps have it
     and 5 do not, and writes a rent_model.py that carries wide-interval
     warnings. Everything below describes input type 1.

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


def fit_comps(raw, source_label):
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

    info = dict(trained=str(datetime.date.today()), source=source_label,
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
rent = exp(that) * smearing. For a yes/no variable, exp(coef) - 1 is the percent
rent difference. The "_unknown" terms only absorb listings whose status was not
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
    print(f"\nleave-one-out error of candidate models (at most {r['budget']} predictors; "
          f"plain median of the other comps: ${r['base'][0]:,.0f}):")
    for mae, mape, preds in r["table"][:6]:
        print(f"  ${mae:>6,.0f} ({mape:4.1f}%)  {' + '.join(preds)}")
    lo = i["leave_one_out"]
    print(f"\nselected: {' + '.join(i['predictors'])}   in-sample R2 {i['r2_in_sample']}")
    print(f"leave-one-out MAE ${lo['model_mae_usd']:,} ({lo['model_mape_pct']}%) vs "
          f"${lo['baseline_mae_usd']:,} ({lo['baseline_mape_pct']}%) for the plain median"
          f"{'' if lo['beats_baseline'] else '   <-- does not clearly beat the median'}")
    for k, v in p["coefs"].items():
        eff = f"{(math.exp(v['coef']) - 1) * 100:+.0f}%" if k != "log_sqft" else "elasticity"
        print(f"  {k:<24} coef {v['coef']:+.3f} ({eff:>10})  95% CI [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]  p={v['p']:.3f}")
    print("\ncaveats:")
    for c in i["caveats"]:
        print("  -", c)


def main():
    ap = argparse.ArgumentParser(description="Fit the SF rent regression and write rent_model.py")
    ap.add_argument("input", help="listings .xlsx (sheet 'All SF Listings'), listings .csv, or RealtyAPI comps .csv")
    ap.add_argument("--out", default="rent_model.py")
    a = ap.parse_args()
    table = read_table(a.input)
    label = a.input.replace("\\", "/").split("/")[-1]
    if "asking_rent_usd_month" in table.columns:
        params, steps = fit_comps(table, label + " (RealtyAPI comps; advertised asking rents, hand-reviewed)")
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
