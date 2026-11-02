#!/usr/bin/env python3
"""Train the SF rent stacking model and save it to Code/models/rent_stack.joblib.

Working name while we test it; it will replace train_rent_model.py / rent_model.py.
The app (rent_lookup_app.py) still uses rent_model.py and is not touched here.

Usage (from the repo root):
    pip install pandas numpy scikit-learn joblib
    python Code/create_model_test.py                       # trains on Data/realty_training_combined.csv
    python Code/create_model_test.py --data other.csv --out Code/models/other.joblib
    python Code/create_model_test.py --predict             # load the saved model and run a sample prediction

What it does (follows test_models.ipynb):
  1. Cleans the unit-level rows with the same filters as clean_units() in train_rent_model.py.
  2. Builds features: beds, baths, log(sqft), distance to downtown, zip dummies and amenity dummies
     (an amenity needs 5 buildings with it and 5 without; an unknown amenity list gets the share of
     buildings that have it, never "no").
  3. Sets 20% of the buildings aside as a test set (no building is in both halves).
  4. Tunes ridge, SVM, KNN and gradient boosting with building-grouped 5-fold CV on the training set.
  5. Stacks them: out-of-fold predictions from building-grouped folds, blended by a non-negative
     linear regression. Reports CV and test RMSE for every model and the stack.
  6. Refits the stack on all buildings with the tuned settings and saves it (compressed) together with
     everything predict_rent() needs to turn raw inputs into features.

Rerun it whenever new data arrives; the previous model stays in git history.
"""
import argparse
import datetime
import math
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_DATA = os.path.join(REPO, "Data", "realty_training_combined.csv")
DEFAULT_OUT = os.path.join(HERE, "models", "rent_stack.joblib")

MIN_SQFT = 200            # smaller rows are shared-living rooms / non-standard units
MAX_BEDS = 4              # larger homes are too few to model
OUT_OF_SCOPE_TYPES = {"house", "single family", "single-family home", "townhouse", "multi-family"}
MIN_ZIP_BUILDINGS = 3     # zips with fewer buildings are pooled into "zip_other"
MIN_AMEN_BUILDINGS = 5    # an amenity needs this many buildings with it and without it
CBD_LAT, CBD_LON = 37.788, -122.407   # Union Square, for distance to downtown
AMENITY_GROUPS = {
    "in_unit_laundry": ["washer dryer"], "shared_laundry": ["laundry facilities"],
    "parking": ["parking", "garage parking"], "balcony_patio": ["balcony", "patio"],
    "doorman_concierge": ["doorman", "concierge"],
}
CORE_COLS = ["beds", "baths", "log_sqft", "dist_km"]
TEST_FRACTION = 0.20
SPLIT_SEED = 2026
N_FOLDS = 5

# Tuning grids (from test_models.ipynb; gradient boosting gets a wider grid since time is not a concern)
GRIDS = {
    "ridge": {"ridge__alpha": np.logspace(-1, 4, 60),
              "coreweight__core_weight": [1, 2, 5, 10, 30, 100]},
    "svm": {"svr__C": [0.1, 0.3, 1, 3, 10, 30],
            "svr__gamma": ["scale", 0.001, 0.003, 0.01, 0.03],
            "svr__epsilon": [0.01, 0.05, 0.1]},
    "knn": {"kneighborsregressor__n_neighbors": [3, 5, 7, 10, 15, 20, 30, 50],
            "kneighborsregressor__weights": ["uniform", "distance"],
            "kneighborsregressor__p": [1, 2],
            "coreweight__core_weight": [1, 3, 10]},
    "gbm": {"learning_rate": [0.02, 0.05, 0.1],
            "max_leaf_nodes": [7, 15, 31],
            "min_samples_leaf": [10, 20, 40],
            "l2_regularization": [0.0, 1.0]},
}


# ---------------------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------------------
def _tok(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower().replace("/", " ").replace("-", " "))).strip()


def _tokens(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return set()
    return {t for t in (_tok(x) for x in str(s).split(";")) if t}


def _dist_km(lat, lon):
    return np.hypot((np.asarray(lat, float) - CBD_LAT) * 111.0,
                    (np.asarray(lon, float) - CBD_LON) * 111.0 * math.cos(math.radians(37.8)))


class CoreWeight(BaseEstimator, TransformerMixin):
    """Multiply the first n_core columns (beds, baths, log_sqft, dist_km) by core_weight after
    scaling. A bigger weight means ridge shrinks them less and KNN distances lean on them more."""

    def __init__(self, core_weight=1.0, n_core=len(CORE_COLS)):
        self.core_weight = core_weight
        self.n_core = n_core

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        X[:, :self.n_core] *= self.core_weight
        return X


def clean(df):
    d = df.copy()
    for c in ("rent", "beds", "baths", "sqft"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["amen_known"] = d["amenities"].notna() & (d["amenities"].astype(str).str.strip() != "")
    if "date_downloaded" in d:   # newest download first, so the dedup keeps the latest rent
        d = d.sort_values("date_downloaded", ascending=False, kind="stable")
    steps = []

    def step(name, keep):
        nonlocal d
        before = len(d)
        d = d[keep(d)].copy()
        steps.append((name, before - len(d)))

    step("duplicate unit (same listing_key + unit_key)",
         lambda x: x["unit_key"].isna() | ~x.duplicated(["listing_key", "unit_key"], keep="first"))
    d["zip_code"] = d["zip_code"].astype(str).str.extract(r"(\d{5})", expand=False)
    step("missing rent, beds, baths or zip",
         lambda x: x.rent.notna() & x.baths.notna() & x.beds.notna() & x.zip_code.notna())
    step("outside SF (zip not 941xx)", lambda x: x.zip_code.str.startswith("941"))
    step("houses / townhouses",
         lambda x: ~x["property_type"].astype(str).str.strip().str.lower().isin(OUT_OF_SCOPE_TYPES))
    step(f"more than {MAX_BEDS} bedrooms", lambda x: x.beds <= MAX_BEDS)
    step("missing sqft", lambda x: x.sqft.notna())
    step(f"under {MIN_SQFT} sqft", lambda x: x.sqft >= MIN_SQFT)
    return d.reset_index(drop=True), steps


def build_features(d):
    """Return (data table, feature spec). The spec holds what predict_rent() needs to
    rebuild the same columns from raw inputs."""
    toks = d["amenities"].map(_tokens)
    known = d["amen_known"].values.astype(bool)
    grouped = {t for v in AMENITY_GROUPS.values() for t in v}
    A, A_tokens = {}, {}
    for k, v in AMENITY_GROUPS.items():
        A[k] = toks.map(lambda s, v=v: float(bool(s & set(v))))
        A_tokens[k] = list(v)
    for t in sorted({t for s, kn in zip(toks, known) if kn for t in s} - grouped):
        A[t.replace(" ", "_")] = toks.map(lambda s, t=t: float(t in s))
        A_tokens[t.replace(" ", "_")] = [t]
    A = pd.DataFrame(A, index=d.index)

    bld = d["listing_key"]
    keep_cols = []
    for c in A:
        per = A.loc[known, c].groupby(bld[known]).max()
        if (per == 1).sum() >= MIN_AMEN_BUILDINGS and (per == 0).sum() >= MIN_AMEN_BUILDINGS:
            keep_cols.append(c)
    A = A[keep_cols].copy()
    shares = A[known].groupby(bld[known]).max().mean()
    for c in A.columns:       # unknown amenity list -> typical building, never "no"
        A.loc[~known, c] = shares[c]

    zb = d.groupby("zip_code")["listing_key"].nunique()
    zgroups = sorted(str(z) for z in zb[zb >= MIN_ZIP_BUILDINGS].index)
    Z = pd.DataFrame({f"zip_{z}": (d["zip_code"] == z).astype(float) for z in zgroups}, index=d.index)
    Z["zip_other"] = (~d["zip_code"].isin(zgroups)).astype(float)
    if Z["zip_other"].sum() == 0:
        Z = Z.drop(columns="zip_other")

    dist = pd.Series(_dist_km(d["latitude"], d["longitude"]), index=d.index)
    dist = dist.fillna(dist.groupby(d["zip_code"]).transform("mean")).fillna(dist.mean())
    core = pd.DataFrame({"beds": d["beds"], "baths": d["baths"],
                         "log_sqft": np.log(d["sqft"]), "dist_km": dist})

    features = CORE_COLS + list(Z.columns) + list(A.columns)
    data = pd.concat([core, Z, A], axis=1)
    data["log_rent"] = np.log(d["rent"])
    data["rent"] = d["rent"]
    data["listing_key"] = d["listing_key"]

    spec = dict(
        features=features,
        zip_columns=list(Z.columns),
        zip_groups=zgroups,
        training_zips=sorted(d["zip_code"].unique().tolist()),
        zip_distance_km={str(z): round(float(v), 3) for z, v in dist.groupby(d["zip_code"]).mean().items()},
        mean_dist_km=float(dist.mean()),
        amenities=list(A.columns),
        amenity_tokens={c: A_tokens[c] for c in A.columns},
        amenity_share_yes={c: round(float(shares[c]), 4) for c in A.columns},
        means={c: float(core[c].mean()) for c in CORE_COLS},
        support=dict(beds=[float(d.beds.min()), float(d.beds.max())],
                     baths=[float(d.baths.min()), float(d.baths.max())],
                     sqft=[float(d.sqft.min()), float(d.sqft.max())],
                     beds_counts={str(int(b)): int(n) for b, n in d["beds"].round().value_counts().sort_index().items()}),
    )
    return data, spec


# ---------------------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------------------
def base_pipelines():
    return {
        "ridge": make_pipeline(StandardScaler(), CoreWeight(), Ridge()),
        "svm": make_pipeline(StandardScaler(), SVR(kernel="rbf")),
        "knn": make_pipeline(StandardScaler(), CoreWeight(), KNeighborsRegressor()),
        "gbm": HistGradientBoostingRegressor(max_iter=500, early_stopping=False, random_state=0),
    }


def make_stack(base_models, X, groups):
    """Out-of-fold base predictions from building-grouped folds, blended with a non-negative
    linear regression."""
    folds = list(GroupKFold(n_splits=N_FOLDS).split(X, groups=groups))
    return StackingRegressor(estimators=[(n, clone(m)) for n, m in base_models.items()],
                             final_estimator=LinearRegression(positive=True),
                             cv=folds, n_jobs=-1)


def _rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def train(data_path, out_path):
    t0 = datetime.datetime.now()
    raw = pd.read_csv(data_path)
    d, steps = clean(raw)
    print(f"read {len(raw)} rows from {data_path}")
    for name, n in steps:
        print(f"  dropped {n:>5}  {name}")
    print(f"kept {len(d)} units in {d['listing_key'].nunique()} buildings")

    data, spec = build_features(d)
    F = spec["features"]
    print(f"{len(F)} features: {len(CORE_COLS)} core, {len(spec['zip_columns'])} zip, {len(spec['amenities'])} amenity\n")

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=SPLIT_SEED)
    tr_idx, te_idx = next(gss.split(data, groups=data["listing_key"]))
    tr, te = data.iloc[tr_idx], data.iloc[te_idx]
    X_tr, y_tr, g_tr = tr[F], tr["log_rent"], tr["listing_key"]
    X_te, y_te = te[F], te["log_rent"]
    print(f"train: {len(tr)} units in {g_tr.nunique()} buildings; "
          f"test: {len(te)} units in {te['listing_key'].nunique()} buildings")

    cv = GroupKFold(n_splits=N_FOLDS)
    results, tuned = {}, {}
    for name, pipe in base_pipelines().items():
        search = GridSearchCV(pipe, GRIDS[name], cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
        search.fit(X_tr, y_tr, groups=g_tr)
        pred = search.predict(X_te)
        tuned[name] = search.best_estimator_
        results[name] = dict(cv_rmse_log=float(np.sqrt(-search.best_score_)),
                             test_rmse_log=_rmse(y_te, pred),
                             test_rmse_usd=_rmse(np.exp(y_te), np.exp(pred)),
                             best_params={k: (v.item() if hasattr(v, "item") else v)
                                          for k, v in search.best_params_.items()})
        r = results[name]
        print(f"{name:>5}: CV {r['cv_rmse_log']:.4f}  test {r['test_rmse_log']:.4f}  "
              f"(${r['test_rmse_usd']:,.0f})  {r['best_params']}")

    # CV RMSE of the stack; the out-of-fold residuals also give the interval and smearing factor
    oof = pd.Series(np.nan, index=X_tr.index)
    for a, b in cv.split(X_tr, y_tr, groups=g_tr):
        s = make_stack(tuned, X_tr.iloc[a], g_tr.iloc[a]).fit(X_tr.iloc[a], y_tr.iloc[a])
        oof.iloc[b] = s.predict(X_tr.iloc[b])
    stack = make_stack(tuned, X_tr, g_tr).fit(X_tr, y_tr)
    pred = stack.predict(X_te)
    results["stack"] = dict(cv_rmse_log=_rmse(y_tr, oof), test_rmse_log=_rmse(y_te, pred),
                            test_rmse_usd=_rmse(np.exp(y_te), np.exp(pred)),
                            blend={n: round(float(c), 3) for n, c in zip(tuned, stack.final_estimator_.coef_)})
    r = results["stack"]
    print(f"stack: CV {r['cv_rmse_log']:.4f}  test {r['test_rmse_log']:.4f}  "
          f"(${r['test_rmse_usd']:,.0f})  blend {r['blend']}")

    # out-of-building residuals (training-set CV + test set) for the 80% range and exp() bias correction
    resid = np.concatenate([(y_tr - oof).values, (y_te - pred).values])
    smearing = float(np.mean(np.exp(resid)))
    resid_q = {f"p{q}": float(np.percentile(resid, q)) for q in (10, 25, 50, 75, 90)}

    # final model: tuned settings, refit on every building
    print("\nrefitting the stack on all buildings...")
    final = make_stack(tuned, data[F], data["listing_key"]).fit(data[F], data["log_rent"])
    blend_all = {n: round(float(c), 3) for n, c in zip(tuned, final.final_estimator_.coef_)}

    bundle = dict(
        model=final,
        spec=spec,
        smearing=smearing,
        log_residual_quantiles_out_of_sample=resid_q,
        info=dict(
            trained=datetime.date.today().isoformat(),
            data_file=os.path.relpath(data_path, REPO),
            n_rows_read=int(len(raw)), n_used=int(len(d)), n_buildings=int(d["listing_key"].nunique()),
            cleaning_steps=steps,
            target="log(advertised asking monthly rent, USD)",
            validation=dict(split=f"{int(TEST_FRACTION * 100)}% of buildings held out (seed {SPLIT_SEED}); "
                                  f"{N_FOLDS}-fold building-grouped CV on the rest",
                            results=results),
            final_blend=blend_all,
            final_intercept=float(final.final_estimator_.intercept_),
            sklearn_version=sklearn.__version__,
            numpy_version=np.__version__,
            train_seconds=round((datetime.datetime.now() - t0).total_seconds()),
        ),
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    joblib.dump(bundle, out_path, compress=("xz", 9))
    size_kb = os.path.getsize(out_path) / 1024
    print(f"final blend {blend_all}")
    print(f"saved {out_path} ({size_kb:,.0f} KB) in {bundle['info']['train_seconds']} s")
    return bundle


# ---------------------------------------------------------------------------------------
# prediction (what the app will call once we switch over)
# ---------------------------------------------------------------------------------------
_LOADED = {}


def load_model(path=DEFAULT_OUT):
    if path not in _LOADED:
        _LOADED[path] = joblib.load(path)
    return _LOADED[path]


def predict_rent(bedrooms=None, bathrooms=None, sqft=None, zip_code=None, amenities=None,
                 latitude=None, longitude=None, model_path=DEFAULT_OUT):
    """Return {"estimate", "low", "high", "warnings"}, the same keys rent_model.predict_rent uses.

    amenities: a dict like {"in_unit_laundry": True, "gym": None} (None = not known) or a list of
    amenity names the building has. Unknown amenities get the share of buildings that have them.
    low/high are the 10th/90th percentile of errors on buildings the model had not seen.
    """
    b = load_model(model_path)
    spec = b["spec"]
    sup = spec["support"]
    warnings = []
    if bedrooms is not None and not sup["beds"][0] <= bedrooms <= sup["beds"][1]:
        warnings.append("bedrooms outside the range of the training data")
    if bathrooms is not None and not sup["baths"][0] <= bathrooms <= sup["baths"][1]:
        warnings.append("bathrooms outside the range of the training data")
    if sqft is not None and not sup["sqft"][0] <= sqft <= sup["sqft"][1]:
        warnings.append("square footage outside the range of the training data")

    z = str(zip_code)[:5] if zip_code is not None else None
    if latitude is not None and longitude is not None:
        dist = float(_dist_km(latitude, longitude))
    elif z in spec["zip_distance_km"]:
        dist = spec["zip_distance_km"][z]
    else:
        dist = spec["mean_dist_km"]
        warnings.append("no location given; using the average distance to downtown")
    row = {"beds": bedrooms, "baths": bathrooms, "log_sqft": math.log(sqft) if sqft else None, "dist_km": dist}
    for k in CORE_COLS:
        if row[k] is None:
            row[k] = spec["means"][k]
            warnings.append(f"{k} not given; using the training average")

    for c in spec["zip_columns"]:
        row[c] = 0.0
    if z is None:
        warnings.append("no zip code given; no zip effect applied")
    elif "zip_" + z in spec["zip_columns"]:
        row["zip_" + z] = 1.0
    elif z in spec["training_zips"] and "zip_other" in spec["zip_columns"]:
        row["zip_other"] = 1.0
    else:
        warnings.append("zip code is not in the training data; no zip effect applied")

    if isinstance(amenities, dict):
        flags = {k: (None if v is None else bool(v)) for k, v in amenities.items()}
    else:   # a list of names: anything on it is yes, everything else no
        have = {_tok(a).replace(" ", "_") for a in (amenities or [])}
        flags = {c: (c in have or bool(set(spec["amenity_tokens"][c]) & {_tok(a) for a in (amenities or [])}))
                 for c in spec["amenities"]} if amenities is not None else {}
    for c in spec["amenities"]:
        v = flags.get(c)
        row[c] = spec["amenity_share_yes"][c] if v is None else float(v)
    unused = [k for k in flags if k not in spec["amenities"]]
    if unused:
        warnings.append("not used by this model: " + ", ".join(sorted(unused)))

    X = pd.DataFrame([row], columns=spec["features"])
    est = math.exp(float(b["model"].predict(X)[0])) * b["smearing"]
    q = b["log_residual_quantiles_out_of_sample"]
    return {"estimate": round(est), "low": round(est * math.exp(q["p10"])),
            "high": round(est * math.exp(q["p90"])), "warnings": warnings}


def main():
    ap = argparse.ArgumentParser(description="Train the SF rent stacking model")
    ap.add_argument("--data", default=DEFAULT_DATA, help="unit-level training CSV (default %(default)s)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="where to save the model (default %(default)s)")
    ap.add_argument("--predict", action="store_true", help="skip training; load the saved model and run a sample")
    a = ap.parse_args()
    if not a.predict:
        train(a.data, a.out)
    if not os.path.exists(a.out):
        sys.exit(f"no model at {a.out}")
    sample = dict(bedrooms=1, bathrooms=1, sqft=700, zip_code="94107",
                  amenities={"in_unit_laundry": True, "gym": None})
    print("\nsample", sample, "->", predict_rent(**sample, model_path=a.out))


if __name__ == "__main__":
    # Run through the imported module so the saved model refers to create_model_test.CoreWeight
    # (not __main__.CoreWeight); otherwise other code could not load it.
    sys.path.insert(0, HERE)
    import create_model_test
    create_model_test.main()
