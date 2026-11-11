#!/usr/bin/env python3
"""Train the SF rent model and save it to Code/models/rent_blend.joblib.

The model is an equal-weight average of four models of log rent: ridge regression, support vector
regression (RBF kernel), k-nearest neighbors and gradient boosting (HistGradientBoostingRegressor).

Working name while we test it; it replaced train_rent_model.py / rent_model.py. The app
(rent_lookup_app.py) imports this file for predict_rent() and model_info(); training only runs
when the file is executed as a script.

Usage (from the repo root):
    pip install pandas numpy scikit-learn joblib
    python Code/edit_training_data.py                      # first: adds the crime columns (needs geopandas)
    python Code/create_model_test.py                       # trains on Data/realty_training_combined_final.csv
    python Code/create_model_test.py --weights unit        # every unit counts once instead of every building
    python Code/create_model_test.py --under-weight 1.5    # underestimates cost 1.5x overestimates (default 1.25)
    python Code/create_model_test.py --data other.csv --out Code/models/other.joblib
    python Code/create_model_test.py --predict             # load the saved model and run a sample prediction

What it does (follows test_models.ipynb):
  1. Cleans the unit-level rows with the same filters as the old train_rent_model.py, plus the
     rent-per-sq-ft filter ($2-$12) the spreadsheet path used.
  2. Builds features: beds, baths, log(sqft), sq ft per bedroom, distance to downtown, map coordinates,
     zip, building amenities (an amenity needs 5 buildings with it and 5 without; an unknown amenity
     list gets the share of buildings that have it, never "no"), floor (parsed from the unit number),
     property type, units in the listing, a floor-plan flag, crime near the building (crime_pc1 /
     crime_pc2 from edit_training_data.py) and "nearby rents": the average log rent
     per sq ft of the nearest OTHER buildings. Nearby rents are computed inside every fit from that
     fit's training rows only, so no held-out building ever sees its own rent.
     The crime grid (Data/crime_grid.csv) is saved with the model so predict_rent() can look up
     crime for any latitude / longitude.
  3. Each base model gets the columns that suit it: ridge the dummies, KNN and SVM a short list of
     size and location (SVM also crime), gradient boosting the raw map coordinates and zip as one categorical column.
  4. Training 1 (test): sets 20% of the buildings aside, tunes ridge, SVM, KNN and gradient boosting with
     building-grouped 5-fold CV on the rest, averages them and reports the test RMSE of each model
     and of the average.
  5. Repeated CV: with the tuned settings, 3 x 5-fold building-grouped CV over all buildings scores the
     four models and their average. One ~90-building test set is noisy; use these numbers
     to compare versions. The out-of-fold errors also set the low/high range (by bedroom count).
  6. Training 2 (final): refits all four models on every building with the tuned settings and saves it
     (compressed) with everything predict_rent() needs to rebuild the features from raw inputs.

Weights: by default every building counts as 1 in total (as in the old lasso), so a 40-unit complex does
not outweigh 40 small buildings. Ridge, SVM and gradient boosting use the weights; KNN cannot.
RMSE is reported both per unit and per building (weighted).

Rerun it whenever new data arrives; the previous model stays in git history.

AI live-check rows (optional): a file from build_training_view.py has a data_source column
("realtyapi_download" or "ai_audit") and a source_weight column. Building weights are multiplied by
source_weight, so AI rows count for less, and every reported score (test set and repeated CV) is
computed on the original rows only, so they stay comparable with a model trained without AI rows.
Scores on the AI rows are reported separately. Without those columns nothing changes.

Asymmetric loss: underestimating rent is worse than overestimating it, so the estimate is judged by
asymmetric squared loss on log rent (about a percent error): r = log(actual) - log(estimate), loss
UNDER_WEIGHT * r^2 when r > 0 (an underestimate) and r^2 when r < 0. The four models are still
tuned and fit for the mean (plain squared loss). Two corrections are applied afterwards, both from the
out-of-building residuals of the average (step 5), kept apart so each can be judged on its own:
  1. bias correction: per bedroom count, the mean residual, so each bedroom count is unbiased
     (the blend can run high or low for a bedroom count; this has nothing to do with the loss);
  2. asymmetry shift: one overall shift that minimises the asymmetric loss of the bias-corrected
     residuals (their tau-expectile, tau = w / (1 + w)).
predict_rent() multiplies the estimate by exp(bias correction + shift). The low/high range is not
shifted: it describes where actual rents fall, which the corrections do not change.
--under-weight 1 gives asymmetry shift 0 (bias correction only).
Every score is also reported as root asymmetric loss ("asym"); RMSE gets worse by design when the
shift is on, so compare versions on asym. The corrections are chosen on the same out-of-fold
residuals they are scored on (a few numbers in all), so the corrected rows are slightly optimistic.
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
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sklearn.set_config(enable_metadata_routing=True)   # lets sample weights reach the models that accept them

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_DATA = os.path.join(REPO, "Data", "realty_training_combined_final.csv")
DEFAULT_CRIME_GRID = os.path.join(REPO, "Data", "crime_grid.csv")
DEFAULT_OUT = os.path.join(HERE, "models", "rent_blend.joblib")

MIN_SQFT = 200            # smaller rows are shared-living rooms / non-standard units
MAX_BEDS = 4              # larger homes are too few to model
PPSF_RANGE = (2, 12)      # rent per sq ft outside this is a typo, furnished/short-term or a hotel room
OUT_OF_SCOPE_TYPES = {"house", "single family", "single-family home", "townhouse", "multi-family"}
MIN_ZIP_BUILDINGS = 3     # zips with fewer buildings are pooled into "zip_other"
MIN_AMEN_BUILDINGS = 5    # an amenity needs this many buildings with it and without it
MAX_FLOOR = 60            # parsed floors above this are unit-numbering schemes, not floors
CBD_LAT, CBD_LON = 37.788, -122.407   # Union Square, for distance to downtown
KM_PER_DEG_LAT, KM_PER_DEG_LON = 111.0, 111.0 * math.cos(math.radians(37.8))
NEARBY_K = 8              # buildings averaged for the nearby-rents feature
SAME_BUILDING_KM = 0.02   # neighbours closer than this are the same building and are skipped
AMENITY_GROUPS = {
    "in_unit_laundry": ["washer dryer"], "shared_laundry": ["laundry facilities"],
    "parking": ["parking", "garage parking"], "balcony_patio": ["balcony", "patio"],
    "doorman_concierge": ["doorman", "concierge"],
}
CORE_COLS = ["beds", "baths", "log_sqft", "dist_km"]   # always first, so CoreWeight can find them
LOCATION_COLS = ["x_km", "y_km"]
NEARBY_COLS = ["nearby_ppsf", "nearby_dist_km"]        # added inside each fit by NearbyRents
EXTRA_COLS = ["sqft_per_bed", "floor", "floor_known", "condo", "ptype_unknown",
              "log_units_in_listing", "is_floor_plan"]
AI_SOURCE = "ai_audit"    # data_source of rows added by the AI live check (build_training_view.py)
CRIME_COLS = ["crime_pc1", "crime_pc2"]                # from edit_training_data.py
TEST_FRACTION = 0.20
SPLIT_SEED = 2026
N_FOLDS = 5
CV_REPEATS = 3
MIN_BEDS_FOR_OWN_RANGE = 40   # bedroom counts with fewer out-of-fold units use the overall range and shift
UNDER_WEIGHT = 1.25       # asymmetric loss: an underestimate costs this many times an overestimate of the same size

# Tuning grids (from test_models.ipynb, widened where the best value sat at an edge)
GRIDS = {
    "ridge": {"ridge__alpha": np.logspace(-1, 4.5, 50),
              "coreweight__core_weight": [1, 3, 10, 30, 100]},
    "svm": {"svr__C": [0.1, 0.3, 1, 3, 10, 30, 100],
            "svr__gamma": ["scale", 0.01, 0.03, 0.1, 0.3],
            "svr__epsilon": [0.01, 0.05, 0.1]},
    "knn": {"kneighborsregressor__n_neighbors": [3, 5, 7, 10, 15, 20, 30],
            "kneighborsregressor__weights": ["uniform", "distance"],
            "kneighborsregressor__p": [1, 2],
            "coreweight__core_weight": [1, 3, 10, 30]},
    "gbm": {"histgradientboostingregressor__learning_rate": [0.01, 0.02, 0.05],
            "histgradientboostingregressor__max_iter": [200, 500, 1000],
            "histgradientboostingregressor__max_leaf_nodes": [3, 5, 7, 15],
            "histgradientboostingregressor__min_samples_leaf": [5, 10, 20, 40],
            "histgradientboostingregressor__loss": ["squared_error", "absolute_error"]},
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


def _xy_km(lat, lon):
    return ((np.asarray(lon, float) - CBD_LON) * KM_PER_DEG_LON,
            (np.asarray(lat, float) - CBD_LAT) * KM_PER_DEG_LAT)


def _dist_km(lat, lon):
    return np.hypot(*_xy_km(lat, lon))


def parse_floor(unit):
    """Floor from a unit number: 'SI FL9-ID885' -> 9, '529' / 'E710' / 'N/A-0436' -> 5 / 7 / 4,
    '24F' -> 24. Anything else (2-digit units, street addresses, 'PH') -> NaN."""
    if unit is None or (isinstance(unit, float) and np.isnan(unit)):
        return np.nan
    s = str(unit).strip().upper()
    m = re.search(r"\bFL\s*(\d{1,2})\b", s)
    if m:
        f = int(m.group(1))
    else:
        s = re.sub(r"^N/A-", "", s)
        m3 = re.fullmatch(r"[A-Z]?-?(\d{3,4})[A-Z]?", s)
        m2 = re.fullmatch(r"(\d{1,2})[A-Z]", s)
        if m3:
            f = int(m3.group(1)) // 100
        elif m2:
            f = int(m2.group(1))
        else:
            return np.nan
    return float(f) if 1 <= f <= MAX_FLOOR else np.nan


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


class NearbyRents(BaseEstimator, TransformerMixin):
    """Add nearby_ppsf (average log rent per sq ft of the k nearest other buildings in the data
    this was fit on) and nearby_dist_km (their average distance). Buildings within
    SAME_BUILDING_KM are skipped, both when fitting and when predicting, so a row never sees its
    own building's rent and training rows look like new ones."""

    def __init__(self, k=NEARBY_K):
        self.k = k

    def fit(self, X, y):
        b = pd.DataFrame({"x": X["x_km"].values, "y": X["y_km"].values,
                          "p": np.asarray(y, float) - X["log_sqft"].values})
        b = b.groupby([b.x.round(3), b.y.round(3)])["p"].mean().reset_index()
        self.bx_, self.by_, self.bp_ = b.x.values, b.y.values, b.p.values
        return self

    def transform(self, X):
        dx = X["x_km"].values[:, None] - self.bx_[None, :]
        dy = X["y_km"].values[:, None] - self.by_[None, :]
        dist = np.hypot(dx, dy)
        dist[dist < SAME_BUILDING_KM] = np.inf
        k = min(self.k, dist.shape[1] - 1)
        idx = np.argpartition(dist, k, axis=1)[:, :k]
        nd = np.take_along_axis(dist, idx, axis=1)
        out = X.copy()
        out["nearby_ppsf"] = self.bp_[idx].mean(axis=1)
        out["nearby_dist_km"] = np.where(np.isfinite(nd), nd, np.nan).mean(axis=1)
        return out


class Columns(BaseEstimator, TransformerMixin):
    """Keep only the named columns (each base model gets its own list)."""

    def __init__(self, cols=()):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[list(self.cols)]


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
    step(f"rent per sq ft outside ${PPSF_RANGE[0]}-${PPSF_RANGE[1]} (typos, furnished / short-term, hotel rooms)",
         lambda x: (x.rent / x.sqft).between(*PPSF_RANGE))
    d = d.reset_index(drop=True)
    w = 1.0 / d.groupby("listing_key")["rent"].transform("size")
    d["w_building"] = w / w.groupby(d["listing_key"]).transform("sum")   # every building sums to 1
    if "source_weight" in d:   # AI live-check rows count for less (build_training_view.py)
        d["w_building"] *= pd.to_numeric(d["source_weight"], errors="coerce").fillna(1.0)
    return d, steps


def load_crime_grid(path=DEFAULT_CRIME_GRID):
    """The crime grid from edit_training_data.py as dict(lat0, lon0, step, pc1, pc2), where pc1 / pc2
    are 2-D arrays indexed [latitude row, longitude column]."""
    g = pd.read_csv(path).sort_values(["latitude", "longitude"])
    lats, lons = np.unique(g.latitude), np.unique(g.longitude)
    shape = (len(lats), len(lons))
    if len(g) != shape[0] * shape[1]:
        sys.exit(f"{path} is not a full latitude x longitude grid")
    return dict(lat0=float(lats[0]), lon0=float(lons[0]), step=float(np.diff(lats).mean()),
                **{c: g[c].values.reshape(shape).astype(np.float32) for c in CRIME_COLS})


def crime_at(grid, lat, lon):
    """Bilinear interpolation of the crime grid at (lat, lon); points off the grid use its edge."""
    n_lat, n_lon = grid["crime_pc1"].shape
    r = np.clip((float(lat) - grid["lat0"]) / grid["step"], 0, n_lat - 1)
    c = np.clip((float(lon) - grid["lon0"]) / grid["step"], 0, n_lon - 1)
    r0, c0 = min(int(r), n_lat - 2), min(int(c), n_lon - 2)
    fr, fc = r - r0, c - c0
    out = {}
    for k in CRIME_COLS:
        a = grid[k]
        out[k] = float((1 - fr) * ((1 - fc) * a[r0, c0] + fc * a[r0, c0 + 1])
                       + fr * ((1 - fc) * a[r0 + 1, c0] + fc * a[r0 + 1, c0 + 1]))
    return out


def build_features(d, crime_grid):
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
    zip_codes = {z: i for i, z in enumerate(sorted(d["zip_code"].unique()))}

    lat = d["latitude"].astype(float)
    lon = d["longitude"].astype(float)
    lat = lat.fillna(lat.groupby(d["zip_code"]).transform("mean")).fillna(lat.mean())
    lon = lon.fillna(lon.groupby(d["zip_code"]).transform("mean")).fillna(lon.mean())
    x, y = _xy_km(lat, lon)
    floor = d["unit_number"].map(parse_floor) if "unit_number" in d else pd.Series(np.nan, index=d.index)
    floor_median = float(floor.median()) if floor.notna().any() else 3.0
    ptype = d["property_type"].astype(str).str.strip().str.lower()
    units = pd.to_numeric(d.get("units_in_listing"), errors="coerce") if "units_in_listing" in d else None
    units = units if units is not None else pd.Series(np.nan, index=d.index)
    units_median = float(units.median()) if units.notna().any() else 1.0

    core = pd.DataFrame({"beds": d["beds"], "baths": d["baths"],
                         "log_sqft": np.log(d["sqft"]), "dist_km": np.hypot(x, y)})
    extra = pd.DataFrame({
        "x_km": x, "y_km": y,
        "sqft_per_bed": d["sqft"] / d["beds"].clip(lower=1),
        "floor": floor.fillna(floor_median), "floor_known": floor.notna().astype(float),
        "condo": (ptype == "condo").astype(float),
        "ptype_unknown": (~ptype.isin(["condo", "apartment"])).astype(float),
        "log_units_in_listing": np.log1p(units.fillna(units_median)),
        "is_floor_plan": (d["level"].astype(str) == "floor_plan").astype(float) if "level" in d else 0.0,
        "zip_cat": d["zip_code"].map(zip_codes).astype(float),
    }, index=d.index)
    crime = d[CRIME_COLS].astype(float)
    crime = crime.fillna(crime.mean())
    extra = extra.join(crime)

    amen = list(A.columns)
    zcols = list(Z.columns)
    model_cols = {
        "ridge": CORE_COLS + NEARBY_COLS + EXTRA_COLS + CRIME_COLS + zcols + amen,
        "svm": CORE_COLS + LOCATION_COLS + NEARBY_COLS + EXTRA_COLS + CRIME_COLS + list(AMENITY_GROUPS),
        "knn": CORE_COLS + LOCATION_COLS + NEARBY_COLS,
        "gbm": CORE_COLS + LOCATION_COLS + NEARBY_COLS + EXTRA_COLS + CRIME_COLS + amen + ["zip_cat"],
    }
    features = CORE_COLS + list(extra.columns) + zcols + amen   # what predict_rent() must supply
    data = pd.concat([core, extra, Z, A], axis=1)
    data["log_rent"] = np.log(d["rent"])
    data["rent"] = d["rent"]
    data["listing_key"] = d["listing_key"]
    data["w_building"] = d["w_building"]
    data["is_ai"] = (d["data_source"] == AI_SOURCE).values if "data_source" in d else False

    spec = dict(
        features=features,
        model_columns=model_cols,
        zip_columns=zcols,
        zip_groups=zgroups,
        zip_codes=zip_codes,
        zip_centroid={str(z): (float(a), float(b)) for z, a, b in
                      zip(lat.groupby(d["zip_code"]).mean().index, lat.groupby(d["zip_code"]).mean(),
                          lon.groupby(d["zip_code"]).mean())},
        mean_latlon=(float(lat.mean()), float(lon.mean())),
        amenities=amen,
        amenity_tokens={c: A_tokens[c] for c in amen},
        amenity_share_yes={c: round(float(shares[c]), 4) for c in amen},
        means={c: float(core[c].mean()) for c in CORE_COLS},
        floor_median=floor_median,
        units_median=units_median,
        crime_grid=crime_grid,
        support=dict(beds=[float(d.beds.min()), float(d.beds.max())],
                     baths=[float(d.baths.min()), float(d.baths.max())],
                     sqft=[float(d.sqft.min()), float(d.sqft.max())],
                     beds_counts={str(int(b)): int(n) for b, n in d["beds"].round().value_counts().sort_index().items()}),
    )
    return data, spec


# ---------------------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------------------
def base_pipelines(model_cols, weighted):
    def wt(est):   # ask for sample weights when fitting (metadata routing)
        return est.set_fit_request(sample_weight=weighted)

    def scaler():
        return StandardScaler().set_fit_request(sample_weight=False)

    gbm_cols = model_cols["gbm"]
    return {
        "ridge": make_pipeline(NearbyRents(), Columns(model_cols["ridge"]), scaler(), CoreWeight(), wt(Ridge())),
        "svm": make_pipeline(NearbyRents(), Columns(model_cols["svm"]), scaler(), wt(SVR(kernel="rbf"))),
        "knn": make_pipeline(NearbyRents(), Columns(model_cols["knn"]), scaler(), CoreWeight(),
                             KNeighborsRegressor()),
        "gbm": make_pipeline(NearbyRents(), Columns(gbm_cols), wt(HistGradientBoostingRegressor(
            early_stopping=False, random_state=0,
            categorical_features=[gbm_cols.index("zip_cat")]))),
    }


class AverageRegressor(BaseEstimator, RegressorMixin):
    """Equal-weight average of the base models' predictions of log rent. Each model is fit with the
    sample weights except KNN, which cannot take them."""

    def __init__(self, models=None):
        self.models = models

    def fit(self, X, y, sample_weight=None):
        self.models_ = {n: _fit(clone(m), X, y, None if n == "knn" or sample_weight is None
                                else pd.Series(sample_weight))
                        for n, m in self.models.items()}
        return self

    def predict_each(self, X):
        return {n: m.predict(X) for n, m in self.models_.items()}

    def predict(self, X):
        return np.mean(list(self.predict_each(X).values()), axis=0)


def _fit(model, X, y, w):
    """Fit with sample weights when there are any (w is None when every unit counts once)."""
    return model.fit(X, y) if w is None else model.fit(X, y, sample_weight=np.asarray(w))


def _rmse(a, b, w=None):
    return float(np.sqrt(mean_squared_error(a, b, sample_weight=w)))


def asym_loss(y, pred, under_weight, w=None):
    """Root of the mean asymmetric squared loss on log rent: residuals y - pred > 0 (underestimates)
    count under_weight times as much as overestimates. On the same scale as rmse_log."""
    r = np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)
    return float(np.sqrt(np.average(np.where(r > 0, under_weight, 1.0) * r ** 2, weights=w)))


def expectile(r, under_weight, iters=200):
    """The shift s that minimises the asymmetric squared loss of residuals r - s (the tau-expectile of r,
    tau = under_weight / (1 + under_weight)). under_weight = 1 gives the mean."""
    r = np.asarray(r, dtype=float)
    s = float(np.mean(r))
    for _ in range(iters):
        wt = np.where(r > s, under_weight, 1.0)
        s_new = float(np.sum(wt * r) / np.sum(wt))
        if abs(s_new - s) < 1e-12:
            break
        s = s_new
    return s


def _scores(y, pred, w, under_weight=UNDER_WEIGHT):
    return dict(rmse_log=_rmse(y, pred), rmse_log_by_building=_rmse(y, pred, w),
                rmse_usd=_rmse(np.exp(y), np.exp(pred)),
                rmse_usd_by_building=_rmse(np.exp(y), np.exp(pred), w),
                asym_log=asym_loss(y, pred, under_weight),
                asym_log_by_building=asym_loss(y, pred, under_weight, w))


def _group_folds(groups, k, seed):
    """k folds that hold out whole buildings, shuffled by seed (for repeated CV)."""
    uniq = np.array(sorted(groups.unique()))
    fold_of = dict(zip(np.random.default_rng(seed).permutation(uniq), np.arange(len(uniq)) % k))
    f = groups.map(fold_of).values
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


def repeated_cv(tuned, data, F, weighted, fit_w, under_weight=UNDER_WEIGHT):
    """CV_REPEATS x N_FOLDS building-grouped CV with the tuned settings. Returns
    (scores per model, out-of-fold predictions of the average per repeat). Scores use the original
    rows only; AI live-check rows are trained on but scored separately (scores["average"]["ai_rows"])."""
    X, y, g, wb = data[F], data["log_rent"], data["listing_key"], data["w_building"]
    preds = {n: np.zeros((CV_REPEATS, len(y))) for n in list(tuned) + ["average"]}
    for r in range(CV_REPEATS):
        for a, b in _group_folds(g, N_FOLDS, seed=100 + r):
            Xa, ya = X.iloc[a], y.iloc[a]
            wa = None if fit_w is None else fit_w.iloc[a]
            base = []
            for n, m in tuned.items():
                p = _fit(clone(m), Xa, ya, None if n == "knn" else wa).predict(X.iloc[b])   # KNN takes no weights
                preds[n][r, b] = p
                base.append(p)
            preds["average"][r, b] = np.mean(base, axis=0)
    base = ~data["is_ai"].values
    scores = {}
    for n, P in preds.items():
        per = [_scores(y[base], P[r][base], wb[base], under_weight) for r in range(CV_REPEATS)]
        scores[n] = {k: float(np.mean([s[k] for s in per])) for k in per[0]}
        scores[n]["rmse_log_sd_across_repeats"] = float(np.std([s["rmse_log"] for s in per]))
    if (~base).any():
        per = [_scores(y[~base], preds["average"][r][~base], wb[~base], under_weight) for r in range(CV_REPEATS)]
        scores["average"]["ai_rows"] = {k: float(np.mean([s[k] for s in per])) for k in per[0]}
    return scores, preds["average"]


def train(data_path, out_path, weights="building", crime_grid_path=DEFAULT_CRIME_GRID, under_weight=UNDER_WEIGHT):
    t0 = datetime.datetime.now()
    weighted = weights == "building"
    raw = pd.read_csv(data_path)
    missing = [c for c in CRIME_COLS if c not in raw]
    if missing or not os.path.exists(crime_grid_path):
        sys.exit(f"{data_path} has no {', '.join(missing) or 'crime grid'}; "
                 "run Code/edit_training_data.py first (it writes the crime columns and Data/crime_grid.csv)")
    d, steps = clean(raw)
    print(f"read {len(raw)} rows from {data_path}")
    for name, n in steps:
        print(f"  dropped {n:>5}  {name}")
    print(f"kept {len(d)} units in {d['listing_key'].nunique()} buildings")

    data, spec = build_features(d, load_crime_grid(crime_grid_path))
    F = spec["features"]
    fit_w = data["w_building"] if weighted else None
    print(f"{len(F)} columns ({len(spec['zip_columns'])} zip, {len(spec['amenities'])} amenity); "
          f"weights: every {'building' if weighted else 'unit'} counts once\n")

    # ---- training 1: tune on 80% of buildings, score on the other 20% ----
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=SPLIT_SEED)
    tr_idx, te_idx = next(gss.split(data, groups=data["listing_key"]))
    tr, te = data.iloc[tr_idx], data.iloc[te_idx]
    X_tr, y_tr, g_tr = tr[F], tr["log_rent"], tr["listing_key"]
    w_tr = None if fit_w is None else fit_w.iloc[tr_idx]
    te_all = te
    te = te[~te["is_ai"].values]          # scores on the original rows only (no-op without AI rows)
    X_te, y_te, wb_te = te[F], te["log_rent"], te["w_building"]
    print(f"training 1: train {len(tr)} units in {g_tr.nunique()} buildings; "
          f"test {len(te)} units in {te['listing_key'].nunique()} buildings"
          + (f" (+{len(te_all) - len(te)} AI rows held out, not scored)" if len(te_all) > len(te) else ""))

    scorer = make_scorer(mean_squared_error, greater_is_better=False).set_score_request(sample_weight=weighted)
    cv = GroupKFold(n_splits=N_FOLDS)
    test_results, tuned = {}, {}
    for name, pipe in base_pipelines(spec["model_columns"], weighted).items():
        search = GridSearchCV(pipe, GRIDS[name], cv=cv, scoring=scorer, n_jobs=-1)
        # with weights, KNN's search still uses them for scoring (the scorer asks for them)
        kw = {} if w_tr is None else {"sample_weight": w_tr.values}
        search.fit(X_tr, y_tr, groups=g_tr, **kw)
        pred = search.predict(X_te)
        tuned[name] = clone(search.best_estimator_)   # settings only; the fitted copy is not kept
        test_results[name] = dict(cv_rmse_log=float(np.sqrt(-search.best_score_)),
                                  **_scores(y_te, pred, wb_te, under_weight),
                                  best_params={k.split("__")[-1]: (v.item() if hasattr(v, "item") else v)
                                               for k, v in search.best_params_.items()})
        r = test_results[name]
        print(f"  {name:>5}: CV {r['cv_rmse_log']:.4f}  test {r['rmse_log']:.4f} (${r['rmse_usd']:,.0f})  "
              f"by building {r['rmse_log_by_building']:.4f}  asym {r['asym_log']:.4f}  {r['best_params']}")
    pred = AverageRegressor(tuned).fit(X_tr, y_tr, sample_weight=w_tr).predict(X_te)
    test_results["average"] = _scores(y_te, pred, wb_te, under_weight)
    r = test_results["average"]
    print(f"  average of all 4: test {r['rmse_log']:.4f} (${r['rmse_usd']:,.0f})  "
          f"by building {r['rmse_log_by_building']:.4f}  asym {r['asym_log']:.4f}")

    # ---- repeated CV over all buildings with the tuned settings ----
    print(f"\nrepeated CV ({CV_REPEATS} x {N_FOLDS}-fold by building, all {len(data)} units, tuned settings):")
    cv_results, oof = repeated_cv(tuned, data, F, weighted, fit_w, under_weight)
    for n, s in cv_results.items():
        print(f"  {n:>7}: {s['rmse_log']:.4f} +/- {s['rmse_log_sd_across_repeats']:.4f}  (${s['rmse_usd']:,.0f})  "
              f"by building {s['rmse_log_by_building']:.4f} (${s['rmse_usd_by_building']:,.0f})  "
              f"asym {s['asym_log']:.4f} (by building {s['asym_log_by_building']:.4f})")

    # out-of-building residuals of the average (original rows): exp() bias correction and the 80% range
    base = ~data["is_ai"].values
    resid = (data["log_rent"].values[None, base] - oof[:, base]).ravel()
    beds = np.tile(data["beds"].round().astype(int).values[base], CV_REPEATS)
    smearing = float(np.mean(np.exp(resid)))

    def q(r):
        return {f"p{p}": float(np.percentile(r, p)) for p in (10, 25, 50, 75, 90)}
    resid_q = q(resid)
    own_beds = [b for b in np.unique(beds) if (beds == b).sum() / CV_REPEATS >= MIN_BEDS_FOR_OWN_RANGE]
    resid_q_beds = {str(b): q(resid[beds == b]) for b in own_beds}

    # two corrections from the out-of-building residuals, in log rent on top of the average's prediction
    # (the smearing factor is applied on top of both):
    #   1. bias correction per bedroom count: the mean residual, so each bedroom count is unbiased
    #   2. asymmetry shift: one overall shift that minimises the asymmetric loss of the bias-corrected residuals
    overall_bias = float(np.mean(resid))
    bias_beds = {str(b): float(np.mean(resid[beds == b])) for b in own_beds}
    bias_per_row = np.array([bias_beds.get(str(b), overall_bias) for b in beds])
    asym_shift = expectile(resid - bias_per_row, under_weight)
    y_base, wb = data["log_rent"].values[base], data["w_building"].values[base]

    def shifted_scores(shift):
        per = [_scores(y_base, oof[r, base] + shift.reshape(CV_REPEATS, -1)[r], wb, under_weight)
               for r in range(CV_REPEATS)]
        return {k: float(np.mean([p[k] for p in per])) for k in per[0]}
    cv_results["average_bias_corrected"] = shifted_scores(bias_per_row)
    cv_results["average_bias_corrected_shifted"] = shifted_scores(bias_per_row + asym_shift)
    n_beds = {b: int((beds == b).sum() // CV_REPEATS) for b in own_beds}
    print("  bias correction by bedrooms: " + ", ".join(
        f"{b}BR {math.expm1(bias_beds[str(b)]):+.1%} ({n_beds[b]} units)" for b in own_beds)
          + f"; other {math.expm1(overall_bias):+.1%}")
    print(f"  asymmetry shift (under-weight {under_weight:g}): {math.expm1(asym_shift):+.1%} for every unit")
    for label, key in (("+ bias corr", "average_bias_corrected"), ("+ bias + asym", "average_bias_corrected_shifted")):
        s = cv_results[key]
        print(f"  {label:>13}: {s['rmse_log']:.4f}  (${s['rmse_usd']:,.0f})  by building "
              f"{s['rmse_log_by_building']:.4f}  asym {s['asym_log']:.4f} (by building "
              f"{s['asym_log_by_building']:.4f})  [corrections chosen on these residuals]")

    # ---- training 2: tuned settings, refit on every building ----
    print("\ntraining 2: refitting all 4 models on all buildings...")
    final = AverageRegressor(tuned).fit(data[F], data["log_rent"], sample_weight=fit_w)

    bundle = dict(
        model=final,
        spec=spec,
        smearing=smearing,
        log_residual_quantiles_out_of_sample=resid_q,
        log_residual_quantiles_by_beds=resid_q_beds,
        under_weight=float(under_weight),
        bias_correction=overall_bias,
        bias_correction_by_beds=bias_beds,
        asym_shift=asym_shift,
        info=dict(
            trained=datetime.date.today().isoformat(),
            data_file=os.path.relpath(data_path, REPO),
            n_rows_read=int(len(raw)), n_used=int(len(d)), n_buildings=int(d["listing_key"].nunique()),
            cleaning_steps=steps,
            weights=weights,
            data_sources={str(k): int(v) for k, v in
                          (d["data_source"].value_counts().items() if "data_source" in d else [("realtyapi_download", len(d))])},
            ai_rows_weight=(float(pd.to_numeric(d.loc[d["data_source"] == AI_SOURCE, "source_weight"]).mean())
                            if "data_source" in d and (d["data_source"] == AI_SOURCE).any() else None),
            target="log(advertised asking monthly rent, USD)",
            test=dict(split=f"{int(TEST_FRACTION * 100)}% of buildings held out (seed {SPLIT_SEED}); "
                            f"tuned by {N_FOLDS}-fold building-grouped CV on the rest",
                      results=test_results),
            repeated_cv=dict(split=f"{CV_REPEATS} x {N_FOLDS}-fold building-grouped CV on all buildings "
                                   f"with the settings tuned in the test run", results=cv_results),
            blend="equal-weight average of log rent from ridge, svm, knn and gbm",
            loss=f"asymmetric squared loss on log rent, underestimates x{under_weight:g}; applied as a per-bedroom "
                 f"bias correction (mean out-of-building residual) plus one overall shift (tau-expectile of the "
                 f"bias-corrected residuals)",
            sklearn_version=sklearn.__version__,
            numpy_version=np.__version__,
            train_seconds=round((datetime.datetime.now() - t0).total_seconds()),
        ),
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    joblib.dump(bundle, out_path, compress=("xz", 9))
    size_kb = os.path.getsize(out_path) / 1024
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


def model_info(model_path=DEFAULT_OUT):
    """What the app shows about the model: units and buildings trained on, a one-line reliability
    note, and the amenities it prices (with the share of training buildings that have each)."""
    b = load_model(model_path)
    info, spec = b["info"], b["spec"]
    cv = info["repeated_cv"]["results"]["average"]
    reliability = ("average of ridge, SVM, KNN and gradient boosting; on buildings it had not seen, a typical "
                   "error of about {:.0%} (RMSE ${:,.0f})".format(math.exp(cv["rmse_log"]) - 1, cv["rmse_usd"]))
    if b.get("under_weight", 1.0) != 1.0:
        reliability += ("; estimates lean high by about {:+.1%} because underestimates count {:g}x as much as "
                        "overestimates".format(math.expm1(b["asym_shift"]), b["under_weight"]))
    return {"n_used": info["n_used"], "n_buildings": info["n_buildings"], "trained": info["trained"],
            "reliability": reliability, "amenity_share_yes": dict(spec["amenity_share_yes"])}


def predict_rent(bedrooms=None, bathrooms=None, sqft=None, zip_code=None, amenities=None,
                 latitude=None, longitude=None, floor=None, property_type=None, units_in_listing=None,
                 model_path=DEFAULT_OUT):
    """Return {"estimate", "low", "high", "reliability", "warnings"}, the keys the old rent_model.predict_rent used.

    amenities: a dict like {"in_unit_laundry": True, "gym": None} (None = not known) or a list of
    amenity names the building has. Unknown amenities get the share of buildings that have them.
    floor, property_type ("Apartment" / "Condo") and units_in_listing are optional.
    low/high are the 10th/90th percentile of errors on buildings the model had not seen, for units
    with the same number of bedrooms when there were enough of them.
    The estimate includes the bias correction and asymmetric-loss shift (see the module docstring);
    low/high do not.
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
    if latitude is None or longitude is None:
        if z in spec["zip_centroid"]:
            latitude, longitude = spec["zip_centroid"][z]
            warnings.append("no coordinates given; using the zip's average location")
        else:
            latitude, longitude = spec["mean_latlon"]
            warnings.append("no location given; using the average location of training buildings")
    x, y = _xy_km(latitude, longitude)
    row = {"beds": bedrooms, "baths": bathrooms, "log_sqft": math.log(sqft) if sqft else None,
           "dist_km": float(np.hypot(x, y))}
    for k in CORE_COLS:
        if row[k] is None:
            row[k] = spec["means"][k]
            warnings.append(f"{k} not given; using the training average")
    pt = str(property_type).strip().lower() if property_type else ""
    row.update({
        "x_km": float(x), "y_km": float(y),
        "sqft_per_bed": math.exp(row["log_sqft"]) / max(row["beds"], 1),
        "floor": float(floor) if floor is not None else spec["floor_median"],
        "floor_known": float(floor is not None),
        "condo": float(pt == "condo"), "ptype_unknown": float(pt not in ("condo", "apartment")),
        "log_units_in_listing": math.log1p(units_in_listing if units_in_listing else spec["units_median"]),
        "is_floor_plan": 0.0,
        "zip_cat": float(spec["zip_codes"][z]) if z in spec["zip_codes"] else np.nan,
        **crime_at(spec["crime_grid"], latitude, longitude),
    })

    for c in spec["zip_columns"]:
        row[c] = 0.0
    if z is None:
        warnings.append("no zip code given; no zip effect applied")
    elif "zip_" + z in spec["zip_columns"]:
        row["zip_" + z] = 1.0
    elif z in spec["zip_codes"] and "zip_other" in spec["zip_columns"]:
        row["zip_other"] = 1.0
    else:
        warnings.append("zip code is not in the training data; no zip effect applied")

    if isinstance(amenities, dict):
        flags = {k: (None if v is None else bool(v)) for k, v in amenities.items()}
    elif amenities is not None:   # a list of names: anything on it is yes, everything else no
        have = {_tok(a) for a in amenities}
        flags = {c: (c in {h.replace(" ", "_") for h in have} or bool(set(spec["amenity_tokens"][c]) & have))
                 for c in spec["amenities"]}
    else:
        flags = {}
    for c in spec["amenities"]:
        v = flags.get(c)
        row[c] = spec["amenity_share_yes"][c] if v is None else float(v)
    unused = [k for k in flags if k not in spec["amenities"]]
    if unused:
        warnings.append("not used by this model: " + ", ".join(sorted(unused)))

    X = pd.DataFrame([row], columns=spec["features"])
    mean_est = math.exp(float(b["model"].predict(X)[0])) * b["smearing"]
    nbeds = str(int(round(row["beds"])))
    q = b["log_residual_quantiles_by_beds"].get(nbeds, b["log_residual_quantiles_out_of_sample"])
    # per-bedroom bias correction plus the asymmetric-loss shift (models saved before this have neither)
    shift = b.get("bias_correction_by_beds", {}).get(nbeds, b.get("bias_correction", 0.0)) + b.get("asym_shift", 0.0)
    est = mean_est * math.exp(shift)
    return {"estimate": round(est), "low": round(mean_est * math.exp(q["p10"])),   # range is not shifted
            "high": round(mean_est * math.exp(q["p90"])), "reliability": model_info(model_path)["reliability"],
            "warnings": warnings}


def main():
    ap = argparse.ArgumentParser(description="Train the SF rent model (average of 4 models)")
    ap.add_argument("--data", default=DEFAULT_DATA, help="unit-level training CSV (default %(default)s)")
    ap.add_argument("--crime-grid", default=DEFAULT_CRIME_GRID, help="crime grid CSV (default %(default)s)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="where to save the model (default %(default)s)")
    ap.add_argument("--weights", choices=["building", "unit"], default="building",
                    help="count every building once (default) or every unit once")
    ap.add_argument("--under-weight", type=float, default=UNDER_WEIGHT,
                    help="asymmetric loss: cost of an underestimate relative to an overestimate "
                         "(default %(default)s; 1 = plain mean estimate)")
    ap.add_argument("--predict", action="store_true", help="skip training; load the saved model and run a sample")
    a = ap.parse_args()
    if not a.predict:
        if a.under_weight <= 0:
            sys.exit("--under-weight must be positive")
        train(a.data, a.out, a.weights, a.crime_grid, a.under_weight)
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
