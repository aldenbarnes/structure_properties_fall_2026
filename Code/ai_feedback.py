#!/usr/bin/env python3
"""Turn recurring AI live-check findings into model adjustments, with evidence gates.

    python Code/ai_feedback.py               # findings report + learned adjustments
    python Code/ai_feedback.py --compare     # also: would adding the AI rows improve the SF model?
    python Code/ai_feedback.py --retrain     # also: retrain on base + AI rows and switch to it if it wins

An adjustment needs two independent signals, so neither the AI's opinion alone nor noise in a few
listings can move prices:
  1. The AI keeps raising it: the same factor (and feature) is a driver in at least MIN_CHECKS
     different disagreement checks in the last WINDOW_DAYS, pointing the same way at least
     MIN_AGREEMENT of the time.
  2. The listings it read back it up: on the AI training rows (the comps, not the AI's words) the
     effect is measured in code and differs from what the tool currently assumes with |t| >= MIN_T.
     Buildings, not units, are the observations (a 40-unit complex is one data point).

What each factor changes when both gates pass
  concessions     The SF model is trained on asking rents, so free rent is not something it can
                  learn. Learned per zip (citywide when a zip has under MIN_ZIP_BUILDINGS buildings):
                  share of buildings with a special, average free weeks and the discount on
                  effective rent. The app shows an expected effective rent next to the estimate.
  mandatory_fees  Same, for required monthly fees (median per building, by zip).
  amenities /     For features priced from published studies (feature_adjustments.py), the premium
  condition       is recalibrated: the study value is the prior and the comps' measured premium
                  (vs the SF model's prediction) the new evidence, combined by precision. Shown in
                  the app with "AI live-check findings" as an added source.
                  For amenities the SF model itself prices, the fix is more data: --compare and
                  --retrain add the labeled AI rows to training.

Learned adjustments are rebuilt from scratch on every run, so one that stops passing the gates
drops out: the tool moves back as well as forward. Every run is logged to
Data/ai_audit/model_adjustments.jsonl.

--compare uses the current model's tuned settings (no new tuning) and the same building folds for
both versions, and scores only original rows, so the only difference is the added AI rows.
--retrain switches models only if the version with AI rows is no worse on original rows (within
MAX_BASE_WORSENING) and better on the AI rows; the old model is kept as rent_blend_previous.joblib.
"""
import argparse
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ai_audit_log as al            # noqa: E402
import build_training_view as btv    # noqa: E402
import create_model_test as rm       # noqa: E402
import feature_adjustments as fa     # noqa: E402

WINDOW_DAYS = 90
MIN_CHECKS = 8
MIN_AGREEMENT = 0.65
MIN_T = 2.0
MIN_BUILDINGS = 8             # buildings needed on each side of a yes/no comparison
MIN_ZIP_BUILDINGS = 4
MAX_BASE_WORSENING = 0.01     # --retrain: up to 1% worse RMSE on original rows is tolerated
COMPARE_REPEATS = 2
AI_SOURCE_ENTRY = {"name": "AI live-check findings", "year": None,
                   "title": "Comps read by the AI live check (Data/ai_audit), measured against the SF model",
                   "url": None}
UNDER = {"missing", "underweighted"}


def _now():
    return datetime.now(timezone.utc)


def _recent(records, days, key="checked_at"):
    cut = _now() - timedelta(days=days)
    out = []
    for r in records:
        try:
            if datetime.fromisoformat(r[key]) >= cut:
                out.append(r)
        except (KeyError, ValueError):
            pass
    return out


# ---------------------------------------------------------------- signal 1: what the AI keeps saying
def tally_findings(disagreements):
    """{(factor, feature): {"checks", "assessment": Counter, "direction": Counter}}, one vote per check."""
    out = defaultdict(lambda: {"checks": 0, "assessment": Counter(), "direction": Counter(), "check_ids": []})
    for chk in disagreements:
        seen = {}
        for d in chk.get("drivers") or []:
            seen.setdefault((d["factor"], d.get("feature")), d)
        for key, d in seen.items():
            t = out[key]
            t["checks"] += 1
            t["assessment"]["under" if d["assessment"] in UNDER else "over"] += 1
            t["direction"][d.get("direction")] += 1
            t["check_ids"].append(chk["check_id"])
    return dict(out)


def ai_signal(tally, factor, feature=None):
    t = tally.get((factor, feature))
    if not t:
        return {"checks": 0, "passes": False}
    side, n = t["assessment"].most_common(1)[0]
    share = n / t["checks"]
    return {"checks": t["checks"], "majority": side, "agreement": round(share, 2),
            "passes": t["checks"] >= MIN_CHECKS and share >= MIN_AGREEMENT}


# ---------------------------------------------------------------- signal 2: what the comps show
def ai_rows_frame(days=WINDOW_DAYS):
    rows = al.read_ai_rows()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["checked"] = pd.to_datetime(df["ai_checked_at"], errors="coerce", utc=True)
    df = df[df["checked"] >= pd.Timestamp(_now() - timedelta(days=days))]
    for c in ("rent", "beds", "baths", "sqft", "latitude", "longitude", "concession_weeks",
              "mandatory_monthly_fees", "lease_term_months", "units_in_listing"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["unit_key_"] = df["unit_key"].fillna("") + "|" + df["floor_plan"].fillna("") + "|" + df["sqft"].astype(str)
    df = df.sort_values("checked", ascending=False).drop_duplicates(["listing_key", "unit_key_"])
    return df.reset_index(drop=True)


def add_residuals(df, model_path=rm.DEFAULT_OUT):
    """log(asking rent / SF model prediction) for every AI row; the model sees no study adjustments."""
    res = []
    for r in df.itertuples():
        try:
            pred = rm.predict_rent(
                bedrooms=r.beds, bathrooms=r.baths, sqft=r.sqft, zip_code=str(r.zip_code)[:5],
                latitude=None if pd.isna(r.latitude) else r.latitude,
                longitude=None if pd.isna(r.longitude) else r.longitude,
                amenities=([a for a in str(r.amenities).split(";") if a.strip()] or None)
                if isinstance(r.amenities, str) else None,
                floor=None if pd.isna(rm.parse_floor(r.unit_number)) else rm.parse_floor(r.unit_number),
                property_type=r.property_type if r.property_type in ("Condo", "Apartment") else None,
                units_in_listing=None if pd.isna(r.units_in_listing) else int(r.units_in_listing),
                model_path=model_path)["estimate"]
            res.append(math.log(r.rent / pred))
        except Exception:
            res.append(np.nan)
    df = df.copy()
    df["resid"] = res
    return df


def _mean_t(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return None, None, None
    m, se = float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))
    return m, se, (m / se if se > 0 else (math.inf if m else 0.0))


def market_findings(df):
    """Concessions and required fees by building, citywide and by zip."""
    if df.empty:
        return {}
    term = df["lease_term_months"].fillna(12).clip(lower=1)
    df = df.assign(free_share=df["concession_weeks"].fillna(0) / (term * 52 / 12),
                   zip5=df["zip_code"].astype(str).str[:5])
    b = df.groupby("listing_key").agg(zip5=("zip5", "first"), free_share=("free_share", "mean"),
                                      weeks=("concession_weeks", "mean"), fees=("mandatory_monthly_fees", "median"))

    def summarize(g):
        m, se, t = _mean_t(g["free_share"])
        fm, fse, ft = _mean_t(g["fees"])
        return {"buildings": int(len(g)), "share_with_concession": round(float((g["weeks"] > 0).mean()), 3),
                "avg_free_weeks_when_offered": round(float(g.loc[g["weeks"] > 0, "weeks"].mean()), 2)
                if (g["weeks"] > 0).any() else 0.0,
                "effective_discount_pct": round(100 * m, 2) if m is not None else None,
                "discount_t": round(t, 2) if t is not None else None,
                "median_required_fees": round(float(g["fees"].median()), 2),
                "fees_t": round(ft, 2) if ft is not None else None}
    out = {"citywide": summarize(b), "by_zip": {}}
    for z, g in b.groupby("zip5"):
        if len(g) >= MIN_ZIP_BUILDINGS:
            out["by_zip"][z] = summarize(g)
    return out


def _model_priced_features():
    spec = rm.load_model()["spec"]
    toks = {t for v in spec["amenity_tokens"].values() for t in v}
    return {f for f, names in fa.AMENITY_NAMES.items() if any(rm._tok(n) in toks for n in names)}


def premium_findings(df):
    """Measured premium (log points vs the SF model) for study-priced features and condition levels."""
    if df.empty or "resid" not in df:
        return {}
    df = df[df["resid"].notna()]
    b = df.groupby("listing_key").agg(resid=("resid", "mean"), amenities=("amenities", "first"),
                                      condition=("condition", "first"))
    out = {}
    have = b["amenities"].fillna("").map(lambda s: {fa._norm(t) for t in s.split(";") if t.strip()})
    model_priced = _model_priced_features()
    for f, names in fa.AMENITY_NAMES.items():
        x = have.map(lambda s, names=names: any(fa._norm(n) in s for n in names))
        out[("feature", f)] = _compare(b["resid"], x, model_priced=f in model_priced)
    known = b[b["condition"].isin(fa.SELECTS["condition"][1])]
    for level in fa.SELECTS["condition"][1]:
        if level == fa.BASELINE["condition"]:
            continue
        sub = known[known["condition"].isin([level, fa.BASELINE["condition"]])]
        out[("condition", level)] = _compare(sub["resid"], sub["condition"] == level)
    return out


def _compare(resid, flag, model_priced=False):
    yes, no = resid[flag], resid[~flag]
    r = {"buildings_with": int(len(yes)), "buildings_without": int(len(no)), "model_priced": model_priced}
    if len(yes) < MIN_BUILDINGS or len(no) < MIN_BUILDINGS:
        return dict(r, measurable=False)
    diff = float(yes.mean() - no.mean())
    se = math.sqrt(yes.var(ddof=1) / len(yes) + no.var(ddof=1) / len(no))
    return dict(r, measurable=True, log_premium=round(diff, 4), se=round(se, 4))


# ---------------------------------------------------------------- both gates -> learned adjustments
def decide(tally, market, premiums):
    learned = {"generated_at": _now().isoformat(timespec="seconds"), "source": AI_SOURCE_ENTRY,
               "window_days": WINDOW_DAYS, "feature": {}, "condition": {}, "market": {}}
    report = []

    # market: concessions and required fees
    for factor, tkey, field in (("concessions", "discount_t", "effective_discount_pct"),
                                ("mandatory_fees", "fees_t", "median_required_fees")):
        sig = ai_signal(tally, factor)
        city = (market or {}).get("citywide") or {}
        t = city.get(tkey)
        data_ok = t is not None and abs(t) >= MIN_T and (city.get(field) or 0) > 0
        ok = sig["passes"] and sig.get("majority") == "under" and data_ok
        report.append({"factor": factor, "feature": None, "ai": sig, "data": {k: city.get(k) for k in (field, tkey, "buildings")},
                       "applied": ok})
        if ok:
            learned["market"][factor] = {"citywide": city, "by_zip": market["by_zip"],
                                         "checks_flagging": sig["checks"]}

    # premiums for study-priced features and condition
    for (kind, name), m in premiums.items():
        sig = ai_signal(tally, "amenities" if kind == "feature" else "condition", name)
        row = {"factor": "amenities" if kind == "feature" else "condition", "feature": name, "ai": sig,
               "data": m, "applied": False}
        study = fa.EFFECTS.get((kind, name))
        if m.get("model_priced"):
            row["action"] = "priced by the SF model: add AI rows to training (--compare / --retrain)"
        elif study and m.get("measurable") and sig["passes"]:
            current = math.log1p(study["value"] / 100) if study["unit"] == "pct" else None
            if current is not None:
                diff = m["log_premium"] - current
                t = diff / m["se"] if m["se"] > 0 else 0.0
                data_side = "under" if (abs(m["log_premium"]) > abs(current) and
                                        (current == 0 or np.sign(m["log_premium"]) == np.sign(current))) else "over"
                row["data"] = dict(m, current_log=round(current, 4), t_vs_current=round(t, 2), data_side=data_side)
                if abs(t) >= MIN_T and data_side == sig["majority"]:
                    prior_sd = max((study["high"] - study["low"]) / 4, 0.5)
                    data_pct, data_sd = math.expm1(m["log_premium"]) * 100, m["se"] * 100
                    w0, w1 = 1 / prior_sd ** 2, 1 / data_sd ** 2
                    value = (study["value"] * w0 + data_pct * w1) / (w0 + w1)
                    sd = math.sqrt(1 / (w0 + w1))
                    learned[kind][name] = {
                        "value": round(value, 2), "low": round(value - 1.64 * sd, 2), "high": round(value + 1.64 * sd, 2),
                        "study_value": study["value"], "measured_value": round(data_pct, 2),
                        "measured_se": round(data_sd, 2), "buildings_with": m["buildings_with"],
                        "buildings_without": m["buildings_without"], "checks_flagging": sig["checks"],
                        "ai_assessment": sig["majority"]}
                    row["applied"] = True
        report.append(row)
    return learned, report


# ---------------------------------------------------------------- does adding AI rows help the SF model?
def compare(view_path, repeats=COMPARE_REPEATS, model_path=rm.DEFAULT_OUT):
    bundle = rm.load_model(model_path)
    d, _ = rm.clean(pd.read_csv(view_path))
    data, spec = rm.build_features(d, rm.load_crime_grid())
    F = spec["features"]
    is_ai = data["is_ai"].values
    if not is_ai.any():
        return {"skipped": "no AI rows in the training view"}
    old = bundle["model"].models
    tuned = rm.base_pipelines(spec["model_columns"], weighted=True)
    for n in tuned:
        p = old[n].get_params()
        tuned[n].set_params(**{k: p[k] for k in rm.GRIDS[n] if k in p})
    X, y, w, g = data[F], data["log_rent"], data["w_building"], data["listing_key"]
    pa, pb = np.full((repeats, len(y)), np.nan), np.full((repeats, len(y)), np.nan)
    for r in range(repeats):
        for tr, te in rm._group_folds(g, rm.N_FOLDS, seed=500 + r):
            tr_base = tr[~is_ai[tr]]
            pa[r, te] = rm.AverageRegressor(tuned).fit(X.iloc[tr_base], y.iloc[tr_base], w.iloc[tr_base]).predict(X.iloc[te])
            pb[r, te] = rm.AverageRegressor(tuned).fit(X.iloc[tr], y.iloc[tr], w.iloc[tr]).predict(X.iloc[te])

    def rmse(P, mask):
        return float(np.mean([math.sqrt(np.mean((y.values[mask] - P[r][mask]) ** 2)) for r in range(repeats)]))
    base = ~is_ai
    out = {"rows": {"original": int(base.sum()), "ai": int(is_ai.sum())},
           "without_ai": {"rmse_log_original": rmse(pa, base), "rmse_log_ai_rows": rmse(pa, is_ai)},
           "with_ai": {"rmse_log_original": rmse(pb, base), "rmse_log_ai_rows": rmse(pb, is_ai)},
           "repeats": repeats, "folds": rm.N_FOLDS}
    out["passes"] = (out["with_ai"]["rmse_log_original"] <= out["without_ai"]["rmse_log_original"] * (1 + MAX_BASE_WORSENING)
                     and out["with_ai"]["rmse_log_ai_rows"] < out["without_ai"]["rmse_log_ai_rows"])
    return out


def retrain(view_path, cmp):
    if not cmp.get("passes"):
        return {"switched": False, "reason": "the comparison did not pass; the current model stays"}
    cand = rm.DEFAULT_OUT.replace(".joblib", "_candidate.joblib")
    rm.train(str(view_path), cand)
    prev = rm.DEFAULT_OUT.replace(".joblib", "_previous.joblib")
    shutil.copyfile(rm.DEFAULT_OUT, prev)
    shutil.move(cand, rm.DEFAULT_OUT)
    return {"switched": True, "previous_model": prev, "training_file": str(view_path)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Learn from recurring AI live-check findings")
    ap.add_argument("--compare", action="store_true", help="test whether the labeled AI rows improve the SF model")
    ap.add_argument("--retrain", action="store_true", help="retrain with the AI rows and switch if the comparison passes")
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    a = ap.parse_args(argv)

    disagreements = _recent(al.read_jsonl(al.DISAGREEMENT_LOG), WINDOW_DAYS)
    checks = _recent(al.read_jsonl(al.CRITIQUE_LOG), WINDOW_DAYS)
    tally = tally_findings(disagreements)
    df = ai_rows_frame()
    if not df.empty:
        df = add_residuals(df)
    market = market_findings(df)
    learned, report = decide(tally, market, premium_findings(df))
    run = {"run_at": _now().isoformat(timespec="seconds"), "window_days": WINDOW_DAYS,
           "checks": len(checks), "disagreements": len(disagreements), "ai_rows": int(len(df)),
           "verdicts": dict(Counter(c["verdict"] for c in checks)),
           "gates": {"min_checks": MIN_CHECKS, "min_agreement": MIN_AGREEMENT, "min_t": MIN_T,
                     "min_buildings": MIN_BUILDINGS},
           "findings": [r for r in report if r["ai"]["checks"] or r["applied"]],
           "applied": [(r["factor"], r["feature"]) for r in report if r["applied"]]}

    print("{} checks in the last {} days, {} disagreements, {} AI rows".format(
        len(checks), WINDOW_DAYS, len(disagreements), len(df)))
    for r in run["findings"]:
        print("  {:<15} {:<26} AI: {} checks, {} ({:.0%})  -> {}".format(
            r["factor"], str(r["feature"] or ""), r["ai"]["checks"], r["ai"].get("majority"),
            r["ai"].get("agreement", 0), "APPLIED" if r["applied"] else r.get("action", "not applied")))

    view = None
    if a.compare or a.retrain:
        view, counts = btv.main([])
        run["training_view"] = {"file": str(view), "counts": counts}
        run["compare"] = compare(view)
        print("compare:", json.dumps(run["compare"], indent=1))
        if a.retrain and not a.dry_run:
            run["retrain"] = retrain(view, run["compare"])
            print("retrain:", run["retrain"])
    if not a.dry_run:
        al.write_learned_adjustments(learned)
        al.append_jsonl(al.ADJUSTMENTS_LOG, [run], "AI feedback run")
        print("wrote {} ({} adjustment(s) in force) and logged the run".format(
            al.LEARNED_ADJUSTMENTS, len(run["applied"])))
    return run, learned


if __name__ == "__main__":
    main()
