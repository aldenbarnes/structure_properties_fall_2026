#!/usr/bin/env python3
"""Build a training file that adds the AI live-check rows to the original training data, labeled.

    python Code/build_training_view.py                        # base + every eligible AI row
    python Code/build_training_view.py --disagreements-only   # only comps from Review / Disagrees checks
    python Code/build_training_view.py --ai-weight 0.25
    python Code/create_model_test.py --data Data/training_views/training_with_ai.csv

Reads (never writes) Data/realty_training_combined_final.csv (the file the model trains on, with
the crime columns from edit_training_data.py) and Data/ai_audit/ai_training_rows.csv. AI rows get
crime_pc1 / crime_pc2 from Data/crime_grid.csv at their coordinates, the same lookup the model
uses for a new address.
Writes Data/training_views/training_with_ai.csv and a manifest next to it. The base file's
checksum is taken before and after, and the run fails if it changed.

Every row gets
  data_source     "realtyapi_download" (base) or "ai_audit"
  source_weight   1.0 for base rows, --ai-weight (default 0.5) for AI rows; create_model_test
                  multiplies its building weights by this, so AI rows count for less
and AI rows keep all their labels (ai_source_type, ai_check_id, LLM models, prompt version, ...).

Which AI rows are used
  - comparability High or Medium (--min-comparability), and every source label present;
  - checked within --max-age-days (default 180);
  - one row per unit: the newest check wins;
  - a unit already in the base data is left out: the base row wins, so AI rows only add units
    the downloads don't have and never change or duplicate a base row.
  - by default comps from Confirmed checks are used too, not only disagreements, so the added
    data is not drawn only from the places the model got wrong (which would push it too far).
"""
import argparse
import hashlib
import json
from datetime import date, datetime
from pathlib import Path

import ai_audit_log as al
import create_model_test as rm
import download_training as dt

BASE_PATH = Path(rm.DEFAULT_DATA)
OUT_PATH = al.TRAINING_VIEWS / "training_with_ai.csv"
COMPARABILITY_RANK = {"High": 2, "Medium": 1, "Low": 0}
VIEW_COLUMNS = dt.COMBINED_COLUMNS + rm.CRIME_COLS + al.AI_FIELD_COLUMNS + al.PROVENANCE_COLUMNS + ["source_weight"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_rows(base_rows, ai_rows, ai_weight=0.5, include_confirmed=True, max_age_days=180,
               min_comparability="Medium", today=None, half_life=dt.HALF_LIFE_DAYS, crime_grid=None):
    """Return (rows, counts). Pure function: no files."""
    today = today or date.today()
    counts = {"base_rows": len(base_rows), "ai_rows_read": len(ai_rows)}
    types = {"ai_disagreement", "ai_insufficient_evidence"} | ({"ai_confirmation"} if include_confirmed else set())
    out = []
    for r in base_rows:
        r = dict(r)
        r["data_source"], r["source_weight"] = al.SOURCE_BASE, 1.0
        out.append(r)
    base_keys = {dt._row_key(r) for r in base_rows}

    def skip(reason):
        counts["ai_skipped_" + reason] = counts.get("ai_skipped_" + reason, 0) + 1

    kept = {}
    for r in sorted(ai_rows, key=lambda r: r.get("ai_checked_at") or "", reverse=True):
        if r.get("data_source") != al.SOURCE_AI or any(r.get(c) in (None, "") for c in al.PROVENANCE_COLUMNS):
            skip("missing_labels"); continue
        if r["ai_source_type"] not in types:
            skip("source_type"); continue
        if COMPARABILITY_RANK.get(r.get("comparability"), 0) < COMPARABILITY_RANK[min_comparability]:
            skip("low_comparability"); continue
        try:
            age = (today - datetime.fromisoformat(r["ai_checked_at"]).date()).days
        except ValueError:
            skip("bad_date"); continue
        if age > max_age_days:
            skip("too_old"); continue
        k = dt._row_key(r)
        if k in base_keys:
            skip("unit_in_base_data"); continue
        if k in kept:
            skip("older_duplicate"); continue
        kept[k] = r
    for r in kept.values():
        r = dict(r)
        days = (today - date.fromisoformat(r["date_downloaded"])).days
        r["days_since_download"] = days
        r["recency_weight"] = round(0.5 ** (max(days, 0) / half_life), 6)
        r["source_weight"] = ai_weight
        if crime_grid is not None:
            try:
                r.update(rm.crime_at(crime_grid, float(r["latitude"]), float(r["longitude"])))
            except (TypeError, ValueError, KeyError):
                skip("no_coordinates"); continue
        out.append(r)
    counts["ai_rows_used"] = len(out) - len(base_rows)
    by_type = {}
    for r in out[len(base_rows):]:
        by_type[r["ai_source_type"]] = by_type.get(r["ai_source_type"], 0) + 1
    counts["ai_rows_used_by_type"] = by_type
    counts["rows_written"] = len(out)
    return out, counts


def main(argv=None):
    ap = argparse.ArgumentParser(description="Base training data + labeled AI live-check rows, in a separate file")
    ap.add_argument("--ai-weight", type=float, default=0.5)
    ap.add_argument("--disagreements-only", action="store_true")
    ap.add_argument("--max-age-days", type=int, default=180)
    ap.add_argument("--min-comparability", choices=["High", "Medium"], default="Medium")
    ap.add_argument("--base", default=str(BASE_PATH))
    ap.add_argument("--ai", default=None, help="default Data/ai_audit/ai_training_rows.csv")
    ap.add_argument("--out", default=None, help="default Data/training_views/training_with_ai.csv")
    a = ap.parse_args(argv)
    base_path, out_path = Path(a.base), Path(a.out or al.TRAINING_VIEWS / "training_with_ai.csv")
    a.ai = a.ai or str(al.AI_ROWS)
    al.assert_not_base(out_path)
    before = sha256(base_path)
    base_rows = dt.read_csv(base_path)
    ai_rows = al.read_ai_rows(a.ai)
    rows, counts = build_rows(base_rows, ai_rows, a.ai_weight, not a.disagreements_only,
                              a.max_age_days, a.min_comparability, crime_grid=rm.load_crime_grid())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dt.write_csv(out_path, VIEW_COLUMNS, rows)
    if sha256(base_path) != before:
        raise SystemExit("the base training file changed during the build; nothing should write to it")
    manifest = {"built": datetime.now().isoformat(timespec="seconds"), "output": str(out_path),
                "base_file": str(base_path), "base_sha256": before, "ai_file": str(a.ai),
                "settings": {"ai_weight": a.ai_weight, "disagreements_only": a.disagreements_only,
                             "max_age_days": a.max_age_days, "min_comparability": a.min_comparability},
                "counts": counts}
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))
    print("wrote {} ({} rows); base file unchanged".format(out_path, len(rows)))
    return out_path, counts


if __name__ == "__main__":
    main()
