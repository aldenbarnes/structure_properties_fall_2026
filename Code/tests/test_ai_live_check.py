"""Tests for the AI live check, its labeled logs and the feedback gates.

    pip install pytest
    python -m pytest Code/tests -q

Uses the saved RealtyAPI responses in Data/training/raw as listings and a rule-based stand-in for
the LLM, and writes only to a temporary folder.
"""
import gzip
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import ai_audit_log as al          # noqa: E402
import ai_feedback as fb           # noqa: E402
import build_training_view as btv  # noqa: E402
import download_training as dt     # noqa: E402
import feature_adjustments as fa   # noqa: E402
import live_check as lc            # noqa: E402

RAW = sorted((CODE.parent / "Data" / "training" / "raw").glob("byid_*.jsonl.gz"))
NOW = datetime(2026, 11, 2, 18, 0, tzinfo=timezone.utc)
SUBJECT = dict(beds=1, baths=1, sqft=650, zip_code="94107", latitude=37.7785, longitude=-122.3950,
               listing_key=None, address="test unit", amenities=[], condition="Updated")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class FakeClient:
    """Serves the saved RealtyAPI responses, like realty_lookup.Client."""

    def __init__(self):
        self.recs = {}
        for f in RAW:
            for line in gzip.open(f, "rt", encoding="utf-8"):
                r = json.loads(line)
                self.recs[r["listing_key"]] = r
        self.requests = 0

    def get(self, endpoint, params):
        self.requests += 1
        if endpoint == "/search/bylocation":
            return {"searchResults": [r["search_result"] for r in self.recs.values()]}
        return self.recs[params["listingKey"]]["details"]


@pytest.fixture
def audit_dir(tmp_path, monkeypatch):
    d = tmp_path / "Data" / "ai_audit"
    for name, val in dict(AI_DIR=d, CRITIQUE_LOG=d / "critique_log.jsonl", DISAGREEMENT_LOG=d / "disagreement_log.jsonl",
                          DECISIONS_LOG=d / "decisions.jsonl", AI_ROWS=d / "ai_training_rows.csv",
                          ADJUSTMENTS_LOG=d / "model_adjustments.jsonl",
                          LEARNED_ADJUSTMENTS=d / "learned_adjustments.json",
                          TRAINING_VIEWS=tmp_path / "Data" / "training_views").items():
        monkeypatch.setattr(al, name, val)
    monkeypatch.setattr(al, "_GITHUB", None)
    return d


def run(estimate=4800, llm=None, **subject):
    s = dict(SUBJECT, **subject)
    return lc.run_check(FakeClient(), llm or lc.RuleBasedLLM(), s,
                        {"model": estimate, "final": estimate, "low": estimate * .8, "high": estimate * 1.2},
                        "2026-11-02", now=NOW)


# ---------------------------------------------------------------- base data protection
def test_base_training_data_is_never_writable(audit_dir):
    for p in [al.DATA / "training" / "x.csv", al.DATA / "training" / "raw" / "y.jsonl",
              al.DATA / "realty_training_combined.csv", al.DATA / "realty_training.csv", al.DATA / "elsewhere.csv"]:
        with pytest.raises(PermissionError):
            al.assert_not_base(p)
    al.assert_not_base(audit_dir / "critique_log.jsonl")     # allowed


def test_logging_leaves_base_files_untouched_and_labels_every_row(audit_dir):
    import create_model_test as rm
    base = [dt.COMBINED_PATH, rm.DEFAULT_DATA] + list(dt.TRAINING_DIR.glob("*.csv"))
    before = {p: sha(p) for p in base}
    chk = run(estimate=4800)
    assert chk["verdict"] in ("confirmed", "review", "disagree")
    assert al.record_check(chk) == []
    assert {p: sha(p) for p in base} == before
    rows = al.read_ai_rows()
    assert rows and all(r["data_source"] == "ai_audit" for r in rows)
    assert all(r["ai_source_type"] == al.SOURCE_TYPE[chk["verdict"]] for r in rows)
    assert all(r["ai_check_id"] == chk["check_id"] and r["llm_extraction_model"] == "rule-based-test" for r in rows)
    assert set(dt.COLUMNS) <= set(rows[0])                    # same columns as the training data
    assert len(al.read_jsonl(al.CRITIQUE_LOG)) == 1
    assert len(al.read_jsonl(al.DISAGREEMENT_LOG)) == (1 if chk["verdict"] in ("review", "disagree") else 0)


def test_confirmed_check_is_not_a_disagreement(audit_dir):
    first = run()
    chk = run(estimate=first["live_effective_median"])
    assert chk["verdict"] == "confirmed" and chk["gap_pct"] == 0
    al.record_check(chk)
    assert al.read_jsonl(al.DISAGREEMENT_LOG) == []
    assert {r["ai_source_type"] for r in al.read_ai_rows()} == {"ai_confirmation"}


def test_rows_without_labels_are_refused(audit_dir):
    with pytest.raises(ValueError):
        al.append_ai_rows([{"listing_key": "x", "rent": 3000}])


# ---------------------------------------------------------------- the AI's output is checked in code
class LyingLLM(lc.RuleBasedLLM):
    """Claims specials and fees the listings don't have, and cites comps that don't exist."""

    def extract(self, subject, buildings):
        out = super().extract(subject, buildings)
        for b in out["buildings"]:
            b.update(concession_weeks=8, concession_applies="all_units", mandatory_monthly_fees=5000)
        out["units"].append({"unit_id": "made-up:1", "comparability": "High", "comparability_reason": "",
                             "excluded": False})
        return out

    def explain(self, subject, summary, comps):
        return {"explanation": "x", "flags": [], "drivers": [
            {"factor": "concessions", "assessment": "missing", "direction": "estimate_too_high",
             "explanation": "", "cited_unit_ids": ["made-up:1"]}]}


def test_invented_numbers_and_citations_are_rejected():
    chk = run(llm=LyingLLM())
    errs = " ".join(chk["validation_errors"])
    assert "unknown unit 'made-up:1'" in errs and "out of range" in errs
    used = [c for c in chk["comps"] if c["used"]]
    assert all(c["mandatory_monthly_fees"] == 0 for c in used)
    assert all(c["concession_weeks"] == 0 for c in used)       # none of these listings has a special
    assert "shows no such special" in errs
    assert chk["drivers"] == []                               # cited nothing real, so dropped


def test_effective_rent_and_verdicts():
    assert lc.effective_rent(4000, 4.33, 12) == pytest.approx(4000 * (1 - 4.33 / 52), abs=0.01)
    assert lc.effective_rent(4000, 4.33, 13) == pytest.approx(4000 * (1 - 4.33 / (13 * 52 / 12)), abs=0.01)
    assert lc.effective_rent(4000, 0) == 4000                 # fees are reported, not added
    assert lc.verdict_for(3.9, 6, 3) == "confirmed"
    assert lc.verdict_for(-7, 6, 3) == "review"
    assert lc.verdict_for(9, 6, 3) == "disagree"
    assert lc.verdict_for(9, 4, 3) == "insufficient_evidence"
    assert lc.verdict_for(9, 6, 2) == "insufficient_evidence"


# ---------------------------------------------------------------- training view
def _ai_row(key, unit, src="ai_disagreement", comp="High", when="2026-11-01T00:00:00+00:00"):
    r = {c: "" for c in al.AI_ROW_COLUMNS}
    r.update(listing_key=key, level="unit", unit_key=unit, rent="3500", beds="1", baths="1", sqft="650",
             zip_code="94107", date_downloaded="2026-11-01", comparability=comp, data_source="ai_audit",
             ai_source_type=src, ai_check_id="c1", ai_checked_at=when, ai_verdict="disagree",
             llm_extraction_model="m", llm_verdict_model="m", prompt_version="v1", pricing_model_trained="t",
             listing_site="apartments.com via RealtyAPI")
    return r


def test_view_keeps_base_rows_and_labels_ai_rows():
    base = [{"listing_key": "b1", "unit_key": "u1", "level": "unit", "rent": "3000", "date_downloaded": "2026-10-28"}]
    ai = [_ai_row("b1", "u1"),                                  # same unit as base: base wins
          _ai_row("n1", "u9"), _ai_row("n1", "u9", when="2026-10-01T00:00:00+00:00"),   # older duplicate
          _ai_row("n2", "u2", comp="Low"),
          _ai_row("n3", "u3", src="ai_confirmation"),
          _ai_row("n4", "u4", when="2026-01-01T00:00:00+00:00")]
    rows, counts = btv.build_rows(base, ai, ai_weight=0.5, today=NOW.date())
    by = {(r["listing_key"], r["unit_key"]): r for r in rows}
    assert by[("b1", "u1")]["data_source"] == "realtyapi_download" and by[("b1", "u1")]["rent"] == "3000"
    assert by[("n1", "u9")]["ai_checked_at"].startswith("2026-11-01") and by[("n1", "u9")]["source_weight"] == 0.5
    assert ("n2", "u2") not in by and ("n4", "u4") not in by and ("n3", "u3") in by
    assert counts["ai_skipped_unit_in_base_data"] == 1 and counts["ai_skipped_older_duplicate"] == 1
    rows2, _ = btv.build_rows(base, ai, include_confirmed=False, today=NOW.date())
    assert ("n3", "u3") not in {(r["listing_key"], r["unit_key"]) for r in rows2}


def test_training_weights_and_scores_use_source_labels():
    import pandas as pd
    import create_model_test as rm
    base = pd.read_csv(rm.DEFAULT_DATA).head(200)
    base["data_source"], base["source_weight"] = "realtyapi_download", 1.0
    ai = base.head(20).copy()
    ai["listing_key"] = ai["listing_key"] + "_ai"
    ai["data_source"], ai["source_weight"] = "ai_audit", 0.5
    d, _ = rm.clean(pd.concat([base, ai]))
    w = d.groupby(["listing_key"])["w_building"].sum()
    assert w[[k for k in w.index if k.endswith("_ai")]].round(6).eq(0.5).all()
    assert w[[k for k in w.index if not k.endswith("_ai")]].round(6).eq(1.0).all()
    data, _ = rm.build_features(d, rm.load_crime_grid())
    assert data["is_ai"].sum() == (d["data_source"] == "ai_audit").sum() > 0


# ---------------------------------------------------------------- feedback gates
def _disagreement(i, assessment="missing", factor="concessions", feature=None):
    return {"check_id": "c{}".format(i), "checked_at": NOW.isoformat(), "verdict": "disagree", "drivers": [
        {"factor": factor, "feature": feature, "assessment": assessment, "direction": "estimate_too_high",
         "cited_unit_ids": ["x"]}]}


MARKET = {"citywide": {"buildings": 30, "share_with_concession": 0.4, "avg_free_weeks_when_offered": 5.0,
                       "effective_discount_pct": 3.5, "discount_t": 4.0, "median_required_fees": 30.0, "fees_t": 1.0},
          "by_zip": {}}


def test_concessions_need_both_the_ai_and_the_listings():
    tally = fb.tally_findings([_disagreement(i) for i in range(10)])
    learned, _ = fb.decide(tally, MARKET, {})
    assert "concessions" in learned["market"] and "mandatory_fees" not in learned["market"]
    few = fb.tally_findings([_disagreement(i) for i in range(5)])            # AI hasn't raised it enough
    assert fb.decide(few, MARKET, {})[0]["market"] == {}
    weak = dict(MARKET, citywide=dict(MARKET["citywide"], discount_t=1.2))  # listings don't back it up
    assert fb.decide(tally, weak, {})[0]["market"] == {}
    mixed = fb.tally_findings([_disagreement(i, "overweighted" if i % 2 else "missing") for i in range(10)])
    assert fb.decide(mixed, MARKET, {})[0]["market"] == {}                  # AI doesn't agree with itself


def test_study_premium_is_recalibrated_and_shown_with_its_source(monkeypatch):
    tally = fb.tally_findings([_disagreement(i, "underweighted", "amenities", "Private outdoor space") for i in range(9)])
    prem = {("feature", "Private outdoor space"): {"buildings_with": 12, "buildings_without": 20, "model_priced": False,
                                                   "measurable": True, "log_premium": 0.09, "se": 0.02}}
    learned, report = fb.decide(tally, {}, prem)
    got = learned["feature"]["Private outdoor space"]
    study = fa.EFFECTS[("feature", "Private outdoor space")]["value"]
    assert study < got["value"] < got["measured_value"]                   # moved toward the comps, not all the way
    monkeypatch.setattr(fa, "LEARNED", learned)
    e = fa.effect("feature", "Private outdoor space")
    assert e["value"] == got["value"] and e["sources"][-1]["name"] == "AI live-check findings"
    flat = dict(prem[("feature", "Private outdoor space")], log_premium=0.045)   # matches the study: no change
    assert fb.decide(tally, {}, {("feature", "Private outdoor space"): flat})[0]["feature"] == {}


def test_model_priced_amenities_go_to_retraining_not_premiums():
    tally = fb.tally_findings([_disagreement(i, "underweighted", "amenities", "Fitness center") for i in range(9)])
    prem = {("feature", "Fitness center"): {"buildings_with": 12, "buildings_without": 20, "model_priced": True,
                                            "measurable": True, "log_premium": 0.2, "se": 0.02}}
    learned, report = fb.decide(tally, {}, prem)
    assert learned["feature"] == {} and "retrain" in report[-1]["action"]


def test_learned_adjustments_drop_out_when_findings_stop(audit_dir):
    old = (NOW - timedelta(days=fb.WINDOW_DAYS + 5)).isoformat()
    recs = [dict(_disagreement(i), checked_at=old) for i in range(10)]
    al.append_jsonl(al.DISAGREEMENT_LOG, recs)
    assert fb._recent(al.read_jsonl(al.DISAGREEMENT_LOG), fb.WINDOW_DAYS) == []


def test_github_mirror_writes_to_its_own_branch(monkeypatch):
    from urllib.error import HTTPError
    calls, files = [], {}

    class FakeMirror(al.GitHubMirror):
        def _api(self, method, path, body=None):
            calls.append((method, path, body))
            if path == "git/ref/heads/" + al.LOG_BRANCH and not any(c[0] == "POST" for c in calls):
                raise HTTPError(path, 404, "missing", {}, None)
            return {"object": {"sha": "main-sha"}}

        def _req(self, method, rel, body=None, raw=False):
            calls.append((method, rel, body and body.get("branch")))
            if method == "GET":
                if rel not in files:
                    raise HTTPError(rel, 404, "missing", {}, None)
                return files[rel].encode() if raw else {"sha": "s"}
            import base64
            files[rel] = base64.b64decode(body["content"]).decode()
            return {}

    m = FakeMirror("token")
    m.append("Data/ai_audit/critique_log.jsonl", "a\n", "msg")
    m.append("Data/ai_audit/critique_log.jsonl", "b\n", "msg")
    assert files["Data/ai_audit/critique_log.jsonl"] == "a\nb\n"
    assert ("POST", "git/refs", {"ref": "refs/heads/ai-audit-logs", "sha": "main-sha"}) in calls
    assert all(c[2] == "ai-audit-logs" for c in calls if c[0] == "PUT")
