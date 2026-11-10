"""Logs written by the AI live check, kept apart from the original training data.

Everything the AI side produces goes to Data/ai_audit/. This module refuses to write anywhere
under Data/training/ or to the base training files, so AI output can never end up in them; it
reaches training only through Code/build_training_view.py, which writes a separate file.

Files (all append-only)
  critique_log.jsonl      one record per live check, whatever the verdict: the subject unit, the
                          model estimate, the live comps the AI read, its explanation and drivers.
  disagreement_log.jsonl  the same record, only for checks the AI did not confirm (Review or
                          Disagrees). The disagreements that drive Code/ai_feedback.py.
  decisions.jsonl         what the person did with a check: kept the model, used the live median,
                          or sent it to an analyst.
  ai_training_rows.csv    every comp unit the AI read and passed the checks, in the same columns
                          as Data/realty_training_combined.csv (parsed by download_training.unit_rows)
                          plus the AI-extracted fields and the source labels below.
  model_adjustments.jsonl what Code/ai_feedback.py found and did (adjustments, retrain results).
  learned_adjustments.json the adjustments currently in force (read by feature_adjustments.py).

Source labels on every AI training row
  data_source       "ai_audit" (base rows get "realtyapi_download" in the training view only)
  ai_source_type    "ai_disagreement"            comp read during a Review / Disagrees check
                    "ai_confirmation"            comp read during a Confirmed check
                    "ai_insufficient_evidence"   comp read during a check with too few comps
  ai_check_id, ai_checked_at, ai_verdict, llm_extraction_model, llm_verdict_model,
  prompt_version, pricing_model_trained, listing_site

On Streamlit Community Cloud the app's disk is wiped on every restart. Call configure_github()
with a token (the app does this when a GITHUB_TOKEN secret is set) and every append is also
committed to the repository, on its own branch (LOG_BRANCH, created from main if missing) so the
commits don't redeploy the app, which watches main. To work with the logs locally:
    git fetch origin ai-audit-logs && git checkout origin/ai-audit-logs -- Data/ai_audit
"""
import base64
import csv
import io
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import download_training as dt

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "Data"
AI_DIR = DATA / "ai_audit"
CRITIQUE_LOG = AI_DIR / "critique_log.jsonl"
DISAGREEMENT_LOG = AI_DIR / "disagreement_log.jsonl"
DECISIONS_LOG = AI_DIR / "decisions.jsonl"
AI_ROWS = AI_DIR / "ai_training_rows.csv"
ADJUSTMENTS_LOG = AI_DIR / "model_adjustments.jsonl"
LEARNED_ADJUSTMENTS = AI_DIR / "learned_adjustments.json"
TRAINING_VIEWS = DATA / "training_views"     # generated training files (base + labeled AI rows)

# Never written by the AI side.
PROTECTED = [DATA / "training", DATA / "realty_training_combined.csv", DATA / "realty_training.csv",
             DATA / "realty_training_combined_final.csv", DATA / "realty_training_with_extra.csv",
             DATA / "properties.csv", DATA / "crime_grid.csv"]

SOURCE_BASE = "realtyapi_download"
SOURCE_AI = "ai_audit"
SOURCE_TYPE = {"review": "ai_disagreement", "disagree": "ai_disagreement",
               "confirmed": "ai_confirmation", "insufficient_evidence": "ai_insufficient_evidence"}
DISAGREEMENT_VERDICTS = {"review", "disagree"}

AI_FIELD_COLUMNS = ["concession_weeks", "concession_text", "mandatory_monthly_fees", "lease_term_months",
                    "condition", "comparability", "comparability_reason", "effective_rent"]
PROVENANCE_COLUMNS = ["data_source", "ai_source_type", "ai_check_id", "ai_checked_at", "ai_verdict",
                      "llm_extraction_model", "llm_verdict_model", "prompt_version", "pricing_model_trained",
                      "listing_site"]
AI_ROW_COLUMNS = dt.COLUMNS + AI_FIELD_COLUMNS + PROVENANCE_COLUMNS


def assert_not_base(path):
    """Raise if `path` is the base training data or inside it."""
    p = Path(path).resolve()
    for q in PROTECTED:
        q = q.resolve()
        if p == q or q in p.parents:
            raise PermissionError("AI audit output may not be written to {} (base training data)".format(q))
    allowed = [AI_DIR.resolve(), TRAINING_VIEWS.resolve()]
    if not any(a in p.parents for a in allowed):
        raise PermissionError("AI audit output goes under {} or {}, not {}".format(AI_DIR, TRAINING_VIEWS, p))


# ---------------------------------------------------------------- optional GitHub mirror
_GITHUB = None


LOG_BRANCH = "ai-audit-logs"
GITHUB_REPO = "aldenbarnes/structure_properties_fall_2026"


class GitHubMirror:
    """Append to a file in the GitHub repository through the contents API (one commit per append)."""

    def __init__(self, token, repo=GITHUB_REPO, branch=LOG_BRANCH, base="main"):
        self.token, self.repo, self.branch, self.base = token, repo, branch, base
        self._branch_ok = False

    def _api(self, method, path, body=None):
        headers = {"Authorization": "Bearer " + self.token, "X-GitHub-Api-Version": "2022-11-28",
                   "Accept": "application/vnd.github+json"}
        data = json.dumps(body).encode() if body is not None else None
        req = Request("https://api.github.com/repos/{}/{}".format(self.repo, path), data=data, method=method,
                      headers=headers)
        with urlopen(req, timeout=30) as r:
            return json.loads(r.read() or b"{}")

    def ensure_branch(self):
        if self._branch_ok:
            return
        try:
            self._api("GET", "git/ref/heads/" + self.branch)
        except HTTPError as e:
            if e.code != 404:
                raise
            sha = self._api("GET", "git/ref/heads/" + self.base)["object"]["sha"]
            self._api("POST", "git/refs", {"ref": "refs/heads/" + self.branch, "sha": sha})
        self._branch_ok = True

    def _req(self, method, rel, body=None, raw=False):
        url = "https://api.github.com/repos/{}/contents/{}".format(self.repo, rel)
        if method == "GET":
            url += "?ref=" + self.branch
        headers = {"Authorization": "Bearer " + self.token, "X-GitHub-Api-Version": "2022-11-28",
                   "Accept": "application/vnd.github.raw" if raw else "application/vnd.github+json"}
        data = json.dumps(body).encode() if body is not None else None
        with urlopen(Request(url, data=data, method=method, headers=headers), timeout=30) as r:
            out = r.read()
        return out if raw else json.loads(out or b"{}")

    def append(self, rel, text, message):
        self.ensure_branch()
        for attempt in range(2):           # one retry if someone else committed in between
            try:
                sha = self._req("GET", rel).get("sha")
                old = self._req("GET", rel, raw=True).decode("utf-8")
            except HTTPError as e:
                if e.code != 404:
                    raise
                sha, old = None, ""
            body = {"message": message, "branch": self.branch,
                    "content": base64.b64encode((old + text).encode("utf-8")).decode()}
            if sha:
                body["sha"] = sha
            try:
                self._req("PUT", rel, body)
                return
            except HTTPError as e:
                if e.code not in (409, 422) or attempt:
                    raise


def configure_github(token, repo=GITHUB_REPO, branch=LOG_BRANCH):
    """Mirror every append to GitHub (needed on Streamlit Cloud, whose disk does not persist)."""
    global _GITHUB
    if not token:
        _GITHUB = None
    elif _GITHUB is None or (_GITHUB.token, _GITHUB.repo, _GITHUB.branch) != (token, repo, branch):
        _GITHUB = GitHubMirror(token, repo, branch)


def persistence_note():
    return ("Saved to GitHub on the {} branch.".format(_GITHUB.branch) if _GITHUB else
            "Saved to this machine's Data/ai_audit folder. On Streamlit Cloud that folder is wiped on restart; "
            "add a GITHUB_TOKEN secret to keep the logs.")


# ---------------------------------------------------------------- writing
def _append_text(path, text, message):
    assert_not_base(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="") as f:
        f.write(text)
    if _GITHUB is not None:
        try:
            _GITHUB.append(path.relative_to(REPO).as_posix(), text, message)
        except Exception as exc:   # the local copy is written either way
            return "GitHub save failed: {}".format(exc)
    return None


def append_jsonl(path, records, message="AI live check log"):
    text = "".join(json.dumps(r, default=str, ensure_ascii=False) + "\n" for r in records)
    return _append_text(Path(path), text, message) if text else None


def append_ai_rows(rows, path=None, message="AI live check training rows"):
    if not rows:
        return None
    path = Path(path or AI_ROWS)
    new_file = not path.exists() or path.stat().st_size == 0
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=AI_ROW_COLUMNS, extrasaction="ignore", lineterminator="\n")
    if new_file:
        w.writeheader()
    for r in rows:
        bad = [c for c in PROVENANCE_COLUMNS if r.get(c) in (None, "")]
        if bad:
            raise ValueError("AI training row is missing source labels: " + ", ".join(bad))
        w.writerow(r)
    return _append_text(path, buf.getvalue(), message)


def provenance(check):
    """Source-label columns for every training row from one check."""
    return {"data_source": SOURCE_AI, "ai_source_type": SOURCE_TYPE[check["verdict"]],
            "ai_check_id": check["check_id"], "ai_checked_at": check["checked_at"],
            "ai_verdict": check["verdict"], "llm_extraction_model": check["llm"]["extraction_model"],
            "llm_verdict_model": check["llm"]["verdict_model"], "prompt_version": check["llm"]["prompt_version"],
            "pricing_model_trained": check["pricing_model"]["trained"],
            "listing_site": "apartments.com via RealtyAPI"}


def record_check(check):
    """Write one finished live check to the logs. Returns a list of problems (empty if none)."""
    problems = []
    msg = "AI live check {} ({})".format(check["check_id"], check["verdict"])
    problems.append(append_jsonl(CRITIQUE_LOG, [check], msg))
    if check["verdict"] in DISAGREEMENT_VERDICTS:
        problems.append(append_jsonl(DISAGREEMENT_LOG, [check], msg))
    if check["verdict"] in SOURCE_TYPE:
        labels = provenance(check)
        rows = [dict(c["training_row"], **labels) for c in check["comps"] if c.get("training_row")]
        problems.append(append_ai_rows(rows, message=msg))
    return [p for p in problems if p]


def record_decision(check_id, decision, by=None, note=None, when=None):
    from datetime import datetime, timezone
    rec = {"check_id": check_id, "decision": decision, "by": by, "note": note,
           "at": when or datetime.now(timezone.utc).isoformat(timespec="seconds")}
    return append_jsonl(DECISIONS_LOG, [rec], "AI live check decision {}".format(check_id))


# ---------------------------------------------------------------- reading
def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_ai_rows(path=None):
    path = Path(path or AI_ROWS)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_learned_adjustments(path=None):
    try:
        with open(path or LEARNED_ADJUSTMENTS, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def write_learned_adjustments(obj, path=None):
    path = Path(path or LEARNED_ADJUSTMENTS)
    assert_not_base(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)   # ai_feedback.py runs offline (Colab / a laptop); commit it with git like the downloads
