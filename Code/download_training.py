"""Rolling RealtyAPI training download: each run adds the San Francisco buildings not downloaded yet.

    python Code/download_training.py                  # up to 1,000 requests
    python Code/download_training.py --max-requests 200
    python Code/download_training.py --dry-run        # search only: how many buildings would be requested
    python Code/download_training.py --combine-only   # rebuild the combined file, no API calls
    python Code/download_training.py --reparse        # rebuild every run file from the saved raw responses

Set REALTYAPI_KEY (or enter it at the hidden prompt). Standard library only; uses realty_lookup.

What a run does
  1. Pages through the apartments.com search for San Francisco (1 request per 500 results).
  2. Picks the buildings to request, 1 request each, until --max-requests (default 1,000, searches
     included) is used up:
       first   buildings never downloaded (not in Data/training/queried_listings.csv),
       then    buildings whose latest download is more than --refresh-days (default 60) days old,
               oldest first. Ones the cap doesn't reach keep their old rows and wait for a later run.
     Buildings downloaded more recently are never requested again, including ones that came back
     with no priced units. A building whose request failed is not logged, so it counts as new.
  3. Requests /details/byid for each and reads its units from detail.availabilities (see unit_rows).
  4. Writes the new rows to Data/training/realty_training_<date>.csv (a new file every run, never
     overwriting earlier ones), the raw responses to Data/training/raw/byid_<date>.jsonl.gz (same
     <date> part, e.g. 2026-10-28 or 2026-10-28_2), and appends the buildings to queried_listings.csv.
  5. Rebuilds Data/realty_training_combined.csv from every file in Data/training. A unit (or floor
     plan) downloaded more than once keeps only its newest row; units no longer listed when a
     building is refreshed keep their older rows, which recency_weight counts for less.

Columns
  The same columns as Data/realty_training.csv, plus:
    date_downloaded      the date the building's details were requested (YYYY-MM-DD).
    days_since_download  days between date_downloaded and the day the combined file was built.
    recency_weight       0.5 ** (days_since_download / --half-life), so a row --half-life days old
                         counts half as much as one downloaded today. Recomputed on every rebuild.
  The raw responses are kept so a parsing fix can be applied to past runs with --reparse, without
  new requests.

Importing an earlier download (done once for the original file):
    python Code/download_training.py --import Data/realty_training.csv --date 2026-10-20
"""
import argparse
import csv
import getpass
import gzip
import json
import os
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import realty_lookup as rl

DATA = Path(__file__).resolve().parent.parent / "Data"
TRAINING_DIR = DATA / "training"
QUERIED_PATH = TRAINING_DIR / "queried_listings.csv"
COMBINED_PATH = DATA / "realty_training_combined.csv"
RUN_PREFIX = "realty_training_"
SEARCH_PAGE = 500
HALF_LIFE_DAYS = 90
REFRESH_DAYS = 60
MAX_REQUESTS = 1000

COLUMNS = ["listing_key", "level", "unit_key", "unit_number", "floor_plan", "rent", "rent_min", "rent_max",
           "price_with_fees", "beds", "baths", "sqft", "available_date", "rent_source", "name", "address",
           "zip_code", "latitude", "longitude", "property_type", "amenities", "units_in_listing", "weight",
           "date_downloaded"]
COMBINED_COLUMNS = COLUMNS + ["days_since_download", "recency_weight"]
QUERIED_COLUMNS = ["listing_key", "date_downloaded", "units_found", "status"]
# Errors that will repeat on every following request, so the run stops instead of spending more.
STOP_ERRORS = re.compile(r"request cap|HTTP (401|402|403|429)|key|credit|quota|unauthorized", re.I)


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, fields, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def run_files():
    return sorted(TRAINING_DIR.glob(RUN_PREFIX + "*.csv"))


def run_path(stem):
    return TRAINING_DIR / (RUN_PREFIX + stem + ".csv")


def raw_path(stem):
    return TRAINING_DIR / "raw" / ("byid_" + stem + ".jsonl.gz")


def new_stem(day):
    """The run name for `day`: the date, with _2, _3 ... added when a run already used it."""
    stem, n = day, 2
    while run_path(stem).exists() or raw_path(stem).exists():
        stem, n = "{}_{}".format(day, n), n + 1
    return stem


def last_downloaded():
    """{listing_key: latest date_downloaded} over the queried log and the rows of every run file."""
    rows = read_csv(QUERIED_PATH) if QUERIED_PATH.exists() else []
    rows += [r for path in run_files() for r in read_csv(path)]
    latest = {}
    for r in rows:
        k, d = r["listing_key"], r["date_downloaded"]
        if d > latest.get(k, ""):
            latest[k] = d
    return latest


def plan_requests(found, latest, budget, refresh_days, today):
    """(new, refresh): listing keys to request, new buildings first, then stale ones oldest first,
    together at most `budget`."""
    new = [k for k in found if k not in latest]
    stale = sorted((k for k in found if k in latest and (today - date.fromisoformat(latest[k])).days > refresh_days),
                   key=lambda k: latest[k])
    new = new[:budget]
    return new, stale[:budget - len(new)]


def append_queried(rows):
    new = not QUERIED_PATH.exists()
    with open(QUERIED_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=QUERIED_COLUMNS)
        if new:
            w.writeheader()
        w.writerows(rows)


def search_all(client, location, log):
    """Every search result for `location`, following nextPage, keyed by listing_key."""
    results, page = {}, 1
    while True:
        data = client.get("/search/bylocation", {"location": location, "page": page, "resultCount": SEARCH_PAGE})
        batch = [r for r in data.get("searchResults") or [] if isinstance(r, dict) and r.get("listingKey")]
        added = sum(1 for r in batch if r["listingKey"] not in results)
        for r in batch:
            results.setdefault(r["listingKey"], r)
        log("search page {}: {} results, {} new keys (reported total {})".format(page, len(batch), added, data.get("total")))
        if not data.get("nextPage") or not batch or not added:
            return results
        page += 1


# detail.propertyType -> the search's propertyType, for search results that leave it out.
PROPERTY_TYPES = {"apartment community": "Apartment", "apartment unit for rent": "Apartment",
                  "residential condo unit for rent": "Condo", "single-family home": "House", "townhome": "Townhouse"}


def _money_range(text):
    """"$3,641 - 3,741" -> (3641.0, 3741.0); "$3,641" -> (3641.0, 3641.0); "Call for Rent" -> (None, None)."""
    nums = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", str(text or ""))]
    return (min(nums), max(nums)) if nums else (None, None)


def unit_rows(search_result, details, day):
    """Training rows (COLUMNS) for one building: one per unit or floor plan that has a rent.

    /details/byid lists them in detail.availabilities: groups ("All", "Studio", "1 Bed", ...) whose
    "details" are floor plans (a model key "mk", no "uk") and units ("uk", with the "mk" of their
    plan when they have one). "All" holds every entry, so only it is read. A floor plan becomes a
    row only when none of its units are listed, so a unit is never counted twice. Rent is the low
    end of rentRange ("Call for Rent" rows are skipped); price_with_fees is the low end of priceRange.
    """
    detail = details.get("detail") if isinstance(details.get("detail"), dict) else {}
    groups = [g for g in detail.get("availabilities") or [] if isinstance(g, dict)]
    entries = [x for g in groups if g.get("name") == "All" for x in g.get("details") or [] if isinstance(x, dict)]
    if not entries:   # no "All" group: read every group, dropping repeats
        seen = set()
        for x in (x for g in groups for x in g.get("details") or [] if isinstance(x, dict)):
            k = json.dumps(x, sort_keys=True)
            if k not in seen:
                seen.add(k)
                entries.append(x)
    plans_with_units = {x.get("mk") for x in entries if x.get("uk") and x.get("mk")}
    summary = rl.property_summary(search_result, details)
    ptype = search_result.get("propertyType") or PROPERTY_TYPES.get(str(detail.get("propertyType") or "").lower())
    amenities = "; ".join(n.strip() for n in search_result.get("amenityNames") or [] if isinstance(n, str) and n.strip())
    units = []
    for x in entries:
        if not x.get("uk") and x.get("mk") in plans_with_units:
            continue
        rent_min, rent_max = _money_range(x.get("rentRange"))
        if rent_min is None:
            continue
        level = "unit" if x.get("uk") else "floor_plan"
        units.append({
            "listing_key": search_result["listingKey"], "level": level, "unit_key": x.get("uk"),
            "unit_number": x.get("unitNumber"), "floor_plan": x.get("modelName"), "rent": rent_min,
            "rent_min": rent_min, "rent_max": rent_max, "price_with_fees": _money_range(x.get("priceRange"))[0],
            "beds": rl._num(x.get("bedNum") if x.get("bedNum") not in (None, "") else x.get("beds"), beds=True),
            "baths": rl._num(x.get("bathNum") if x.get("bathNum") not in (None, "") else x.get("baths")),
            "sqft": rl._num(x.get("area")), "available_date": str(x.get("availabilityDate") or "")[:10] or None,
            "rent_source": "detail_" + level, "name": summary["name"], "address": summary["address"],
            "zip_code": summary["zip_code"], "latitude": summary["latitude"], "longitude": summary["longitude"],
            "property_type": ptype, "amenities": amenities, "date_downloaded": day,
        })
    for u in units:
        u["units_in_listing"], u["weight"] = len(units), 1.0 / len(units)
    return units


def reparse():
    """Rebuild each run file that has saved raw responses, and correct its buildings in the queried log."""
    counts = {}
    for raw in sorted((TRAINING_DIR / "raw").glob("byid_*.jsonl.gz")):
        stem = raw.name[len("byid_"):-len(".jsonl.gz")]
        day = stem.split("_")[0]
        rows = []
        with gzip.open(raw, "rt", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                units = unit_rows(rec["search_result"], rec["details"], day)
                rows.extend(units)
                counts[(rec["listing_key"], day)] = len(units)
        if rows:
            write_csv(run_path(stem), COLUMNS, rows)
        elif run_path(stem).exists():
            run_path(stem).unlink()
        print("{}: {} rows".format(run_path(stem).name, len(rows)))
    if QUERIED_PATH.exists():
        log = read_csv(QUERIED_PATH)
        for r in log:
            n = counts.get((r["listing_key"], r["date_downloaded"]))
            if n is not None and r["status"] != "imported":
                r["units_found"], r["status"] = n, "ok" if n else "no_priced_units"
        write_csv(QUERIED_PATH, QUERIED_COLUMNS, log)


def _row_key(r):
    """Same duplicate rule as train_rent_model.load_pool: a unit by its unit_key, otherwise a floor plan."""
    if r.get("unit_key"):
        return (r["listing_key"], "unit", r["unit_key"])
    return (r["listing_key"], "plan", r.get("floor_plan"), r.get("beds"), r.get("baths"), r.get("sqft"))


def combine(half_life, today=None):
    """Rebuild COMBINED_PATH from every run file: newest download wins on a duplicate row."""
    today = today or date.today()
    rows = [r for path in run_files() for r in read_csv(path)]
    rows.sort(key=lambda r: r.get("date_downloaded") or "", reverse=True)
    out, seen = [], set()
    for r in rows:
        k = _row_key(r)
        if k in seen:
            continue
        seen.add(k)
        days = (today - date.fromisoformat(r["date_downloaded"])).days
        r["days_since_download"] = days
        r["recency_weight"] = round(0.5 ** (max(days, 0) / half_life), 6)
        out.append(r)
    write_csv(COMBINED_PATH, COMBINED_COLUMNS, out)
    return len(rows), len(out)


def import_file(path, day):
    """Copy an earlier training CSV into Data/training with a date_downloaded column."""
    rows = read_csv(path)
    missing = [c for c in COLUMNS[:-1] if c not in rows[0]] if rows else COLUMNS[:-1]
    if missing:
        sys.exit("{} is missing training columns: {}".format(path, ", ".join(missing)))
    for r in rows:
        r["date_downloaded"] = day
    out = run_path(new_stem(day))
    write_csv(out, COLUMNS, rows)
    counts = {}
    for r in rows:
        counts[r["listing_key"]] = counts.get(r["listing_key"], 0) + 1
    append_queried([{"listing_key": k, "date_downloaded": day, "units_found": n, "status": "imported"}
                    for k, n in counts.items()])
    print("Imported {} rows ({} buildings) to {}".format(len(rows), len(counts), out))


def download(args):
    key = os.environ.get("REALTYAPI_KEY", "").strip()
    if not key and sys.stdin.isatty():
        key = getpass.getpass("RealtyAPI key (hidden): ").strip()
    if not key:
        sys.exit("Set REALTYAPI_KEY, or run in a terminal for the hidden key prompt.")
    client = rl.Client(key, max_requests=args.max_requests)
    today = date.today()
    day = today.isoformat()
    latest = last_downloaded()
    print("{} buildings already downloaded".format(len(latest)))
    try:
        found = search_all(client, args.location, print)
    except rl.APIError as exc:
        sys.exit("Search failed: {}".format(exc))
    budget = args.max_requests - client.requests
    new, refresh = plan_requests(found, latest, budget, args.refresh_days, today)
    n_new = sum(1 for k in found if k not in latest)
    n_stale = sum(1 for k in found if k in latest and (today - date.fromisoformat(latest[k])).days > args.refresh_days)
    print("{} buildings in the search: {} new, {} last downloaded over {} days ago".format(
        len(found), n_new, n_stale, args.refresh_days))
    print("This run: {} new and {} refreshes ({} requests left after the search); {} new and {} refreshes wait "
          "for a later run".format(len(new), len(refresh), budget, n_new - len(new), n_stale - len(refresh)))
    if args.dry_run:
        print("Dry run: requests used {}".format(client.requests))
        return 0
    todo = new + refresh
    rows, queried, errors = [], [], 0
    stem = new_stem(day)
    raw_file = raw_path(stem)
    raw_file.parent.mkdir(exist_ok=True)
    try:
        with gzip.open(raw_file, "wt", encoding="utf-8") as raw:
            for i, listing_key in enumerate(todo, 1):
                try:
                    details = rl.details(client, listing_key)
                except rl.APIError as exc:
                    errors += 1
                    print("{}/{} {}: {}".format(i, len(todo), listing_key, exc))
                    if STOP_ERRORS.search(str(exc)):
                        print("Stopping: this error would repeat on every request.")
                        break
                    continue   # not logged as queried, so the next run tries it again
                raw.write(json.dumps({"listing_key": listing_key, "fetched_at": datetime.now(timezone.utc).isoformat(),
                                      "search_result": found[listing_key], "details": details}) + "\n")
                units = unit_rows(found[listing_key], details, day)
                rows.extend(units)
                queried.append({"listing_key": listing_key, "date_downloaded": day, "units_found": len(units),
                                "status": "ok" if units else "no_priced_units"})
                print("{}/{} {}{}: {} units".format(i, len(todo), listing_key,
                                                    " (refresh)" if listing_key in latest else "", len(units)))
    except KeyboardInterrupt:
        print("Interrupted; saving what was downloaded.")
    finally:
        if rows:
            write_csv(run_path(stem), COLUMNS, rows)
            print("Wrote {} rows from {} buildings to {}".format(len(rows), len(queried), run_path(stem)))
        if len(queried) >= 10 and not rows:
            # Every building empty means the response format changed, not that nothing is for rent.
            # Leave them out of the log so the next run asks again; the raw file keeps the responses.
            print("WARNING: none of the {} buildings had priced units, so they are not marked as downloaded. "
                  "Check the parser against {} before running again.".format(len(queried), raw_file))
            errors += 1
        elif queried:
            append_queried(queried)
        else:
            raw_file.unlink(missing_ok=True)
    total, kept = combine(args.half_life)
    print("Combined: {} rows ({} duplicates dropped) in {}".format(kept, total - kept, COMBINED_PATH))
    print("Requests used: {} (credits may differ). Errors: {}".format(client.requests, errors))
    return 1 if errors else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--location", default="San Francisco, CA")
    ap.add_argument("--max-requests", type=int, default=MAX_REQUESTS,
                    help="cap on RealtyAPI requests this run, searches included (default %(default)s)")
    ap.add_argument("--refresh-days", type=int, default=REFRESH_DAYS,
                    help="re-download a building once its latest download is more than this many days old "
                         "(default %(default)s), after all new buildings")
    ap.add_argument("--half-life", type=float, default=HALF_LIFE_DAYS,
                    help="days after which a row's recency_weight is 0.5 (default %(default)s)")
    ap.add_argument("--dry-run", action="store_true", help="search only; report how many buildings are new")
    ap.add_argument("--combine-only", action="store_true", help="rebuild the combined file without API calls")
    ap.add_argument("--reparse", action="store_true",
                    help="rebuild every run file from Data/training/raw, then the combined file (no API calls)")
    ap.add_argument("--import", dest="import_path", type=Path, help="add an earlier training CSV to Data/training")
    ap.add_argument("--date", help="with --import: the date that file was downloaded (YYYY-MM-DD)")
    args = ap.parse_args(argv)
    if args.max_requests < 1 or args.refresh_days < 0 or args.half_life <= 0:
        ap.error("--max-requests and --half-life must be positive, --refresh-days not negative")
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    if args.import_path:
        if not args.date:
            ap.error("--import needs --date")
        import_file(args.import_path, date.fromisoformat(args.date).isoformat())
        args.combine_only = True
    if args.reparse:
        reparse()
        args.combine_only = True
    if args.combine_only:
        total, kept = combine(args.half_life)
        print("Combined: {} rows ({} duplicates dropped) in {}".format(kept, total - kept, COMBINED_PATH))
        return 0
    return download(args)


if __name__ == "__main__":
    sys.exit(main())
