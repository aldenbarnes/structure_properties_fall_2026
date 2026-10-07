#!/usr/bin/env python3
"""Download Apartments.com property summaries and detailed amenities via RealtyAPI.

Requires Python 3.8+; no packages to install.
Run: python3 realtyapi_pull.py --location "San Francisco, CA" --limit 20
Enter your RealtyAPI key at the hidden prompt, or set REALTYAPI_KEY.
Optional: --listing-key 26ht3d9 --listing-key mrb74sk (skips location search).

Outputs in --output (default realtyapi_output):
  properties.json: original search fields plus full amenity response and timestamps
  properties.csv: one row per property with amenity lists encoded as JSON
  amenities.csv: one row per advertised amenity, preserving category and qualifiers
  run_summary.json: request count, errors, settings, and completion status

These are advertised property features, NOT verified features of each apartment.
Missing amenities mean unknown, not absent. Rent ranges are asking-rent summaries.
Request count is NOT credit count; check endpoint prices in your API Playground.
Existing output files are replaced. JSON checkpoints are saved after each property.
Docs: https://www.realtyapi.io/api/apartments
"""

import argparse
import csv
import getpass
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = "https://apartments.realtyapi.io"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class APIError(Exception):
    def __init__(self, message, fatal=False):
        super().__init__(message)
        self.fatal = fatal


class Client:
    def __init__(self, key, max_requests, timeout):
        self.key = key
        self.max_requests = max_requests
        self.timeout = timeout
        self.requests = 0

    def get(self, endpoint, params):
        if self.requests >= self.max_requests:
            raise APIError("Request cap reached; increase --max-requests if needed.", True)
        self.requests += 1
        req = Request(BASE_URL + endpoint + "?" + urlencode(params),
                      headers={"x-realtyapi-key": self.key, "Accept": "application/json"})
        try:
            with urlopen(req, timeout=self.timeout) as response:
                data = json.load(response)
        except HTTPError as exc:
            # Do not print response bodies or headers that could contain credentials.
            raise APIError("HTTP {} on {}".format(exc.code, endpoint),
                           exc.code in (401, 402, 403, 429)) from None
        except (URLError, TimeoutError, OSError, ValueError):
            raise APIError("Network timeout, connection failure, or invalid JSON on " + endpoint) from None
        if not isinstance(data, dict):
            raise APIError("Unexpected response format on " + endpoint)
        message = str(data.get("message", ""))
        if message and not message.lower().startswith("success"):
            raise APIError("API did not report success on " + endpoint,
                           any(x in message.lower() for x in ("key", "credit", "quota", "unauthorized")))
        time.sleep(0.15)
        return data


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def amenity_rows(record):
    for category in record.get("amenity_response", {}).get("amenities", []):
        for group in category.get("groups", []):
            for item in group.get("items", []):
                yield {
                    "listing_key": record["property"].get("listingKey", ""),
                    "property_name": record["property"].get("name", ""),
                    "category": category.get("category", ""),
                    "group": group.get("name", ""),
                    "amenity": item,
                    "scope": "property_advertisement; exact_unit_unverified",
                    "fetched_at": record.get("amenities_fetched_at", ""),
                }


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export(output, records, summary):
    write_json(output / "properties.json", records)
    write_json(output / "run_summary.json", summary)
    rows, all_amenities = [], []
    for record in records:
        prop = record["property"]
        address = prop.get("address") or {}
        detailed = list(amenity_rows(record))
        all_amenities.extend(detailed)
        rows.append({
            "listing_key": prop.get("listingKey", ""),
            "name": prop.get("name", ""),
            "address": prop.get("oneLineAddress", ""),
            "city": address.get("city", ""),
            "state": address.get("state", ""),
            "postal_code": address.get("postalCode", ""),
            "latitude": prop.get("latitude", ""),
            "longitude": prop.get("longitude", ""),
            "property_type": prop.get("propertyType", ""),
            "bed_range": prop.get("bedRange", ""),
            "rent_range": prop.get("rentRange", ""),
            "price_range": prop.get("priceRange", ""),
            "has_availabilities": prop.get("hasAvailabilities", ""),
            "search_amenities_json": json.dumps(prop.get("amenityNames", [])),
            "detailed_amenities_json": json.dumps([a["amenity"] for a in detailed]),
            "amenity_status": record.get("amenity_status", "not_fetched"),
            "property_fetched_at": record["property_fetched_at"],
            "amenities_fetched_at": record.get("amenities_fetched_at", ""),
            "source": "apartments.com via RealtyAPI",
        })
    fields = list(rows[0]) if rows else ["listing_key", "name", "amenity_status"]
    write_csv(output / "properties.csv", fields, rows)
    write_csv(output / "amenities.csv", ["listing_key", "property_name", "category", "group",
                                         "amenity", "scope", "fetched_at"], all_amenities)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--location", default="San Francisco, CA")
    parser.add_argument("--limit", type=int, default=20, help="Maximum properties (default 20)")
    parser.add_argument("--listing-key", action="append", help="Specific listing key; repeat for multiple properties")
    parser.add_argument("--output", type=Path, default=Path("realtyapi_output"))
    parser.add_argument("--max-requests", type=int, default=50, help="HTTP request cap, NOT credit cap (default 50)")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args(argv)
    if min(args.limit, args.max_requests, args.timeout) < 1:
        parser.error("limit, max-requests, and timeout must be positive")
    key = os.environ.get("REALTYAPI_KEY", "").strip()
    if not key:
        if not sys.stdin.isatty():
            parser.error("Set REALTYAPI_KEY, or run in a terminal for the hidden key prompt")
        key = getpass.getpass("RealtyAPI key (hidden): ").strip()
    if not key:
        parser.error("API key cannot be empty")
    client = Client(key, args.max_requests, args.timeout)
    args.output.mkdir(parents=True, exist_ok=True)
    records, seen = [], set()
    summary = {"started_at": utc_now(), "location": args.location,
               "property_limit": args.limit, "requests_attempted": 0,
               "errors": [], "complete": False}
    try:
        if args.listing_key:
            for listing_key in dict.fromkeys(args.listing_key):
                if len(records) >= args.limit:
                    break
                response = client.get("/details/byid", {"listingKey": listing_key})
                records.append({"property": {"listingKey": listing_key},
                                "property_detail_response": response,
                                "property_fetched_at": utc_now()})
                export(args.output, records, summary)
        else:
            page = 1
            while len(records) < args.limit:
                response = client.get("/search/bylocation", {
                    "location": args.location, "page": page,
                    "resultCount": min(500, args.limit - len(records))})
                results = response.get("searchResults")
                if not isinstance(results, list):
                    raise APIError("Search response is missing searchResults", True)
                summary["reported_total_matches"] = response.get("total")
                added = 0
                for prop in results:
                    listing_key = prop.get("listingKey")
                    if not listing_key or listing_key in seen:
                        continue
                    seen.add(listing_key)
                    records.append({"property": prop, "property_fetched_at": utc_now()})
                    added += 1
                    if len(records) >= args.limit:
                        break
                export(args.output, records, summary)
                if not response.get("nextPage") or not results:
                    break
                if not added:
                    raise APIError("Pagination produced no new listing keys", True)
                page += 1
        for index, record in enumerate(records, 1):
            listing_key = record["property"]["listingKey"]
            print("Fetching amenities {}/{}: {}".format(index, len(records), listing_key))
            try:
                response = client.get("/details/amenities", {"listingKey": listing_key})
                if not isinstance(response.get("amenities"), list):
                    raise APIError("Amenity response is missing amenities")
                record["amenity_response"] = response
                record["amenities_fetched_at"] = utc_now()
                record["amenity_status"] = "ok" if response["amenities"] else "empty_unknown"
            except APIError as exc:
                record["amenity_status"] = "error"
                summary["errors"].append({"listing_key": listing_key, "error": str(exc)})
                if exc.fatal:
                    raise
            summary["requests_attempted"] = client.requests
            export(args.output, records, summary)
        summary["complete"] = not summary["errors"]
    except (APIError, KeyboardInterrupt) as exc:
        summary["errors"].append({"error": str(exc) or "Interrupted"})
    finally:
        summary["requests_attempted"] = client.requests
        summary["properties_returned"] = len(records)
        summary["amenity_responses_received"] = sum("amenity_response" in r for r in records)
        summary["finished_at"] = utc_now()
        export(args.output, records, summary)
    print("Saved {} properties to {}. Requests attempted: {} (credits may differ).".format(
        len(records), args.output.resolve(), client.requests))
    if summary["errors"]:
        print("Some data is incomplete; see run_summary.json.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
