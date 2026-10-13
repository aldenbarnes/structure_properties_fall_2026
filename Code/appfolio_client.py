"""Pull data from AppFolio into pandas / CSV.

Supports both AppFolio APIs:

* Database API (AppFolio Stack, credentials from Developer Space)
    GET https://api.appfolio.com/api/v0/<resource>
    Basic auth (client_id:client_secret) + X-AppFolio-Developer-ID header.
* Reports API (Property Manager > General Settings > Manage API Settings)
    POST https://<database>.appfolio.com/api/v2/reports/<report>.json
    Basic auth (client_id:client_secret), JSON body of filters.

Credentials are read from environment variables, never hard-coded:
    APPFOLIO_CLIENT_ID, APPFOLIO_CLIENT_SECRET    (both APIs)
    APPFOLIO_DEVELOPER_ID                         (Database API)
    APPFOLIO_DATABASE  e.g. "structureproperties" (Reports API)

Examples:
    python appfolio_client.py db properties --out ../Data/appfolio_properties.csv
    python appfolio_client.py db units --param "filters[LastUpdatedAtFrom]=2026-01-01T00:00:00Z"
    python appfolio_client.py report rent_roll --body '{"property_visibility": "active"}'
"""
import argparse
import json
import os
import time
from urllib.parse import urljoin

import pandas as pd
import requests

DATABASE_API_BASE = 'https://api.appfolio.com/api/v0/'
TIMEOUT = 60
MAX_RETRIES = 5


class AppFolioError(Exception):
    pass


def _env(name, required=True):
    value = os.environ.get(name, '').strip()
    if required and not value:
        raise AppFolioError(f'Missing environment variable {name}')
    return value or None


class AppFolioClient:
    def __init__(self, client_id=None, client_secret=None, developer_id=None,
                 database=None, session=None):
        self.client_id = client_id or _env('APPFOLIO_CLIENT_ID')
        self.client_secret = client_secret or _env('APPFOLIO_CLIENT_SECRET')
        self.developer_id = developer_id or _env('APPFOLIO_DEVELOPER_ID', required=False)
        self.database = database or _env('APPFOLIO_DATABASE', required=False)
        self.session = session or requests.Session()
        self.session.auth = (self.client_id, self.client_secret)
        self.session.headers.update({'Accept': 'application/json'})

    def __repr__(self):
        # Keep the secret out of logs and notebook output
        return f'AppFolioClient(client_id={self.client_id!r}, database={self.database!r})'

    def _request(self, method, url, **kwargs):
        """Send a request, retrying on 429 and 5xx with exponential backoff."""
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.request(method, url, timeout=TIMEOUT, **kwargs)
            except requests.Timeout:
                raise AppFolioError(f'Request to {url} timed out')
            except requests.RequestException as exc:
                raise AppFolioError(f'Request to {url} failed: {type(exc).__name__}')
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = float(resp.headers.get('Retry-After') or 2 ** (attempt + 1))
                time.sleep(wait)
                continue
            if resp.status_code == 401:
                raise AppFolioError('401 Unauthorized: check client ID / secret (and developer ID)')
            if resp.status_code == 403:
                raise AppFolioError(f'403 Forbidden: credentials lack access to {url}')
            if resp.status_code >= 400:
                raise AppFolioError(f'{resp.status_code} error from {url}: {resp.text[:300]}')
            try:
                return resp.json()
            except ValueError:
                raise AppFolioError(f'AppFolio returned invalid JSON from {url}')
        raise AppFolioError(f'Gave up on {url} after {MAX_RETRIES} retries (status {resp.status_code})')

    # ---- Database API (Developer Space) ----

    def database_api(self, resource, params=None, max_pages=None):
        """Return every record of a Database API resource (e.g. 'properties') as a list of dicts."""
        if not self.developer_id:
            raise AppFolioError('Database API needs APPFOLIO_DEVELOPER_ID')
        headers = {'X-AppFolio-Developer-ID': self.developer_id}
        url = urljoin(DATABASE_API_BASE, resource.lstrip('/'))
        rows, pages = [], 0
        while url:
            payload = self._request('GET', url, params=params, headers=headers)
            rows.extend(_records(payload))
            pages += 1
            if max_pages and pages >= max_pages:
                break
            url = _next_url(payload, url)
            params = None  # next-page links already carry the query string
        return rows

    # ---- Reports API ----

    def report(self, report_name, body=None, max_pages=None):
        """Run a v2 report (e.g. 'rent_roll') and return all rows as a list of dicts."""
        if not self.database:
            raise AppFolioError('Reports API needs APPFOLIO_DATABASE (the <name> in <name>.appfolio.com)')
        name = report_name if report_name.endswith('.json') else f'{report_name}.json'
        url = f'https://{self.database}.appfolio.com/api/v2/reports/{name}'
        payload = self._request('POST', url, json=body or {})
        rows, pages = list(_records(payload)), 1
        url = _next_url(payload, url)
        while url and not (max_pages and pages >= max_pages):
            # Filters can't be resent with next_page_url; results expire after 30 min
            payload = self._request('POST', url)
            rows.extend(_records(payload))
            pages += 1
            url = _next_url(payload, url)
        return rows


def _records(payload):
    if isinstance(payload, list):
        return payload
    for key in ('results', 'data'):
        if isinstance(payload.get(key), list):
            return payload[key]
    return [payload]


def _next_url(payload, current_url):
    if not isinstance(payload, dict):
        return None
    nxt = payload.get('next_page_url') or payload.get('next_page_path')
    return urljoin(current_url, nxt) if nxt else None


def to_dataframe(rows):
    return pd.json_normalize(rows) if rows else pd.DataFrame()


def _parse_params(pairs):
    params = {}
    for pair in pairs or []:
        key, sep, value = pair.partition('=')
        if not sep:
            raise SystemExit(f'--param must look like key=value, got {pair!r}')
        params[key] = value
    return params


def main():
    parser = argparse.ArgumentParser(description='Pull data from AppFolio.')
    sub = parser.add_subparsers(dest='api', required=True)
    db = sub.add_parser('db', help='Database API (Developer Space credentials)')
    db.add_argument('resource', help='e.g. properties, units, tenants, leases')
    db.add_argument('--param', action='append', help='query param key=value (repeatable)')
    rp = sub.add_parser('report', help='Reports API v2')
    rp.add_argument('report_name', help='e.g. rent_roll, unit_directory, tenant_directory')
    rp.add_argument('--body', default='{}', help='JSON filters for the report')
    for p in (db, rp):
        p.add_argument('--out', help='CSV path to write (default: print a preview)')
        p.add_argument('--max-pages', type=int, help='stop after N pages (for testing)')
    args = parser.parse_args()

    client = AppFolioClient()
    if args.api == 'db':
        rows = client.database_api(args.resource, _parse_params(args.param), args.max_pages)
    else:
        rows = client.report(args.report_name, json.loads(args.body), args.max_pages)

    df = to_dataframe(rows)
    print(f'Pulled {len(df)} rows, {len(df.columns)} columns')
    if args.out:
        df.to_csv(args.out, index=False)
        print(f'Wrote {args.out}')
    else:
        print(df.head(10).to_string())


if __name__ == '__main__':
    main()
