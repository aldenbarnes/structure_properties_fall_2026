"""Parse an AppFolio commercial rent roll export into one row per unit.

The export is a grouped report, not a flat table:

    "-> 100 Green Street - Comm - Unit 1 (2,900 Sq. Ft.)",,,...   <- unit header
    "Circleback AI, Inc.",2,900,Commercial Rent,08/01/2026,...     <- charge rows
    ,,,,,"32,408.95",...                                           <- unit subtotal
    Total,,,,,...                                                  <- grand total

Steps:
  1. Flatten to one row per charge, tagged with its building / unit / sqft.
  2. Keep base-rent charges only (RENT_GL_ACCOUNTS): no parking, RUBS, bond
     passthrough, SmartEnsure, storage, concessions, CAM, etc.
  3. Keep charges in effect on the as-of date (start <= as_of, end blank or
     >= as_of), which drops future rent steps.
  4. Collapse to one row per unit: monthly rent = sum of its active rent
     charges (tenant portion + Subsidy Rent, so subsidised units show the full
     contract rent). Rent start = latest start date of the unit's Residential /
     Commercial Rent charge.
  5. Keep units whose rent start falls within the year before the as-of date.

Usage:
    python rent_roll_parser.py rent_roll_commercial-20261014.csv
    python rent_roll_parser.py rent_roll.csv --as-of 2026-10-14 --out ../Data/recent_rents.csv
"""
import argparse
import csv
import re

import pandas as pd

RENT_GL_ACCOUNTS = ['Residential Rent', 'Commercial Rent', 'Commercial Rent (Holdover)', 'Subsidy Rent']
PRIMARY_RENT_GL_ACCOUNTS = ['Residential Rent', 'Commercial Rent', 'Commercial Rent (Holdover)']

HEADER_RE = re.compile(r'^->\s*(?P<unit_label>.*?)(?:\s*\((?P<sqft>[\d,]+) Sq\. Ft\.\))?\s*$')


def _number(value):
    value = (value or '').replace(',', '').strip()
    return float(value) if value else None


def parse_rent_roll(path):
    """Flatten the grouped export into one row per charge."""
    rows = []
    unit_label = unit_sqft = None
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        columns = next(reader)
        for record in reader:
            if not record or not any(cell.strip() for cell in record):
                continue
            first = record[0].strip()
            header = HEADER_RE.match(first)
            if header:
                unit_label = header.group('unit_label')
                unit_sqft = _number(header.group('sqft'))
                continue
            # Subtotal / vacant-unit / grand-total rows have no GL account.
            if not record[2].strip():
                continue
            rows.append(dict(zip(columns, record), unit_label=unit_label, unit_sqft=unit_sqft))

    charges = pd.DataFrame(rows)
    split = charges['unit_label'].str.split(' - Unit ', n=1, expand=True, regex=False)
    charges['property'] = split[0].str.strip()
    charges['unit'] = split[1].str.strip() if 1 in split else None
    for col in ['Start Date', 'End Date']:
        charges[col] = pd.to_datetime(charges[col], format='%m/%d/%Y', errors='coerce')
    charges['Amount As Monthly'] = charges['Amount As Monthly'].map(_number)
    return charges


def recent_rents_by_unit(charges, as_of, lookback_days=365):
    """One row per unit with a base rent that started within the lookback window."""
    as_of = pd.Timestamp(as_of)
    window_start = as_of - pd.Timedelta(days=lookback_days)

    rent = charges[charges['GL Account'].isin(RENT_GL_ACCOUNTS)]
    active = rent[(rent['Start Date'] <= as_of)
                  & (rent['End Date'].isna() | (rent['End Date'] >= as_of))].copy()
    active['is_primary'] = active['GL Account'].isin(PRIMARY_RENT_GL_ACCOUNTS)
    active['primary_start'] = active['Start Date'].where(active['is_primary'])
    active['primary_amount'] = active['Amount As Monthly'].where(active['is_primary'], 0)
    active['subsidy_amount'] = active['Amount As Monthly'].where(~active['is_primary'], 0)

    units = (active.groupby('unit_label', sort=False)
             .agg(property=('property', 'first'),
                  unit=('unit', 'first'),
                  sqft=('unit_sqft', 'first'),
                  occupant=('Occupant Name', 'first'),
                  lease_type=('GL Account', lambda s: 'Commercial' if s.str.startswith('Commercial').any() else 'Residential'),
                  rent_start_date=('primary_start', 'max'),
                  tenant_rent=('primary_amount', 'sum'),
                  subsidy_rent=('subsidy_amount', 'sum'),
                  monthly_rent=('Amount As Monthly', 'sum'))
             .reset_index())

    units = units[(units['rent_start_date'] > window_start) & (units['rent_start_date'] <= as_of)]
    units = units[units['monthly_rent'] > 0]
    units['rent_per_sqft'] = (units['monthly_rent'] / units['sqft']).round(2)
    return units.sort_values(['property', 'unit']).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('path', help='AppFolio rent roll CSV export')
    parser.add_argument('--as-of', default=pd.Timestamp.today().strftime('%Y-%m-%d'),
                        help='date the "last year" window ends on (default: today)')
    parser.add_argument('--lookback-days', type=int, default=365)
    parser.add_argument('--out', default='recent_rents_by_unit.csv')
    args = parser.parse_args()

    charges = parse_rent_roll(args.path)
    units = recent_rents_by_unit(charges, args.as_of, args.lookback_days)
    units.to_csv(args.out, index=False)
    print(f'{charges["unit_label"].nunique()} units with charges, '
          f'{len(units)} with base rent starting in the {args.lookback_days} days to {args.as_of}')
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
