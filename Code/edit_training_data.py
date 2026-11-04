#!/usr/bin/env python3
"""Add extra features to the combined rent training file.

Reads Data/realty_training_combined.csv (built by download_training.py) and writes
Data/realty_training_combined_final.csv: the same rows and columns plus

    crime_pc1   overall crime level near the building (higher = more crime)
    crime_pc2   crime mix: positive = burglary / larceny / stolen property,
                negative = robbery / weapons / assault / commercial sex

Both are computed exactly as in add_crime.py (PCA of Data/crime_neighborhood.csv, then a
distance-weighted average over nearby neighborhoods); see that file's docstring for details.

    pip install pandas numpy geopandas
    python Code/edit_training_data.py
    python Code/edit_training_data.py --bandwidth 1500
"""
import argparse
from pathlib import Path

import pandas as pd

from add_crime import BANDWIDTH_M, BOUNDARY_PATH, CRIME_PATH, DATA, building_scores, crime_pca, load_boundaries

IN_PATH = DATA / "realty_training_combined.csv"
OUT_PATH = DATA / "realty_training_combined_final.csv"


def add_crime_features(df, crime_path=CRIME_PATH, boundary_path=BOUNDARY_PATH, bandwidth=BANDWIDTH_M):
    """Return df with crime_pc1 and crime_pc2 columns added (rows and order unchanged)."""
    nh_scores, loadings, explained = crime_pca(pd.read_csv(crime_path))
    print(f"crime PCA on {len(nh_scores)} neighborhoods, {len(loadings)} categories; "
          f"variance explained: PC1 {explained[0]:.0%}, PC2 {explained[1]:.0%}")
    # Scores depend only on location, so compute once per building and map back to every unit.
    buildings = df[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
    scores = building_scores(buildings, load_boundaries(boundary_path), nh_scores, bandwidth)
    out = df.drop(columns=["crime_pc1", "crime_pc2"], errors="ignore").merge(
        buildings.join(scores), on=["latitude", "longitude"], how="left")
    assert len(out) == len(df)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=IN_PATH)
    ap.add_argument("--crime", type=Path, default=CRIME_PATH)
    ap.add_argument("--boundaries", type=Path, default=BOUNDARY_PATH)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--bandwidth", type=float, default=BANDWIDTH_M, help="meters (default %(default)s)")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    out = add_crime_features(df, args.crime, args.boundaries, args.bandwidth)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(out)} rows, bandwidth {args.bandwidth:g} m)")
    print(out[["crime_pc1", "crime_pc2"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
