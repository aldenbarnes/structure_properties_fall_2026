#!/usr/bin/env python3
"""Add extra features to the combined rent training file.

Reads Data/realty_training_combined.csv (built by download_training.py) and writes
Data/realty_training_combined_final.csv: the same rows and columns plus

    crime_pc1   overall crime level near the building (higher = more crime)
    crime_pc2   crime mix: positive = burglary / larceny / stolen property,
                negative = robbery / weapons / assault / commercial sex

Both come from a PCA of Data/crime_neighborhood.csv, then a distance-weighted average over
nearby neighborhoods (see "How the scores are built" below).

It also writes Data/crime_grid.csv: the same two scores on a grid over SF (every 0.001 degrees,
about 100 m). The rent model stores the grid so the app can look up crime for any address without
geopandas.

    pip install pandas numpy geopandas
    python Code/data/edit_training_data.py
    python Code/data/edit_training_data.py --bandwidth 1500

How the scores are built
  1. Crime PCA over SF's analysis neighborhoods (Data/crime_neighborhood.csv). Only neighborhoods
     with residential units are used (the parks, Presidio and Treasure Island have none, so they
     have no per-unit rate). Police paperwork (warrants, lost property, traffic collisions, ...),
     catch-all categories, crimes that are not about the street (fraud, domestic) and categories
     with under 100 incidents citywide are left out; categories SFPD records under two labels are
     merged. Each category becomes incidents per 1,000 residential units, then log(1 + rate), then
     a z-score, and the PCA runs on those.
       PC1 is the overall crime level (every category loads positively; it is nearly identical to
           log total crime per unit).
       PC2 is the mix: positive means burglary / larceny / stolen property, negative means robbery /
           weapons / assault / commercial sex. It partly tracks wealth (burglars go where the
           valuables are), so a positive rent coefficient on it is not "renters like burglary".
     Signs are set so that more crime is positive on PC1.
  2. Each building's score is a weighted average of the neighborhood scores, with weight
     exp(-(d / bandwidth)^2), where d is the distance in meters from the building to the nearest
     edge of that neighborhood (0 for the one it is in). A building deep inside a neighborhood gets
     about that neighborhood's score; one near a border gets a blend. The bandwidth (default
     1,000 m) sets how fast a neighborhood's influence fades: near 0 it is just the own-neighborhood
     score, very large it is the citywide average for everyone.

Neighborhood boundaries are DataSF's Analysis Neighborhoods (dataset j2bu-swwd), the same 41
neighborhoods as SFPD's analysis_neighborhood. They are downloaded on the first run and cached in
Data/sf_analysis_neighborhoods.geojson.
"""
import argparse
import contextlib
import io
import sys
import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "Data"
CRIME_PATH = DATA / "crime_neighborhood.csv"
BOUNDARY_PATH = DATA / "sf_analysis_neighborhoods.geojson"
BOUNDARY_URL = "https://data.sfgov.org/resource/j2bu-swwd.geojson"
IN_PATH = DATA / "realty_training_combined.csv"
OUT_PATH = DATA / "realty_training_combined_final.csv"
GRID_PATH = DATA / "crime_grid.csv"
GRID_STEP = 0.001                                   # degrees
GRID_LAT, GRID_LON = (37.700, 37.835), (-122.520, -122.355)   # covers SF's mainland


METERS_CRS = "EPSG:26910"   # UTM zone 10N, so distances come out in meters
BANDWIDTH_M = 1000

# Categories SFPD records under more than one label.
MERGE = {
    "Drug": ["Drug Offense", "Drug Violation"],
    "Weapons": ["Weapons Offense", "Weapons Offence", "Weapons Carrying Etc"],
    "Vehicle Theft": ["Motor Vehicle Theft", "Motor Vehicle Theft?"],
    "Vandalism": ["Malicious Mischief", "Vandalism"],
    "Commercial Sex": ["Prostitution", "Human Trafficking (A), Commercial Sex Acts",
                       "Human Trafficking, Commercial Sex Acts",
                       "Human Trafficking (B), Involuntary Servitude"],
}
KEEP = ["Larceny Theft", "Assault", "Burglary", "Robbery", "Disorderly Conduct", "Stolen Property", "Arson"]
MIN_INCIDENTS = 100   # citywide; rarer categories are mostly noise per neighborhood


def crime_pca(crime):
    """PC1 and PC2 score per neighborhood, plus loadings and share of variance explained."""
    c = crime[crime.residential_units > 0].set_index("analysis_neighborhood")
    counts = pd.DataFrame({k: c[v].sum(axis=1) for k, v in MERGE.items()}).join(c[KEEP])
    counts = counts.loc[:, counts.sum() >= MIN_INCIDENTS]
    z = np.log1p(counts.div(c.residential_units, axis=0) * 1000)
    z = (z - z.mean()) / z.std()
    _, s, vt = np.linalg.svd(z.values, full_matrices=False)
    loadings = pd.DataFrame(vt[:2].T, index=z.columns, columns=["pc1", "pc2"])
    loadings *= np.sign(loadings.sum())   # sign convention: more crime = positive
    explained = s[:2] ** 2 / (s ** 2).sum()
    return z @ loadings, loadings, explained


def load_boundaries(path=BOUNDARY_PATH, url=BOUNDARY_URL):
    if not path.exists():
        print(f"downloading neighborhood boundaries from {url}")
        urllib.request.urlretrieve(url, path)
    return gpd.read_file(path)[["nhood", "geometry"]]


def building_scores(buildings, boundaries, nh_scores, bandwidth=BANDWIDTH_M):
    """Distance-weighted crime_pc1 / crime_pc2 for each (latitude, longitude) row of buildings."""
    missing = set(nh_scores.index) - set(boundaries.nhood)
    if missing:
        sys.exit(f"crime neighborhoods with no boundary (names must match): {sorted(missing)}")
    nh = boundaries[boundaries.nhood.isin(nh_scores.index)].to_crs(METERS_CRS)
    pts = gpd.GeoSeries(gpd.points_from_xy(buildings.longitude, buildings.latitude),
                        crs="EPSG:4326").to_crs(METERS_CRS)
    dist = pd.DataFrame({name: pts.distance(geom).values for name, geom in zip(nh.nhood, nh.geometry)})
    outside = int((dist.min(axis=1) > 0).sum())
    if outside:
        print(f"note: {outside} buildings are not inside a neighborhood with crime data "
              "(parks, Presidio, Treasure Island or outside SF); they get a blend of the nearest ones")
    log_w = -(dist / bandwidth) ** 2
    w = np.exp(log_w.sub(log_w.max(axis=1), axis=0))   # rescaled per building so far-away ones don't underflow to 0/0
    scores = pd.DataFrame(index=buildings.index)
    for pc in ["pc1", "pc2"]:
        scores[f"crime_{pc}"] = (w * nh_scores.loc[dist.columns, pc].values).sum(axis=1).values / w.sum(axis=1).values
    return scores


def add_crime_features(df, nh_scores, boundaries, bandwidth=BANDWIDTH_M):
    """Return df with crime_pc1 and crime_pc2 columns added (rows and order unchanged)."""
    # Scores depend only on location, so compute once per building and map back to every unit.
    buildings = df[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
    scores = building_scores(buildings, boundaries, nh_scores, bandwidth)
    out = df.drop(columns=["crime_pc1", "crime_pc2"], errors="ignore").merge(
        buildings.join(scores), on=["latitude", "longitude"], how="left")
    assert len(out) == len(df)
    return out


def crime_grid(nh_scores, boundaries, bandwidth=BANDWIDTH_M):
    """crime_pc1 / crime_pc2 at every grid point (latitude, longitude) over SF."""
    lats = np.round(np.arange(GRID_LAT[0], GRID_LAT[1] + GRID_STEP / 2, GRID_STEP), 6)
    lons = np.round(np.arange(GRID_LON[0], GRID_LON[1] + GRID_STEP / 2, GRID_STEP), 6)
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    pts = pd.DataFrame({"latitude": la.ravel(), "longitude": lo.ravel()})
    with contextlib.redirect_stdout(io.StringIO()):   # grid points in the bay / parks are expected
        scores = building_scores(pts, boundaries, nh_scores, bandwidth)
    return pts.join(scores.round(5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=IN_PATH)
    ap.add_argument("--crime", type=Path, default=CRIME_PATH)
    ap.add_argument("--boundaries", type=Path, default=BOUNDARY_PATH)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--bandwidth", type=float, default=BANDWIDTH_M, help="meters (default %(default)s)")
    ap.add_argument("--grid-out", type=Path, default=GRID_PATH)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    nh_scores, loadings, explained = crime_pca(pd.read_csv(args.crime))
    print(f"crime PCA on {len(nh_scores)} neighborhoods, {len(loadings)} categories; "
          f"variance explained: PC1 {explained[0]:.0%}, PC2 {explained[1]:.0%}")
    boundaries = load_boundaries(args.boundaries)
    out = add_crime_features(df, nh_scores, boundaries, args.bandwidth)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(out)} rows, bandwidth {args.bandwidth:g} m)")
    print(out[["crime_pc1", "crime_pc2"]].describe().round(2).to_string())
    grid = crime_grid(nh_scores, boundaries, args.bandwidth)
    grid.to_csv(args.grid_out, index=False)
    print(f"wrote {args.grid_out} ({len(grid)} grid points)")


if __name__ == "__main__":
    main()
