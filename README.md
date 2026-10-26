# Structure Properties — Fall 2026

### Test Colab
[![Open test_colab in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aldenbarnes/structure_properties_fall_2026/blob/main/Code/test_colab.ipynb)

### SF rent lookup app
Look up an apartment on RealtyAPI, pick the unit, edit any feature, and get a rent estimate from `Code/rent_model.py`. Start typing an address ("1010 an") and pick from the dropdown of real SF addresses (from `Data/properties.csv`); the city is assumed to be San Francisco. Each address is looked up with up to three RealtyAPI calls (apartments.com address lookup, apartments.com address search, then Zillow for buildings apartments.com doesn't list), stopping at the first that finds the building; a Zillow match fills the form from the property record and shows its Rent Zestimate next to the estimate. Apartments.com listings match by street address (ranges count) or by coordinates on the same side of the street. "How it was looked up" and "RealtyAPI requests this session" show exactly what was tried. Features the model doesn't use (condition, parking, views, pets, utilities, lease term, unit and building amenities) are priced with premiums from published studies in `Code/feature_adjustments.py`, shown line by line under the estimate with their sources.

```
pip install -r Code/requirements.txt
streamlit run Code/rent_lookup_app.py
```

Put the RealtyAPI key in `.streamlit/secrets.toml` as `REALTYAPI_KEY = "..."` (ignored by git), set the `REALTYAPI_KEY` environment variable, or paste it in the app's sidebar. On Streamlit Community Cloud, point the app at `Code/rent_lookup_app.py` and add the key under the app's Secrets settings.

### Crime features
`Code/add_crime.py` reads `Data/realty_training.csv` and writes `Data/realty_training_with_extra.csv` with two added columns, `crime_pc1` (overall crime level) and `crime_pc2` (property crime vs. street violence). Both come from a PCA of `Data/crime_neighborhood.csv`, averaged over nearby neighborhoods with closer ones weighted more (see the docstring). The first run downloads DataSF's neighborhood boundaries to `Data/sf_analysis_neighborhoods.geojson`.

```
pip install pandas numpy geopandas
python Code/add_crime.py
```
