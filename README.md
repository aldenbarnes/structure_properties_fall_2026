# Structure Properties — Fall 2026

### Test Colab
[![Open test_colab in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aldenbarnes/structure_properties_fall_2026/blob/main/Code/test_colab.ipynb)

### SF rent lookup app
Look up an apartment on RealtyAPI, pick the unit, edit any feature, and get a rent estimate from `Code/rent_model.py`. Addresses can be typed loosely ("1010 anza", "1010 ansa st #3"): they are matched to real SF addresses in `Data/properties.csv` before searching, and the city is assumed to be San Francisco. Features the model doesn't use (condition, parking, views, pets, utilities, lease term, unit and building amenities) are priced with premiums from published studies in `Code/feature_adjustments.py`, shown line by line under the estimate with their sources.

```
pip install -r Code/requirements.txt
streamlit run Code/rent_lookup_app.py
```

Put the RealtyAPI key in `.streamlit/secrets.toml` as `REALTYAPI_KEY = "..."` (ignored by git), set the `REALTYAPI_KEY` environment variable, or paste it in the app's sidebar. On Streamlit Community Cloud, point the app at `Code/rent_lookup_app.py` and add the key under the app's Secrets settings.
