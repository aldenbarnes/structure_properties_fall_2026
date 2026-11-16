# Structure Properties — Fall 2026

### Code layout
- `Code/data/`: collect and clean the training data (`download_training.py`, `edit_training_data.py`, and `download_SF.ipynb` for the SF property and crime files)
- `Code/model/`: the rent model (`rent_model.py`, the saved `rent_blend.joblib`) and `ai_feedback.py`, which learns from the AI live checks
- `Code/tool/`: the Streamlit app (`rent_lookup_app.py`) and what it uses: RealtyAPI lookup, address matching, feature premiums, the AI live check and its logs
- `Code/tests/`: tests (`python -m pytest Code/tests -q`)

### SF rent lookup app
Look up an apartment on RealtyAPI, pick the unit, edit any feature, and get a rent estimate from the SF model in `Code/model/rent_model.py` (an average of ridge, SVM, KNN and gradient boosting, saved in `Code/model/rent_blend.joblib`; retrain with `python Code/data/edit_training_data.py` then `python Code/model/rent_model.py`). Underestimates count 2.2x as much as overestimates (squared error on log rent, `--under-weight`), so estimates are shifted up about 5% and about a third of held-out estimates fall below the actual rent; training reports the percent of held-out units whose estimate is $0-100, $100-250, $250-500, $500-1,000 or $1,000+ below or above the actual rent. Start typing an address ("1010 an") and pick from the dropdown of real SF addresses (from `Data/properties.csv`); the city is assumed to be San Francisco. Each address is looked up with up to three RealtyAPI calls (apartments.com address lookup, apartments.com address search, then Zillow for buildings apartments.com doesn't list), stopping at the first that finds the building; a Zillow match fills the form from the property record and shows its Rent Zestimate next to the estimate. Apartments.com listings match by street address (ranges count) or by coordinates on the same side of the street. "How it was looked up" and "RealtyAPI requests this session" show exactly what was tried. Features the model doesn't use (condition, parking, views, pets, utilities, lease term, unit and building amenities) are priced with premiums from published studies in `Code/tool/feature_adjustments.py`, shown line by line under the estimate with their sources.

```
pip install -r requirements.txt
streamlit run Code/tool/rent_lookup_app.py
```

Put the RealtyAPI key in `.streamlit/secrets.toml` as `REALTYAPI_KEY = "..."` (ignored by git), set the `REALTYAPI_KEY` environment variable, or paste it in the app's sidebar. On Streamlit Community Cloud, point the app at `Code/tool/rent_lookup_app.py` and add the key under the app's Secrets settings.

### Training data download
`Code/data/download_training.py` builds a rolling training set. Each run searches apartments.com (via RealtyAPI) for San Francisco and spends at most 1,000 requests (`--max-requests`), searches included: first on buildings never downloaded, then on buildings whose latest download is more than 60 days old (`--refresh-days`), oldest first. Buildings the cap doesn't reach keep their old rows until a later run. Each run's rows go to a new dated file in `Data/training/` (`queried_listings.csv` there records every building requested), and `Data/realty_training_combined.csv` is rebuilt from all of them with three added columns: `date_downloaded`, `days_since_download`, and `recency_weight` (0.5 at 90 days old, `--half-life`). See the docstring for details.

In Colab, open `download_tool.ipynb` (needs the `REALTYAPI_KEY` and `GITHUBTOKEN` secrets). Locally:

```
python Code/data/download_training.py --dry-run
python Code/data/download_training.py
```

### Crime features
`Code/data/edit_training_data.py` reads `Data/realty_training_combined.csv` and writes `Data/realty_training_combined_final.csv` (the combined file plus `crime_pc1`, overall crime level, and `crime_pc2`, property crime vs. street violence), plus `Data/crime_grid.csv`, the same scores on a ~100 m grid over SF. Both scores come from a PCA of `Data/crime_neighborhood.csv`, averaged over nearby neighborhoods with closer ones weighted more (the method is in the file's docstring). The first run downloads DataSF's neighborhood boundaries to `Data/sf_analysis_neighborhoods.geojson`. The rent model trains on the final file and saves the grid, so the app can look up crime for any address without geopandas. Rerun it after each `download_training.py` run, before retraining.

```
pip install pandas numpy geopandas
python Code/data/edit_training_data.py
```

### AI live check
Under every estimate, **Check against live listings** has an AI read the nearest comparable apartments.com listings: their move-in specials, fee tables, lease terms, descriptions and amenities. It rates how comparable each unit is and explains any gap between the estimate and the live market, citing the comps behind each point. Rent figures and the verdict (estimate vs the comps' net effective rent: Confirmed within ±4%, Review within ±8%, Disagrees, or Not enough comparable listings) are computed in code from the listings; anything the AI claims that the listing doesn't show (a special, a fee, a comp ID) is thrown out. One check uses up to 7 RealtyAPI requests and 2 Anthropic API calls (`Code/tool/live_check.py`).

Add `ANTHROPIC_API_KEY` next to `REALTYAPI_KEY` in `.streamlit/secrets.toml` (or the app's Secrets on Streamlit Cloud). On Streamlit Cloud also add `GITHUB_TOKEN` (contents: write), or the logs are lost when the app restarts. The logs are committed to their own `ai-audit-logs` branch so they don't redeploy the app; get them with `git fetch origin ai-audit-logs && git checkout origin/ai-audit-logs -- Data/ai_audit`.

**Logs, kept apart from the training data.** Everything goes to `Data/ai_audit/` (`Code/tool/ai_audit_log.py`, which refuses to write to `Data/training/`, the base CSVs or the crime grid): `critique_log.jsonl` (every check), `disagreement_log.jsonl` (Review and Disagrees only), `decisions.jsonl` (keep the estimate / use the live median / send to an analyst) and `ai_training_rows.csv` (every comp the AI read, in the same columns as the training data plus the AI-read fields and source labels: `data_source=ai_audit`, `ai_source_type` = `ai_disagreement` / `ai_confirmation` / `ai_insufficient_evidence`, the check ID, the AI models and the prompt version).

**Training with the AI rows** (`python Code/model/ai_feedback.py --build-view`) writes a separate file, `Data/training_views/training_with_ai.csv`: the rows of `realty_training_combined_final.csv` labeled `realtyapi_download`, plus AI rows for units the downloads don't have, at half weight, with crime scores looked up from `Data/crime_grid.csv`. `rent_model.py` reads the labels: AI rows count for `source_weight`, and every score it reports is on original rows only.

```
python Code/model/ai_feedback.py --build-view
python Code/model/rent_model.py --data Data/training_views/training_with_ai.csv --out Code/model/test_with_ai.joblib
```

**Learning from recurring findings** (`Code/model/ai_feedback.py`). When the AI keeps raising the same factor (in 8+ disagreement checks in 90 days, 65%+ of them the same way) and the comps it read measure the effect in code (|t| >= 2), the tool adjusts. Typical concessions by zip show as an expected net effective rent next to the estimate (required fees alongside). Study-based premiums (condition, features the model doesn't price) are recalibrated in `feature_adjustments.py`, citing "AI live-check findings" as a source. Amenities the model prices get more training data instead: `--compare` tests whether the AI rows improve the model on original rows, and `--retrain` switches to the retrained model only if they do (the old one is kept as `rent_blend_previous.joblib`). Adjustments are rebuilt on every run, so one the evidence stops supporting goes away.

```
python Code/model/ai_feedback.py              # report + learned adjustments (Data/ai_audit/learned_adjustments.json)
python Code/model/ai_feedback.py --compare    # does adding the AI rows help? (about 3 minutes)
python Code/model/ai_feedback.py --retrain    # retrain and switch if it does
python -m pytest Code/tests -q          # tests (no API calls)
```
