# SalesForecasting — End-to-End Sales Forecasting & Demand Intelligence System

## Contents
- `Sales_analysis.ipynb` — Complete, **already-executed** Jupyter notebook covering Tasks 1-6
  (EDA, decomposition, 3 forecasting models, segment forecasting, anomaly detection,
  clustering) with markdown explanations and all charts rendered inline.
- `train.csv` — Superstore Sales dataset (9,994 orders, Jan 2015 – Dec 2018).
- `app.py` — Streamlit dashboard (Task 7): 4 pages — Sales Overview, Forecast Explorer,
  Anomaly Report, Product Demand Segments.
- `requirements.txt` — all libraries needed to run the notebook and the app.
- `summary.pdf` — 2-page executive business report (Task 8).
- `charts/` — all 19 chart images exported as `.png`.


## Known limitations 
- Only 48 months of history are available, which is workable but thin for a seasonal model
  with a 12-month period — confidence intervals on the forecasts are correspondingly wide.
- The video game sales merge in Task 5 is a **contextual, year-level** join (there's no
  natural row-level key between the two datasets) — it's explicitly framed as a
  merging-skill exercise, not a causal analysis, in the notebook.
- Forecasts assume historical seasonal/growth patterns continue; they will not anticipate
  genuine step-changes (new competitors, supply shocks, strategy pivots) — flagged
  explicitly in the executive report's Risk & Limitation section.
