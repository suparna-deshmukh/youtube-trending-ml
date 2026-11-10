# YouTube Trending ML — Predicting Viral Content

NLP + ML pipeline over the **YouTube Trending** dataset (US subset, 48,697 trending-video rows): TF-IDF features from titles and tags, **K-Means clustering** of videos, an **XGBoost** classifier for viral content (~82 % AUC), and **SHAP** explanations of what the model actually keys on.

**Author:** Suparna Deshmukh
**Stack:** Python · Jupyter · pandas · scikit-learn · XGBoost · SHAP · SQLite
**Full report:** [YouTube_Trending_ML_Report.pdf](YouTube_Trending_ML_Report.pdf)

## The question

What separates a video that trends for a day from one that keeps pulling engagement? The project answers it in three stages — visualise the landscape, segment the population, then predict and explain.

## Pipeline (single notebook, `Notebook.ipynb`)

Every cell is commented in plain English; the notebook walks through:

| Stage | What happens | Outputs |
|-------|--------------|---------|
| 1. EDA | Trending videos by category, view-count distribution, like-to-view ratio, day-of-week patterns | `charts/01–04` |
| 2. SQL | Load the dataset into SQLite and run three queries: top categories by average views, engagement rate per category, top channels by trending days | `outputs/reports/query*.csv` |
| 3. Target | A video is `is_viral` if it appears in trending on multiple days; class balance chart; target column saved | `data/processed/USvideos_with_target.csv`, `charts/05` |
| 4. NLP features | Regex cleaning of titles/tags → TF-IDF (top 100 terms) + 11 numeric features = 111-column feature matrix | in-memory |
| 5. Clustering | Elbow method → K-Means (k=4) → named cluster profiles | `charts/06–07`, `outputs/reports/cluster_summary.csv` |
| 6. Classification | 80/20 split, `scale_pos_weight` for class imbalance, XGBoost, ROC-AUC, confusion matrix; model saved | `charts/08–09`, `outputs/models/xgboost_viral_classifier.pkl` |
| 7. Explainability | SHAP `TreeExplainer` on the saved model: top-20 feature importance + plain-English read of the top 5 | `charts/10`, `outputs/reports/shap_feature_importance.csv` |

## Results (from committed `outputs/`)

**Four video clusters** (`outputs/reports/cluster_summary.csv`):

| Cluster | Videos | Avg views | Viral rate |
|---------|-------:|----------:|-----------:|
| 💜 Engaged Community | 10,778 | 1.45 M | 97.4 % |
| 📺 Standard Trending | 29,289 | 1.65 M | 95.0 % |
| 🔥 Mainstream Hits | 808 | 31.2 M | 99.4 % |
| 🚀 Viral Giants | 74 | 99.2 M | 100 % |

**XGBoost classifier**: ~82 % ROC-AUC on the 20 % holdout with class-imbalance handling.

**SHAP** says the model leans on topical keywords first — `audio`, `ft`, `youtube`, `science`, `interview` — i.e. *what a video is about* (and its niche) drives virality more than raw engagement metrics.

## Data

Source: **[datasnaek/youtube-new](https://www.kaggle.com/datasets/datasnauk/youtube-new)** on Kaggle (free download, no API key).

| Committed | Excluded (and why) |
|-----------|--------------------|
| `data/raw/USvideos.csv` + `*_category_id.json` — the US subset this project uses | the other 10 country CSVs (≈600 MB; the notebook only uses US) |
| `data/processed/USvideos_with_target.csv` — ready-to-model frame with `is_viral` | `data/archive.zip` (duplicate of the raw bundle) |
| | `data/sqlite/` — the SQLite DB is rebuilt by notebook cell "LOAD DATAFRAME INTO SQLITE" |
| | `ml-env/` — the local virtualenv; recreate with `python -m venv ml-env` |

## How to run

```bash
python -m venv ml-env && source ml-env/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap joblib

jupyter notebook Notebook.ipynb
```

Run top to bottom. If a cell references an absolute `/Users/...` path, repoint the one `data_path` line at your copy of `data/`.

## Repository layout

```
youtube-project/
├── Notebook.ipynb                    # the full pipeline, 70 commented cells
├── charts/                           # 01–10, every figure in the report
├── data/raw/                         # USvideos.csv + category id maps
├── data/processed/                   # USvideos_with_target.csv
├── outputs/
│   ├── models/xgboost_viral_classifier.pkl
│   └── reports/                      # SQL query results, cluster summary, SHAP table
└── YouTube_Trending_ML_Report.pdf
```
