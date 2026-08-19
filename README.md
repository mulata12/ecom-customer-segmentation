# 🛍️ E-Commerce Customer Segmentation

**RFM feature engineering + K-Means / Gaussian Mixture / DBSCAN clustering** on ~1M real e-commerce
transactions, validated by resampling stability rather than a single metric, and shipped as an
interactive segmentation tool.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ecom-customer-segmentation-exg8qppze7u5zed6w6awog.streamlit.app/)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**🔗 Live app:** **[ecom-customer-segmentation-exg8qppze7u5zed6w6awog.streamlit.app](https://ecom-customer-segmentation-exg8qppze7u5zed6w6awog.streamlit.app/)**

---

## Overview

Online retailers rarely have a single "customer" — they have loyal repeat buyers, big one-off spenders,
lapsed customers who might still be won back, and new customers who haven't shown a pattern yet.
This project turns raw transaction logs from the **[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)**
dataset (UCI ML Repository — ~1M transactions from a UK-based online retailer, Dec 2009–Dec 2011) into
**Recency, Frequency, and Monetary (RFM)** features per customer, clusters customers on that behavior,
and turns the clusters into named, actionable segments (e.g. *Champions*, *At Risk*, *Lost / Dormant*).

Try it live: enter a customer's Recency/Frequency/Monetary values in the app and get their predicted
segment back instantly.

## Why this is more than "run KMeans and call it done"

Clustering has no ground-truth labels, so the project treats validation as the central problem rather
than a footnote:

- **k is chosen, then stress-tested.** The silhouette-selected k isn't taken on faith — it's refit 20
  times on random 80% subsamples of customers, and Adjusted Rand Index between runs measures whether the
  same partition reappears. A high mean ARI is what actually justifies calling the segmentation "real"
  rather than an artifact of one sample.
- **Three algorithms are compared, not assumed.** K-Means, Gaussian Mixture Models, and DBSCAN each make
  different structural assumptions about cluster shape and outliers. The choice of K-Means for production
  is justified against the other two, not picked by default.
- **Segments are named, not numbered.** Cluster profiles are translated into plain-language business
  labels using transparent quantile rules — no black box — so the output is something a marketing team
  can act on directly, not just an array of integers.
- **Skew is corrected before distance-based clustering.** Purchase behavior is heavily right-skewed (a
  few customers dominate spend and order count); RFM values are log-transformed and scaled so K-Means'
  Euclidean distance isn't dominated by a handful of outliers.

## Live app

**[ecom-customer-segmentation-exg8qppze7u5zed6w6awog.streamlit.app](https://ecom-customer-segmentation-exg8qppze7u5zed6w6awog.streamlit.app/)**

The app has three tabs:
| Tab | What it does |
|---|---|
| **Overview** | Dataset-wide summary: customer count, cluster sizes, average metrics per cluster, PCA projection of segments |
| **Customer prediction** | Enter Recency / Frequency / Monetary for any customer and get their predicted segment, with that segment's full profile |
| **Cluster profile** | Full table of per-segment statistics and business labels |

## Project structure

```
.                   
├── customer_segmentation/         # Core package
│   ├── data.py                    # Load raw Excel, clean transactions
│   ├── rfm.py                     # Build Recency/Frequency/Monetary features
│   ├── preprocessing.py           # log1p + scaling pipeline
│   ├── clustering.py              # K-Means / GMM / DBSCAN fitting + model selection metrics
│   ├── validation.py              # Resampling stability score (Adjusted Rand Index)
│   ├── profiling.py               # Cluster summaries + business segment labels
│   └── visualization.py           # PCA projection plot
├── app.py                         # End-to-end training pipeline (produces all artifacts below)
├── notebooks/
│   └── segmentation_analysis.ipynb  # Full methodology narrative, EDA, plots, reasoning
├── data/
│   ├── raw/                       # Source Excel file (not committed — see Data section)
│   └── processed/                 # Cleaned transactions, RFM table, RFM + cluster assignments
├── models/                        # Fitted preprocessor, KMeans, GMM, DBSCAN, run metadata
├── reports/
│   ├── figures/                   # PCA cluster plot
│   └── tables/                    # Model comparison, k-selection metrics, stability, cluster profile
├── requirements.txt
└── README.md
```

## Methodology

1. **Load & clean** — concatenate both Online Retail II sheets, drop unattributable rows (missing
   Customer ID), cancelled invoices, non-positive quantity/price, and non-product stock codes (postage,
   discounts, bank charges).
2. **Build RFM** — per customer: days since last purchase (Recency), distinct invoices (Frequency), total
   spend (Monetary), relative to a fixed analysis date (max invoice date + 1 day).
3. **Preprocess** — `log1p` transform to correct right-skew, then standard scaling, fit once and reused
   identically at inference time via a persisted `scikit-learn` pipeline.
4. **Model selection** — K-Means evaluated across k=2..10 on inertia, silhouette, Calinski-Harabasz, and
   Davies-Bouldin (no single metric is trusted alone; each has different blind spots).
5. **Model comparison** — K-Means, GMM, and DBSCAN fit at the selected k and compared on cluster count,
   silhouette, and (for DBSCAN) noise-point share.
6. **Stability validation** — 20 resampled K-Means fits, pairwise Adjusted Rand Index on overlapping
   customers, to confirm the segmentation generalizes rather than overfitting one sample.
7. **Profiling** — per-cluster RFM averages translated into named, human-readable segments.
8. **Visualization** — PCA projection of the final segmentation for a 2D sanity check.

Full reasoning for each choice — including why K-Means beat GMM/DBSCAN here and why stability was checked
before finalizing k — is documented step by step in
[`notebooks/segmentation_analysis.ipynb`](notebooks/segmentation_analysis.ipynb).

## Results

| Metric | Value |
|---|---|
| Customers segmented | see `models/metadata.json` |
| Selected k | see `models/metadata.json` |
| Mean stability (ARI, 20 resamples) | see `reports/tables/kmeans_stability.csv` |
| Segments identified | Champions, Loyal Customers, New / Low-Engagement, At Risk, Lost / Dormant |

*(Exact figures are written to `reports/tables/` and `models/metadata.json` by `run.py` — update this
table with your run's numbers before submission.)*

## Run locally

```bash
git clone https://github.com/<you>/<repo>.git
cd <repo>
pip install -r requirements.txt

# Download online_retail_II.xlsx from the UCI repository and place it in data/raw/
python run.py                 # trains models, writes reports/, figures/, models/
streamlit run app/app.py      # launches the local app
```

## Tech stack

`pandas` · `numpy` · `scikit-learn` · `scipy` · `matplotlib` / `seaborn` · `streamlit` · `joblib`

## Limitations & next steps

RFM is purely transactional — it doesn't account for product category, acquisition channel, or marketing
response history, all of which could sharpen the segments further. Stability was validated across
customer resamples, not across time, so segment membership may drift as the analysis date moves forward;
periodic retraining (rather than a one-time fit) would be the natural production setup. A logical next
step is feeding the `Cluster` label into a downstream churn or customer-lifetime-value model.

## Data source

[Online Retail II, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
— Chen, D. (2019). Not redistributed in this repository; download separately and place under `data/raw/`.

## License

MIT — see [LICENSE](LICENSE).
