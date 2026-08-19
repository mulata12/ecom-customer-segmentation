import sys
from pathlib import Path

# Add project root to sys.path so python can resolve customer_segmentation
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------------------
# Configuration & Paths
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="E-Commerce Customer Segmentation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "rfm_scaler.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

# Segment Profile Mapping based on RFM cluster means
SEGMENT_MAP = {
    0: {
        "name": "Champions",
        "description": "High spenders with frequent, recent purchases. High lifetime value.",
        "badge": "🟢 Top Tier",
    },
    1: {
        "name": "At Risk / Lost",
        "description": "Purchased long ago with low frequency and low spend. Needs win-back campaign.",
        "badge": "🔴 High Churn Risk",
    },
    2: {
        "name": "Recent / New Customers",
        "description": "Bought recently with low order count. Great potential for onboarding.",
        "badge": "🔵 Emerging",
    },
    3: {
        "name": "Needs Attention",
        "description": "Moderate/high historic spend, but hasn't bought recently. Reactivate early.",
        "badge": "🟡 Slipping",
    },
}

# Import preprocessor function safely
from customer_segmentation.preprocessing import transform_rfm


# ------------------------------------------------------------------------------
# Load Cached Artifacts
# ------------------------------------------------------------------------------
@st.cache_resource
def load_pipeline_artifacts():
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        st.error(
            f"Missing required artifacts in `{MODELS_DIR}`. Please run `python run.py` first."
        )
        st.stop()

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    metadata = {}
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    return model, preprocessor, metadata


model, preprocessor, metadata = load_pipeline_artifacts()


# ------------------------------------------------------------------------------
# Sidebar - Model Info & Metadata
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Model Info")
    st.success("Artifacts Loaded Successfully")

    if metadata:
        st.write(f"**Winning Algorithm:** `{metadata.get('winning_model', 'N/A')}`")
        st.write(f"**Silhouette Score:** `{metadata.get('winning_silhouette', 0.0):.4f}`")
        st.write(f"**Optimal Clusters (k):** `{metadata.get('best_k_by_silhouette', 'N/A')}`")
        st.write(f"**Total Customers Trained:** `{metadata.get('customers', 0):,}`")
        st.caption(f"Last updated: {metadata.get('analysis_date', 'N/A')}")
    else:
        st.info("Metadata file not found.")

    st.markdown("---")
    st.markdown("### 📌 Segment Definitions")
    for key, val in SEGMENT_MAP.items():
        st.markdown(f"**Cluster {key}: {val['name']}**  \n*{val['description']}*")


# ------------------------------------------------------------------------------
# Main Application Tabs
# ------------------------------------------------------------------------------
st.title("🎯 E-Commerce Customer Segment Predictor")
st.write("Predict customer segments using RFM (Recency, Frequency, Monetary) metrics.")

tab1, tab2 = st.tabs(["👤 Single Prediction", "📁 Batch Prediction (CSV)"])


# ------------------------------------------------------------------------------
# TAB 1: Single Prediction
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Predict Segment for an Individual Customer")

    with st.form("single_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            recency = st.number_input(
                "Recency (Days)",
                min_value=0,
                max_value=1000,
                value=30,
                help="Days since last purchase",
            )
        with col2:
            frequency = st.number_input(
                "Frequency (Orders)",
                min_value=1,
                max_value=500,
                value=5,
                help="Total number of completed orders",
            )
        with col3:
            monetary = st.number_input(
                "Monetary ($ Spend)",
                min_value=0.0,
                max_value=100000.0,
                value=500.0,
                step=10.0,
                help="Total money spent across all transactions",
            )

        submit_btn = st.form_submit_button("Predict Segment", use_container_width=True)

    if submit_btn:
        # Prepare input data
        input_df = pd.DataFrame(
            [{"Recency": recency, "Frequency": frequency, "Monetary": monetary}]
        )

        # Scale & Predict
        X_scaled = transform_rfm(preprocessor, input_df)
        cluster_id = int(model.predict(X_scaled)[0])

        info = SEGMENT_MAP.get(
            cluster_id,
            {
                "name": f"Cluster {cluster_id}",
                "description": "Standard segment",
                "badge": "Unlabeled",
            },
        )

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.metric(label="Predicted Cluster ID", value=f"Cluster {cluster_id}")
            st.markdown(f"**Status:** `{info['badge']}`")

        with res_col2:
            st.subheader(f"Segment: {info['name']}")
            st.write(info["description"])


# ------------------------------------------------------------------------------
# TAB 2: Batch CSV Upload
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Batch Customer Segmentation")
    st.write("Upload a CSV file containing `Recency`, `Frequency`, and `Monetary` columns.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            required_cols = {"Recency", "Frequency", "Monetary"}

            if not required_cols.issubset(batch_df.columns):
                st.error(f"Missing required columns! Input CSV must contain: {required_cols}")
            else:
                X_batch_scaled = transform_rfm(preprocessor, batch_df[list(required_cols)])
                batch_df["Predicted_Cluster"] = model.predict(X_batch_scaled)
                batch_df["Segment_Name"] = batch_df["Predicted_Cluster"].map(
                    lambda cid: SEGMENT_MAP.get(cid, {}).get("name", f"Cluster {cid}")
                )

                st.success(f"Successfully processed {len(batch_df):,} customers!")

                st.dataframe(batch_df.head(10), use_container_width=True)

                # Download button
                csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Segmented Results CSV",
                    data=csv_bytes,
                    file_name="segmented_customers.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
