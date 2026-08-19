import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RFM_COLUMNS = ["Recency", "Frequency", "Monetary"]


def _validate_and_extract_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """Helper function to validate input DataFrame and extract RFM columns as float."""
    missing = set(RFM_COLUMNS) - set(rfm.columns)
    if missing:
        raise ValueError(f"Missing RFM columns: {sorted(missing)}")

    values = rfm[RFM_COLUMNS].astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("RFM contains non-finite values.")
    if (values < 0).any().any():
        raise ValueError("RFM features must be non-negative.")

    return values


def fit_rfm_preprocessor(rfm: pd.DataFrame) -> StandardScaler:
    """Fit a scaler after log1p transformation."""
    values = _validate_and_extract_rfm(rfm)
    transformed = np.log1p(values)
    
    scaler = StandardScaler()
    scaler.fit(transformed)
    return scaler


def transform_rfm(scaler: StandardScaler, rfm: pd.DataFrame) -> np.ndarray:
    values = _validate_and_extract_rfm(rfm)
    transformed = np.log1p(values)
    return scaler.transform(transformed)