import pandas as pd


def build_rfm(transactions: pd.DataFrame):
    """Aggregate valid purchase transactions into customer-level RFM."""
    required = {"Customer ID", "Invoice", "InvoiceDate", "TotalPrice"}
    missing = required - set(transactions.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    analysis_date = transactions["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        transactions.groupby("Customer ID", as_index=False)
        .agg(
            Recency=("InvoiceDate", lambda s: (analysis_date - s.max()).days),
            Frequency=("Invoice", "nunique"),
            Monetary=("TotalPrice", "sum"),
        )
    )

    if (rfm[["Recency", "Frequency", "Monetary"]] < 0).any().any():
        raise ValueError("RFM contains negative values unexpectedly.")

    return rfm, analysis_date
