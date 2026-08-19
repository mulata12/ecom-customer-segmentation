import pandas as pd


def make_profile(df: pd.DataFrame, cluster_col: str = "Cluster") -> pd.DataFrame:
    """Computes cluster summary metrics and assigns human-readable segment names."""
    profile = (
        df.groupby(cluster_col)
        .agg(
            Customers=("Customer ID", "count"),
            Recency_Median=("Recency", "median"),
            Frequency_Median=("Frequency", "median"),
            Monetary_Median=("Monetary", "median"),
            Recency_Mean=("Recency", "mean"),
            Frequency_Mean=("Frequency", "mean"),
            Monetary_Mean=("Monetary", "mean"),
        )
        .reset_index()
    )

    # Assign business labels based on relative median profiles
    def assign_segment_name(row):
        rec, freq, mon = row["Recency_Median"], row["Frequency_Median"], row["Monetary_Median"]
        if freq >= 5 and mon >= 1000 and rec <= 30:
            return "Champions"
        elif freq >= 3 and rec <= 60:
            return "Loyal Customers"
        elif rec > 180:
            return "At Risk / Churned"
        else:
            return "Recent / Promising"

    profile["Segment Name"] = profile.apply(assign_segment_name, axis=1)
    return profile.round(2)