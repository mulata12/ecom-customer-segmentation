from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = {
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
}


def load_online_retail_ii(path: Path) -> pd.DataFrame:
    """Load and combine all sheets from Online Retail II."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in {".xlsx", ".xls"}:
        raise ValueError("Expected an Excel workbook.")

    sheets = pd.read_excel(path, sheet_name=None)
    if not sheets:
        raise ValueError("Workbook contains no sheets.")

    frames = []
    for name, frame in sheets.items():
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"Sheet {name!r} is missing columns: {sorted(missing)}")
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Create a purchase-only dataset suitable for RFM."""
    out = df.copy()

    out["Invoice"] = out["Invoice"].astype(str).str.strip()
    out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"], errors="coerce")
    out["Quantity"] = pd.to_numeric(out["Quantity"], errors="coerce")
    out["Price"] = pd.to_numeric(out["Price"], errors="coerce")

    valid = (
        out["Customer ID"].notna()
        & out["InvoiceDate"].notna()
        & (out["Quantity"] > 0)
        & (out["Price"] > 0)
        & ~out["Invoice"].str.startswith("C", na=False)
    )

    out = out.loc[valid].drop_duplicates().copy()
    out["Customer ID"] = out["Customer ID"].astype(str).str.strip()
    out["TotalPrice"] = out["Quantity"] * out["Price"]

    return out
