"""
01_preprocess.py
----------------
Data preprocessing script for Perth Crime Hotspot Analysis.

Loads the raw crime statistics CSV, cleans and standardises formats,
then saves a processed version ready for clustering and visualisation.

Data source: WA Police Force crime statistics (data.wa.gov.au)
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "perth_crime_statistics.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PROCESSED_PATH = os.path.join(PROCESSED_DIR, "perth_crime_processed.csv")
SUBURB_SUMMARY_PATH = os.path.join(PROCESSED_DIR, "suburb_crime_summary.csv")


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw CSV file."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows from '{path}'")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning and standardisation steps."""
    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Standardise column names: lowercase with underscores
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[\s/]+", "_", regex=True)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )

    # Ensure numeric columns are correct types
    df["offence_count"] = pd.to_numeric(df["offence_count"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Drop rows with missing critical values
    critical_cols = ["suburb", "offence_subdivision", "offence_count", "latitude", "longitude"]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with missing critical values.")

    # Remove records with zero or negative offence counts
    df = df[df["offence_count"] > 0]

    # Standardise period labels (Q1 … Q4)
    df["period"] = df["period"].str.upper()

    # Add a combined period column for ordering
    df["year_period"] = df["financial_year"] + " " + df["period"]

    print(f"Cleaned data: {len(df):,} rows remaining.")
    return df


def build_suburb_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate total offences per suburb, broken down by offence division.
    Returns a wide-format DataFrame suitable for clustering.
    """
    # Total offences per suburb per division
    pivot = (
        df.groupby(["suburb", "district", "local_government_area", "latitude", "longitude",
                    "offence_division"])["offence_count"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Flatten column names
    pivot.columns.name = None
    pivot.columns = [
        col if isinstance(col, str) else col
        for col in pivot.columns
    ]

    # Add total offences column
    offence_cols = [
        c for c in pivot.columns
        if c not in ["suburb", "district", "local_government_area", "latitude", "longitude"]
    ]
    pivot["total_offences"] = pivot[offence_cols].sum(axis=1)

    pivot = pivot.sort_values("total_offences", ascending=False).reset_index(drop=True)
    print(f"Suburb summary: {len(pivot)} suburbs, {len(offence_cols)} offence categories.")
    return pivot


def save_outputs(df_clean: pd.DataFrame, df_summary: pd.DataFrame) -> None:
    """Persist processed files to disk."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_clean.to_csv(PROCESSED_PATH, index=False)
    df_summary.to_csv(SUBURB_SUMMARY_PATH, index=False)
    print(f"Saved processed data  → {PROCESSED_PATH}")
    print(f"Saved suburb summary  → {SUBURB_SUMMARY_PATH}")


def main() -> None:
    df_raw = load_raw_data(RAW_PATH)
    df_clean = clean_data(df_raw)
    df_summary = build_suburb_summary(df_clean)
    save_outputs(df_clean, df_summary)

    print("\n--- Sample of processed data ---")
    print(df_clean.head())
    print("\n--- Top 5 suburbs by total offences ---")
    print(df_summary[["suburb", "total_offences"]].head())


if __name__ == "__main__":
    main()
