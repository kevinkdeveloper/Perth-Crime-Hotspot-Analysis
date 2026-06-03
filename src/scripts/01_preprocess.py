"""
01_preprocess.py
----------------
Data preprocessing script for Perth Crime Hotspot Analysis.

Loads the raw crime statistics CSV, cleans and standardises formats,
then saves a processed version ready for clustering and visualisation.

By default the script resolves the live public CSV resource from
data.wa.gov.au. If the public lookup fails, it falls back to the sample
repository copy in data/raw/.

Data source: WA Police Force crime statistics (data.wa.gov.au)
"""

import json
import os
from urllib.request import urlopen

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "perth_crime_statistics.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PROCESSED_PATH = os.path.join(PROCESSED_DIR, "perth_crime_processed.csv")
SUBURB_SUMMARY_PATH = os.path.join(PROCESSED_DIR, "suburb_crime_summary.csv")
DATASET_API_URL = os.getenv(
    "WA_CRIME_DATASET_API_URL",
    "https://data.wa.gov.au/api/3/action/package_show?id=crime-statistics-by-suburb",
)


def resolve_public_csv_url(dataset_api_url: str = DATASET_API_URL) -> str:
    """Resolve the live public CSV URL from the WA Open Data dataset metadata."""
    with urlopen(dataset_api_url, timeout=30) as response:
        payload = json.load(response)

    if not payload.get("success"):
        raise ValueError("WA Open Data API did not return a successful response.")

    for resource in payload.get("result", {}).get("resources", []):
        resource_url = resource.get("url")
        resource_format = str(resource.get("format", "")).lower()
        if resource_url and "csv" in resource_format:
            return resource_url

    raise ValueError("Could not find a public CSV resource for the WA crime dataset.")


def resolve_raw_data_source(local_path: str = RAW_PATH) -> str:
    """Prefer the live public CSV, while keeping the local sample as a fallback."""
    csv_url = os.getenv("WA_CRIME_CSV_URL")
    if csv_url:
        print(f"Using CSV URL from WA_CRIME_CSV_URL: {csv_url}")
        return csv_url

    try:
        csv_url = resolve_public_csv_url()
        print(f"Using public CSV URL: {csv_url}")
        return csv_url
    except Exception as exc:
        if os.path.isfile(local_path):
            print(
                "Falling back to the local sample CSV because the public dataset "
                f"could not be resolved: {exc}"
            )
            return local_path
        raise RuntimeError(
            "Could not resolve the public CSV URL and no local sample CSV was found."
        ) from exc


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw CSV file from a local path or public URL."""
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
    df_raw = load_raw_data(resolve_raw_data_source())
    df_clean = clean_data(df_raw)
    df_summary = build_suburb_summary(df_clean)
    save_outputs(df_clean, df_summary)

    print("\n--- Sample of processed data ---")
    print(df_clean.head())
    print("\n--- Top 5 suburbs by total offences ---")
    print(df_summary[["suburb", "total_offences"]].head())


if __name__ == "__main__":
    main()
