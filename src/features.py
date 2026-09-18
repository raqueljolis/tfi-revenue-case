"""Feature engineering and processed-data I/O for the TFI revenue case.

Used by `02_cleaning_features`, which turns the raw frame `data_prep.load_data` returns
into the model-ready table saved to `data/processed/`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Kaggle's TFI competition launched 2015-01-01; restaurant "age" is measured
REFERENCE_DATE = pd.Timestamp("2015-01-01")


def compute_age_days(
    df: pd.DataFrame,
    date_col: str = "Open Date",
    reference_date: pd.Timestamp = REFERENCE_DATE,
) -> pd.Series:
    """Restaurant age in days at `reference_date`, used as a maturity proxy."""
    return (reference_date - df[date_col]).dt.days


def compute_age_months(
    df: pd.DataFrame,
    date_col: str = "Open Date",
    reference_date: pd.Timestamp = REFERENCE_DATE,
) -> pd.Series:
    """Restaurant age in whole calendar months at `reference_date`.
    """
    dates = df[date_col]
    months = (reference_date.year - dates.dt.year) * 12 + (reference_date.month - dates.dt.month)
    months -= (reference_date.day < dates.dt.day).astype(int)
    return months


def extract_date_parts(df: pd.DataFrame, date_col: str = "Open Date") -> pd.DataFrame:
    """Calendar components of `date_col`: year, month, day-of-month, and ISO week-of-year.
    """
    dates = df[date_col]
    return pd.DataFrame(
        {
            "open_year": dates.dt.year,
            "open_month": dates.dt.month,
            "open_day": dates.dt.day,
            "open_week": dates.dt.isocalendar().week.astype(int),
        },
        index=df.index,
    )


def frequency_encode(series: pd.Series, normalize: bool = True) -> pd.Series:
    """Map each category to its frequency in `series` (share of rows, or raw count) for `City` category.
    """
    freq = series.value_counts(normalize=normalize)
    return series.map(freq)


# Sparse P-variable blocks flagged in EDA (each >=50% zeros): P14-P18, P24-P27, P30-P37 — 17 columns total
SPARSE_P_GROUPS: dict[str, list[str]] = {
    "all_zero": [f"P{i}" for i in range(14, 19)] + [f"P{i}" for i in range(24, 28)] + [f"P{i}" for i in range(30, 38)]
}


def add_all_zero_flags(
    df: pd.DataFrame, groups: dict[str, list[str]] = SPARSE_P_GROUPS
) -> pd.DataFrame:
    """One binary column per group: 1 if every column in that group is 0 for the row.
    """
    return pd.DataFrame(
        {name: (df[cols] == 0).all(axis=1).astype(int) for name, cols in groups.items()},
        index=df.index,
    )


def bucket_rare_categories(
    series: pd.Series, min_count: int = 5, other_label: str = "Other"
) -> pd.Series:
    """Collapse categories with fewer than `min_count` occurrences into `other_label`.
    """
    counts = series.value_counts()
    rare = counts[counts < min_count].index
    return series.where(~series.isin(rare), other_label)


def flag_revenue_outliers(df: pd.DataFrame, target: str = "revenue", k: float = 1.5) -> pd.Series:
    """Boolean flag: True where `target` exceeds `Q3 + k x IQR` (fences computed on `df` itself).
    """
    q1, q3 = df[target].quantile([0.25, 0.75])
    upper_fence = q3 + k * (q3 - q1)
    return df[target] > upper_fence


def save_processed(
    df: pd.DataFrame, filename: str, processed_dir: str | Path = "../data/processed"
) -> Path:
    """Save a DataFrame to the shared processed-data directory as a CSV."""
    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    return out_path
