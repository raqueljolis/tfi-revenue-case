"""Reusable data-loading and EDA helper functions for the TFI revenue case."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats

P_COLUMNS = [f"P{i}" for i in range(1, 38)]
CAT_COLUMNS = ["City", "City Group", "Type"]

# Kaggle's TFI competition launched 2015-01-01; restaurant "age" is measured
# relative to that date rather than "today", so results stay reproducible.
REFERENCE_DATE = pd.Timestamp("2015-01-01")

# Province-capital (lat, lon) for every `City` value in the TFI dataset. The raw
# data has no per-restaurant coordinates, so city centers are used as a stand-in
# for plotting restaurant locations on a map.
CITY_COORDS: dict[str, tuple[float, float]] = {
    "İstanbul": (41.0082, 28.9784),
    "Ankara": (39.9334, 32.8597),
    "İzmir": (38.4237, 27.1428),
    "Bursa": (40.1826, 29.0665),
    "Samsun": (41.2867, 36.3300),
    "Sakarya": (40.7569, 30.3781),
    "Antalya": (36.8969, 30.7133),
    "Kayseri": (38.7312, 35.4787),
    "Eskişehir": (39.7767, 30.5206),
    "Adana": (37.0000, 35.3213),
    "Diyarbakır": (37.9144, 40.2306),
    "Tekirdağ": (40.9833, 27.5167),
    "Muğla": (37.2153, 28.3636),
    "Trabzon": (41.0027, 39.7168),
    "Aydın": (37.8560, 27.8416),
    "Konya": (37.8746, 32.4932),
    "Karabük": (41.2061, 32.6204),
    "Isparta": (37.7648, 30.5566),
    "Bolu": (40.7360, 31.6061),
    "Kütahya": (39.4242, 29.9833),
    "Amasya": (40.6499, 35.8353),
    "Balıkesir": (39.6484, 27.8826),
    "Denizli": (37.7765, 29.0864),
    "Kocaeli": (40.8533, 29.8815),
    "Kırklareli": (41.7333, 27.2167),
    "Edirne": (41.6771, 26.5557),
    "Uşak": (38.6823, 29.4082),
    "Kastamonu": (41.3887, 33.7827),
    "Tokat": (40.3167, 36.5500),
    "Şanlıurfa": (37.1591, 38.7969),
    "Elazığ": (38.6810, 39.2264),
    "Gaziantep": (37.0662, 37.3833),
    "Afyonkarahisar": (38.7507, 30.5567),
    "Osmaniye": (37.0742, 36.2478),
}


def city_summary(df: pd.DataFrame, target: str = "revenue") -> pd.DataFrame:
    """Per-city restaurant count and mean target, joined to `CITY_COORDS`.

    Rows for cities missing from `CITY_COORDS` are dropped (with a printed warning)
    rather than silently mis-plotted at (NaN, NaN).
    """
    summary = (
        df.groupby("City")
        .agg(n_restaurants=("City", "size"), mean_target=(target, "mean"))
        .reset_index()
    )
    summary["lat"] = summary["City"].map(lambda c: CITY_COORDS.get(c, (None, None))[0])
    summary["lon"] = summary["City"].map(lambda c: CITY_COORDS.get(c, (None, None))[1])

    missing = summary.loc[summary["lat"].isna(), "City"].tolist()
    if missing:
        print(f"Warning: no coordinates for {missing}, dropped from map.")
    return summary.dropna(subset=["lat", "lon"])


def load_data(path: str | Path) -> pd.DataFrame:
    """Load a raw TFI CSV with Open Date parsed as a datetime."""
    return pd.read_csv(path, parse_dates=["Open Date"], date_format="%m/%d/%Y")


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of missing values per column, non-zero only."""
    n_missing = df.isna().sum()
    pct_missing = (n_missing / len(df)) * 100
    summary = pd.DataFrame({"n_missing": n_missing, "pct_missing": pct_missing})
    return summary[summary["n_missing"] > 0].sort_values("n_missing", ascending=False)


def duplicate_summary(df: pd.DataFrame, subset: list[str] | None = None) -> dict:
    """Count fully duplicated rows, and duplicates ignoring the Id column."""
    full_dupes = df.duplicated().sum()
    cols = subset or [c for c in df.columns if c != "Id"]
    content_dupes = df.duplicated(subset=cols).sum()
    return {
        "full_row_duplicates": int(full_dupes),
        "content_duplicates_ignoring_id": int(content_dupes),
    }


def compute_age_days(
    df: pd.DataFrame,
    date_col: str = "Open Date",
    reference_date: pd.Timestamp = REFERENCE_DATE,
) -> pd.Series:
    """Restaurant age in days at `reference_date`, used as a maturity proxy."""
    return (reference_date - df[date_col]).dt.days


def flag_near_constant_columns(
    df: pd.DataFrame, cols: list[str], threshold: float = 0.95
) -> list[str]:
    """Columns where a single value accounts for more than `threshold` of rows."""
    return [
        col for col in cols if df[col].value_counts(normalize=True).iloc[0] >= threshold
    ]


def flag_sparse_columns(
    df: pd.DataFrame, cols: list[str], zero_threshold: float = 0.5
) -> list[str]:
    """Columns where more than `zero_threshold` of rows are exactly zero."""
    return [col for col in cols if (df[col] == 0).mean() >= zero_threshold]


def spearman_correlation_with_target(
    df: pd.DataFrame, cols: list[str], target: str = "revenue"
) -> pd.Series:
    """Spearman rank correlation of each column with the target, sorted by |corr|."""
    corrs = {col: stats.spearmanr(df[col], df[target]).correlation for col in cols}
    return pd.Series(corrs).sort_values(key=lambda s: s.abs(), ascending=False)


def save_fig(fig, filename: str, figures_dir: str | Path = "../reports/figures") -> Path:
    """Save a matplotlib figure to the shared figures directory as a PNG."""
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return out_path


def bucket_rare_categories(
    series: pd.Series, min_count: int = 5, other_label: str = "Other"
) -> pd.Series:
    """Collapse categories with fewer than `min_count` occurrences into `other_label`.

    Used for `Type`, where `DT` has a single occurrence: leaving it as-is risks a
    train/validation split where that category is unseen on one side.
    """
    counts = series.value_counts()
    rare = counts[counts < min_count].index
    return series.where(~series.isin(rare), other_label)


def save_processed(
    df: pd.DataFrame, filename: str, processed_dir: str | Path = "../data/processed"
) -> Path:
    """Save a DataFrame to the shared processed-data directory as a CSV."""
    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    return out_path
