"""Audit alignment between MRMS rainfall and USGS streamflow records.

Purpose:
    Determine why rainfall totals changed after rainfall and streamflow were
    merged in the final modeling-window diagnostic.

Key principle:
    Rainfall totals must be calculated from the complete rainfall record,
    not from a merged rainfall-streamflow table that may omit hours with
    missing discharge observations.

Outputs:
    - Alignment audit summary text file
    - CSV of any hourly records lacking rainfall or discharge
    - CSV summarizing contiguous missing-discharge periods
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RAINFALL_FILE = Path(
    "data/processed/"
    "mrms_trent_river_basin_average_rainfall_20180910_20181010.csv"
)

STREAMFLOW_FILE = Path(
    "data/processed/"
    "usgs_02092500_discharge_extended_hourly_20180910_20181010.csv"
)

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")

HOURLY_AUDIT_FILE = PROCESSED_DIR / "rainfall_streamflow_alignment_hourly_audit.csv"
GAPS_FILE = PROCESSED_DIR / "streamflow_missing_periods_with_rainfall.csv"
SUMMARY_FILE = RESULTS_DIR / "rainfall_streamflow_alignment_audit_summary.txt"

WINDOW_START = pd.Timestamp("2018-09-10 00:00:00", tz="America/New_York")
WINDOW_END = pd.Timestamp("2018-10-11 00:00:00", tz="America/New_York")


def load_rainfall() -> pd.DataFrame:
    """Load the complete verified-boundary rainfall series."""
    rainfall = pd.read_csv(RAINFALL_FILE)

    rainfall["time_local"] = pd.to_datetime(
        rainfall["timestamp_local"],
        utc=True,
    ).dt.tz_convert("America/New_York")

    rainfall["rainfall_inches"] = (
        pd.to_numeric(rainfall["basin_mean_precip_mm"], errors="coerce") / 25.4
    )

    rainfall = rainfall[
        (rainfall["time_local"] >= WINDOW_START)
        & (rainfall["time_local"] < WINDOW_END)
    ][["time_local", "rainfall_inches"]].copy()

    return rainfall.sort_values("time_local").drop_duplicates("time_local")


def load_streamflow() -> pd.DataFrame:
    """Load the extended hourly discharge series."""
    streamflow = pd.read_csv(STREAMFLOW_FILE)

    streamflow["time_local"] = pd.to_datetime(
        streamflow["time_local"],
        utc=True,
    ).dt.tz_convert("America/New_York")

    streamflow["discharge_cfs"] = pd.to_numeric(
        streamflow["discharge_cfs"],
        errors="coerce",
    )

    streamflow = streamflow[
        (streamflow["time_local"] >= WINDOW_START)
        & (streamflow["time_local"] < WINDOW_END)
    ][["time_local", "discharge_cfs"]].copy()

    return streamflow.sort_values("time_local").drop_duplicates("time_local")


def build_complete_hourly_index() -> pd.DataFrame:
    """Create the required hourly timeline for the modeling window."""
    expected_times = pd.date_range(
        start=WINDOW_START,
        end=WINDOW_END,
        freq="1h",
        inclusive="left",
    )

    return pd.DataFrame({"time_local": expected_times})


def summarize_missing_periods(audit: pd.DataFrame) -> pd.DataFrame:
    """Summarize contiguous periods where discharge is missing."""
    missing = audit[audit["discharge_cfs"].isna()].copy()

    if missing.empty:
        return pd.DataFrame(
            columns=[
                "gap_start",
                "gap_end",
                "missing_hours",
                "rainfall_during_gap_inches",
                "maximum_hourly_rainfall_during_gap_inches",
            ]
        )

    missing["new_gap"] = (
        missing["time_local"].diff().ne(pd.Timedelta(hours=1))
    )
    missing["gap_id"] = missing["new_gap"].cumsum()

    gaps = (
        missing.groupby("gap_id")
        .agg(
            gap_start=("time_local", "min"),
            gap_end=("time_local", "max"),
            missing_hours=("time_local", "count"),
            rainfall_during_gap_inches=("rainfall_inches", "sum"),
            maximum_hourly_rainfall_during_gap_inches=("rainfall_inches", "max"),
        )
        .reset_index(drop=True)
    )

    return gaps


def main() -> None:
    """Run alignment audit."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    expected = build_complete_hourly_index()
    rainfall = load_rainfall()
    streamflow = load_streamflow()

    audit = (
        expected.merge(rainfall, on="time_local", how="left")
        .merge(streamflow, on="time_local", how="left")
        .sort_values("time_local")
        .reset_index(drop=True)
    )

    audit["rainfall_missing"] = audit["rainfall_inches"].isna()
    audit["streamflow_missing"] = audit["discharge_cfs"].isna()

    audit.to_csv(HOURLY_AUDIT_FILE, index=False)

    gaps = summarize_missing_periods(audit)
    gaps.to_csv(GAPS_FILE, index=False)

    expected_hours = len(expected)
    rainfall_hours = int(audit["rainfall_inches"].notna().sum())
    streamflow_hours = int(audit["discharge_cfs"].notna().sum())
    missing_rainfall_hours = int(audit["rainfall_inches"].isna().sum())
    missing_streamflow_hours = int(audit["discharge_cfs"].isna().sum())

    complete_rainfall_total = float(audit["rainfall_inches"].sum())
    rainfall_during_missing_streamflow = float(
        audit.loc[audit["streamflow_missing"], "rainfall_inches"].sum()
    )

    rainfall_available_where_flow_exists = float(
        audit.loc[audit["discharge_cfs"].notna(), "rainfall_inches"].sum()
    )

    gap_lines = []
    if gaps.empty:
        gap_lines.append("No missing discharge periods were identified.")
    else:
        for _, row in gaps.iterrows():
            gap_lines.append(
                f"- {row['gap_start']} through {row['gap_end']}: "
                f"{int(row['missing_hours'])} missing hours, "
                f"{row['rainfall_during_gap_inches']:.2f} inches of rainfall"
            )

    gap_text = "\n".join(gap_lines)

    summary = f"""Rainfall-Streamflow Alignment Audit Summary
===========================================

Project:
Rainfall-to-Streamflow Modeling of Hurricane Florence Flooding
in the Trent River Watershed, North Carolina

Audit period:
{WINDOW_START} through {WINDOW_END}, exclusive

Expected hourly timeline
------------------------
Expected number of hourly timesteps: {expected_hours}

Rainfall coverage
-----------------
Hours with MRMS rainfall values: {rainfall_hours}
Hours missing MRMS rainfall values: {missing_rainfall_hours}
Rainfall total from complete MRMS record: {complete_rainfall_total:.2f} inches

Streamflow coverage
-------------------
Hours with USGS discharge values: {streamflow_hours}
Hours missing USGS discharge values: {missing_streamflow_hours}

Effect of merging rainfall with discharge
-----------------------------------------
Rainfall occurring during hours with missing discharge: {rainfall_during_missing_streamflow:.2f} inches
Rainfall retained only where discharge exists: {rainfall_available_where_flow_exists:.2f} inches

Missing-discharge periods
-------------------------
{gap_text}

Interpretation
--------------
Rainfall totals for model forcing must be calculated from the complete
MRMS rainfall series, regardless of whether discharge observations are
missing during some hours. If missing discharge overlaps substantial
rainfall or the flood hydrograph peak, discharge-volume calculations
will require a documented gap-handling decision before calibration or
runoff-ratio analysis.
"""

    SUMMARY_FILE.write_text(summary, encoding="utf-8")

    print(summary)
    print(f"Saved hourly audit: {HOURLY_AUDIT_FILE}")
    print(f"Saved missing-period summary: {GAPS_FILE}")
    print(f"Saved audit summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()