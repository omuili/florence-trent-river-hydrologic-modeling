"""Verify Florence streamflow volume using official USGS daily mean discharge.

Purpose:
    Compare interval flow depth computed from official USGS daily mean
    discharge against the interpolation-assisted hourly hydrograph.

Why this matters:
    Event runoff depth currently exceeds MRMS rainfall depth. Before any
    HEC-HMS calibration, we must determine whether that result is caused by
    interpolation of missing hourly discharge values or is also present in
    the official daily mean discharge record.

USGS parameter/statistic:
    Parameter code 00060: Discharge, cubic feet per second
    Statistic code 00003: Daily mean

Outputs:
    - Raw USGS daily response JSON
    - Processed daily mean discharge CSV
    - Volume verification summary
    - Daily mean discharge comparison figure
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


# ---------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------

SITE_ID = "USGS-02092500"
SITE_NUMBER = "02092500"
SITE_NAME = "Trent River near Trenton, NC"

PARAMETER_CODE = "00060"   # Discharge, ft3/s
STATISTIC_ID = "00003"      # Daily mean

START_DATE = "2018-09-10"
END_DATE = "2018-10-10"

API_URL = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items"

RAINFALL_FILE = Path(
    "data/processed/"
    "mrms_trent_river_basin_average_rainfall_20180910_20181010.csv"
)

INTERPOLATED_HOURLY_FLOW_FILE = Path(
    "data/processed/"
    "usgs_02092500_observed_and_interpolated_hourly_flow_20180910_20181010.csv"
)

RAW_DIR = Path("data/raw/streamflow")
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

RAW_DAILY_FILE = RAW_DIR / "usgs_02092500_daily_mean_discharge_20180910_20181010.json"
PROCESSED_DAILY_FILE = (
    PROCESSED_DIR / "usgs_02092500_daily_mean_discharge_20180910_20181010.csv"
)
SUMMARY_FILE = RESULTS_DIR / "usgs_daily_discharge_volume_verification_summary.txt"
FIGURE_FILE = FIGURES_DIR / "usgs_daily_discharge_volume_verification.png"

DRAINAGE_AREA_SQMI = 168.0
SQUARE_FEET_PER_SQUARE_MILE = 5280.0 ** 2
SECONDS_PER_DAY = 86400.0
SECONDS_PER_HOUR = 3600.0

BASELINE_START = pd.Timestamp("2018-09-10")
BASELINE_END = pd.Timestamp("2018-09-13")


def create_directories() -> None:
    """Create output folders."""
    for directory in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def download_daily_mean_discharge() -> dict:
    """Download official USGS daily mean discharge records."""
    params = {
        "f": "json",
        "monitoring_location_id": SITE_ID,
        "parameter_code": PARAMETER_CODE,
        "statistic_id": STATISTIC_ID,
        "time": f"{START_DATE}/{END_DATE}",
        "limit": 1000,
    }

    print("Requesting USGS daily mean discharge...")
    response = requests.get(API_URL, params=params, timeout=120)
    print(f"Request URL: {response.url}")
    response.raise_for_status()

    payload = response.json()
    features = payload.get("features", [])

    if not features:
        raise ValueError("USGS returned no daily mean discharge values.")

    with RAW_DAILY_FILE.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print(f"Downloaded {len(features)} daily records.")
    print(f"Saved raw daily response: {RAW_DAILY_FILE}")

    return payload


def parse_daily_mean_discharge(payload: dict) -> pd.DataFrame:
    """Parse daily mean discharge records from USGS JSON."""
    records: list[dict] = []

    for feature in payload["features"]:
        properties = feature.get("properties", {})

        records.append(
            {
                "date": properties.get("time"),
                "daily_mean_discharge_cfs": properties.get("value"),
                "unit": properties.get("unit_of_measure"),
                "approval_status": properties.get("approval_status"),
                "qualifier": properties.get("qualifier"),
                "statistic_id": properties.get("statistic_id"),
            }
        )

    daily = pd.DataFrame(records)
    daily["date"] = pd.to_datetime(daily["date"]).dt.tz_localize(None)
    daily["daily_mean_discharge_cfs"] = pd.to_numeric(
        daily["daily_mean_discharge_cfs"],
        errors="coerce",
    )

    daily = (
        daily.sort_values("date")
        .drop_duplicates(subset="date")
        .dropna(subset=["daily_mean_discharge_cfs"])
        .reset_index(drop=True)
    )

    daily.to_csv(PROCESSED_DAILY_FILE, index=False)
    print(f"Saved processed daily discharge CSV: {PROCESSED_DAILY_FILE}")

    return daily


def load_rainfall_total() -> float:
    """Load complete MRMS rainfall total for the analysis interval."""
    rainfall = pd.read_csv(RAINFALL_FILE)
    rainfall["basin_mean_precip_mm"] = pd.to_numeric(
        rainfall["basin_mean_precip_mm"],
        errors="coerce",
    )

    return float(rainfall["basin_mean_precip_mm"].sum() / 25.4)


def load_interpolated_hourly_depth(baseline_cfs: float) -> tuple[float, float]:
    """Calculate hourly interpolation-assisted flow depths for comparison."""
    hourly = pd.read_csv(INTERPOLATED_HOURLY_FLOW_FILE)

    hourly["discharge_cfs_for_screening"] = pd.to_numeric(
        hourly["discharge_cfs_for_screening"],
        errors="coerce",
    )

    flow = hourly["discharge_cfs_for_screening"].dropna().to_numpy()

    total_volume_cuft = float(np.trapz(flow, dx=SECONDS_PER_HOUR))
    above_baseline_volume_cuft = float(
        np.trapz(np.maximum(flow - baseline_cfs, 0.0), dx=SECONDS_PER_HOUR)
    )

    basin_area_sqft = DRAINAGE_AREA_SQMI * SQUARE_FEET_PER_SQUARE_MILE

    total_depth_in = (total_volume_cuft / basin_area_sqft) * 12.0
    above_baseline_depth_in = (above_baseline_volume_cuft / basin_area_sqft) * 12.0

    return total_depth_in, above_baseline_depth_in


def calculate_daily_depths(
    daily: pd.DataFrame,
    rainfall_total_in: float,
) -> dict:
    """Calculate discharge depth and runoff-ratio screening from daily means."""
    baseline = daily[
        (daily["date"] >= BASELINE_START)
        & (daily["date"] < BASELINE_END)
    ]

    baseline_mean_cfs = float(baseline["daily_mean_discharge_cfs"].mean())

    basin_area_sqft = DRAINAGE_AREA_SQMI * SQUARE_FEET_PER_SQUARE_MILE

    total_volume_cuft = float(
        (daily["daily_mean_discharge_cfs"] * SECONDS_PER_DAY).sum()
    )

    above_baseline_cfs = np.maximum(
        daily["daily_mean_discharge_cfs"] - baseline_mean_cfs,
        0.0,
    )

    above_baseline_volume_cuft = float(
        (above_baseline_cfs * SECONDS_PER_DAY).sum()
    )

    total_depth_in = (total_volume_cuft / basin_area_sqft) * 12.0
    above_baseline_depth_in = (above_baseline_volume_cuft / basin_area_sqft) * 12.0

    hourly_total_depth_in, hourly_above_baseline_depth_in = (
        load_interpolated_hourly_depth(baseline_mean_cfs)
    )

    return {
        "baseline_mean_cfs": baseline_mean_cfs,
        "rainfall_total_in": rainfall_total_in,
        "daily_total_depth_in": total_depth_in,
        "daily_above_baseline_depth_in": above_baseline_depth_in,
        "daily_screening_ratio": above_baseline_depth_in / rainfall_total_in,
        "hourly_total_depth_in": hourly_total_depth_in,
        "hourly_above_baseline_depth_in": hourly_above_baseline_depth_in,
        "hourly_screening_ratio": hourly_above_baseline_depth_in / rainfall_total_in,
    }


def create_summary(daily: pd.DataFrame, statistics: dict) -> None:
    """Write daily-discharge volume verification summary."""
    maximum_daily_row = daily.loc[daily["daily_mean_discharge_cfs"].idxmax()]

    daily_ratio = statistics["daily_screening_ratio"]
    hourly_ratio = statistics["hourly_screening_ratio"]

    if daily_ratio > 1.0:
        decision = """The official USGS daily mean discharge record also yields an
above-baseline flow depth greater than MRMS rainfall depth. Therefore, the
volume mismatch is not caused solely by interpolation of hourly discharge
gaps. This site/event should not be calibrated using runoff volume or total
hydrograph-volume objectives without further investigation of discharge
rating uncertainty, precipitation bias, and possible hydraulic complexity.
Peak discharge and timing may still be used cautiously as observed targets."""
    else:
        decision = """The official USGS daily mean discharge record yields a
physically plausible volume relative to MRMS rainfall, indicating that the
interpolation-assisted hourly hydrograph materially inflated volume. Final
volume-based analysis should use daily mean discharge or a more conservative
hourly-gap treatment rather than the interpolated hourly series."""

    approval_statuses = ", ".join(
        sorted(daily["approval_status"].dropna().astype(str).unique())
    )

    summary = f"""USGS Daily Discharge Volume Verification Summary
==============================================

Project:
Rainfall-to-Streamflow Modeling of Hurricane Florence Flooding
in the Trent River Watershed, North Carolina

Study site:
{SITE_NAME}

USGS station:
{SITE_NUMBER}

Verification objective
----------------------
Determine whether the above-rainfall streamflow-volume result persists when
using official USGS daily mean discharge rather than interpolation-assisted
hourly discharge.

Daily discharge dataset
-----------------------
Daily records downloaded: {len(daily)}
First daily record: {daily["date"].min():%Y-%m-%d}
Last daily record: {daily["date"].max():%Y-%m-%d}
Approval status values returned: {approval_statuses}
Maximum daily mean discharge: {maximum_daily_row["daily_mean_discharge_cfs"]:,.0f} cfs
Date of maximum daily mean discharge: {maximum_daily_row["date"]:%Y-%m-%d}

Rainfall reference
------------------
Complete MRMS basin-average rainfall depth: {statistics["rainfall_total_in"]:.2f} inches

Official daily mean discharge volume check
------------------------------------------
Pre-event daily mean baseline discharge: {statistics["baseline_mean_cfs"]:,.0f} cfs
Total discharge depth from daily mean values: {statistics["daily_total_depth_in"]:.2f} inches
Above-baseline discharge depth from daily mean values: {statistics["daily_above_baseline_depth_in"]:.2f} inches
Daily-data above-baseline depth / rainfall depth: {statistics["daily_screening_ratio"]:.3f}

Interpolation-assisted hourly comparison
----------------------------------------
Total discharge depth from hourly completed series: {statistics["hourly_total_depth_in"]:.2f} inches
Above-baseline discharge depth from hourly completed series: {statistics["hourly_above_baseline_depth_in"]:.2f} inches
Hourly completed-series above-baseline depth / rainfall depth: {statistics["hourly_screening_ratio"]:.3f}

Decision
--------
{decision}

Important flood-discharge limitation
------------------------------------
The USGS Hurricane Florence assessment documents the 67,700 cfs peak at
this station as an estimated peak streamflow based on a provisional
rating-curve extension, with peak stage determined from a high-water mark
identified after the event. Flood-period discharge interpretation should
therefore retain this documented uncertainty.
"""

    SUMMARY_FILE.write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print(f"Saved verification summary: {SUMMARY_FILE}")


def create_figure(daily: pd.DataFrame, rainfall_total_in: float) -> None:
    """Plot daily mean discharge during the Florence-response interval."""
    figure, axis = plt.subplots(figsize=(12, 6))

    axis.plot(
        daily["date"],
        daily["daily_mean_discharge_cfs"],
        linewidth=2,
        marker="o",
        label="Official USGS daily mean discharge",
    )

    axis.set_title(
        "Official USGS Daily Mean Discharge During Florence Response\n"
        f"Trent River near Trenton, NC — MRMS rainfall total = {rainfall_total_in:.2f} in"
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Daily mean discharge, cubic feet per second")
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved daily discharge verification figure: {FIGURE_FILE}")


def main() -> None:
    """Run official daily discharge volume verification."""
    create_directories()

    payload = download_daily_mean_discharge()
    daily = parse_daily_mean_discharge(payload)
    rainfall_total_in = load_rainfall_total()
    statistics = calculate_daily_depths(daily, rainfall_total_in)

    create_summary(daily, statistics)
    create_figure(daily, rainfall_total_in)

    print()
    print("USGS daily discharge volume verification completed successfully.")


if __name__ == "__main__":
    main()