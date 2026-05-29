"""Assess whether the current Hurricane Florence modeling window captures streamflow recession.

Purpose:
    Extend observed USGS discharge beyond September 22, 2018, and determine
    whether the Trent River hydrograph returned near pre-event flow conditions
    before selecting the final rainfall-runoff modeling period.

Study site:
    USGS-02092500 - Trent River near Trenton, North Carolina

Outputs:
    - Extended observed discharge CSV
    - Streamflow recession-window assessment summary
    - Extended hydrograph figure
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests


SITE_ID = "USGS-02092500"
SITE_NUMBER = "02092500"
SITE_NAME = "Trent River near Trenton, NC"
PARAMETER_CODE = "00060"

# Extended screening period: Sep. 10 through Oct. 10, Eastern Time.
START_UTC = "2018-09-10T04:00:00Z"
END_UTC = "2018-10-11T04:00:00Z"

API_URL = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/continuous/items"

RAW_DIR = Path("data/raw/streamflow")
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

RAW_FILE = RAW_DIR / "usgs_02092500_discharge_extended_20180910_20181010.json"
PROCESSED_FILE = (
    PROCESSED_DIR
    / "usgs_02092500_discharge_extended_hourly_20180910_20181010.csv"
)
SUMMARY_FILE = RESULTS_DIR / "streamflow_recession_window_assessment.txt"
FIGURE_FILE = FIGURES_DIR / "extended_streamflow_recession_assessment.png"

# Existing candidate window currently used for rainfall processing.
CURRENT_WINDOW_END = pd.Timestamp("2018-09-23 00:00:00", tz="America/New_York")

# Baseline period selected before substantial Florence rainfall began.
BASELINE_START = pd.Timestamp("2018-09-10 00:00:00", tz="America/New_York")
BASELINE_END = pd.Timestamp("2018-09-12 12:00:00", tz="America/New_York")


def create_directories() -> None:
    """Create output folders if required."""
    for directory in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def download_extended_streamflow() -> dict:
    """Download the extended observed discharge record from USGS."""
    params = {
        "f": "json",
        "monitoring_location_id": SITE_ID,
        "parameter_code": PARAMETER_CODE,
        "time": f"{START_UTC}/{END_UTC}",
        "limit": 10000,
    }

    print("Requesting extended observed streamflow data from USGS...")
    response = requests.get(API_URL, params=params, timeout=120)
    print(f"Request URL: {response.url}")
    response.raise_for_status()

    payload = response.json()
    features = payload.get("features", [])

    if not features:
        raise ValueError("USGS returned no observations for the extended period.")

    with RAW_FILE.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print(f"Downloaded {len(features):,} observations.")
    print(f"Saved raw extended streamflow file: {RAW_FILE}")

    return payload


def parse_to_hourly_streamflow(payload: dict) -> pd.DataFrame:
    """Parse USGS records and calculate hourly mean discharge."""
    records: list[dict] = []

    for feature in payload["features"]:
        properties = feature.get("properties", {})

        records.append(
            {
                "time_utc": properties.get("time"),
                "discharge_cfs": properties.get("value"),
                "approval_status": properties.get("approval_status"),
                "qualifier": properties.get("qualifier"),
            }
        )

    observations = pd.DataFrame(records)
    observations["time_utc"] = pd.to_datetime(observations["time_utc"], utc=True)
    observations["time_local"] = observations["time_utc"].dt.tz_convert(
        "America/New_York"
    )
    observations["discharge_cfs"] = pd.to_numeric(
        observations["discharge_cfs"],
        errors="coerce",
    )

    observations = (
        observations.sort_values("time_local")
        .drop_duplicates(subset="time_local")
        .dropna(subset=["discharge_cfs"])
    )

    hourly = (
        observations.set_index("time_local")["discharge_cfs"]
        .resample("1h")
        .mean()
        .reset_index()
    )

    hourly.insert(0, "site_number", SITE_NUMBER)
    hourly.insert(1, "site_name", SITE_NAME)

    hourly.to_csv(PROCESSED_FILE, index=False)
    print(f"Saved extended hourly discharge CSV: {PROCESSED_FILE}")

    return hourly


def assess_recession(hourly: pd.DataFrame) -> dict:
    """Assess whether discharge returns close to pre-event baseline."""
    baseline = hourly[
        (hourly["time_local"] >= BASELINE_START)
        & (hourly["time_local"] < BASELINE_END)
    ]

    if baseline.empty:
        raise ValueError("No streamflow observations were found in baseline period.")

    baseline_median_cfs = float(baseline["discharge_cfs"].median())
    baseline_threshold_cfs = baseline_median_cfs * 1.25

    peak_row = hourly.loc[hourly["discharge_cfs"].idxmax()]

    current_window = hourly[hourly["time_local"] < CURRENT_WINDOW_END]
    current_window_end_row = current_window.iloc[-1]

    post_peak = hourly[hourly["time_local"] > peak_row["time_local"]].copy()
    post_peak["rolling_24h_median_cfs"] = (
        post_peak["discharge_cfs"].rolling(24, min_periods=24).median()
    )

    return_candidates = post_peak[
        post_peak["rolling_24h_median_cfs"] <= baseline_threshold_cfs
    ]

    return_timestamp = None
    return_flow = None

    if not return_candidates.empty:
        return_row = return_candidates.iloc[0]
        return_timestamp = return_row["time_local"]
        return_flow = float(return_row["rolling_24h_median_cfs"])

    return {
        "baseline_median_cfs": baseline_median_cfs,
        "baseline_threshold_cfs": baseline_threshold_cfs,
        "peak_discharge_cfs": float(peak_row["discharge_cfs"]),
        "peak_timestamp": peak_row["time_local"],
        "current_window_end_discharge_cfs": float(
            current_window_end_row["discharge_cfs"]
        ),
        "current_window_end_timestamp": current_window_end_row["time_local"],
        "return_timestamp": return_timestamp,
        "return_rolling_median_cfs": return_flow,
    }


def create_summary(hourly: pd.DataFrame, assessment: dict) -> None:
    """Write a screening summary for final modeling-period selection."""
    if assessment["return_timestamp"] is not None:
        return_statement = (
            f"First post-peak time when the rolling 24-hour median discharge "
            f"was within 25 percent of pre-event median flow: "
            f"{assessment['return_timestamp']}\n"
            f"Rolling 24-hour median discharge at return point: "
            f"{assessment['return_rolling_median_cfs']:,.0f} cfs"
        )
    else:
        return_statement = (
            "The discharge record did not return within 25 percent of "
            "pre-event median flow during the extended screening period."
        )

    end_to_baseline_ratio = (
        assessment["current_window_end_discharge_cfs"]
        / assessment["baseline_median_cfs"]
    )

    summary = f"""Streamflow Recession-Window Assessment
======================================

Project:
Rainfall-to-Streamflow Modeling of Hurricane Florence Flooding
in the Trent River Watershed, North Carolina

Study site:
{SITE_NAME}

USGS monitoring location:
{SITE_ID}

Extended discharge screening period
-----------------------------------
First hourly observation: {hourly["time_local"].min()}
Last hourly observation: {hourly["time_local"].max()}
Number of hourly discharge values: {len(hourly):,}

Pre-event baseline screening
----------------------------
Baseline interval: {BASELINE_START} through {BASELINE_END}, exclusive
Median baseline discharge: {assessment["baseline_median_cfs"]:,.0f} cfs
Near-baseline threshold used for screening: {assessment["baseline_threshold_cfs"]:,.0f} cfs
Threshold definition: 125 percent of pre-event median discharge

Observed Florence peak
----------------------
Peak discharge: {assessment["peak_discharge_cfs"]:,.0f} cfs
Peak timestamp: {assessment["peak_timestamp"]}

Evaluation of current September 10-22 candidate window
------------------------------------------------------
Discharge at current candidate-window endpoint:
{assessment["current_window_end_discharge_cfs"]:,.0f} cfs at
{assessment["current_window_end_timestamp"]}

Endpoint discharge divided by baseline median:
{end_to_baseline_ratio:,.2f}

Recession screening result
--------------------------
{return_statement}

Interpretation guidance
-----------------------
If discharge at the current endpoint remains substantially greater than
pre-event baseline flow, the September 10-22 period truncates the recession
limb and should not be used for final event-volume analysis or model
calibration. The final model period should extend through a defensible
post-event recession endpoint and use precipitation covering that same period.

This screening uses a simple baseline-return criterion and is intended for
model-period selection rather than formal baseflow separation.
"""

    SUMMARY_FILE.write_text(summary, encoding="utf-8")
    print()
    print(summary)
    print(f"Saved recession assessment summary: {SUMMARY_FILE}")


def create_figure(hourly: pd.DataFrame, assessment: dict) -> None:
    """Plot extended streamflow record with baseline and candidate endpoint."""
    figure, axis = plt.subplots(figsize=(12, 6))

    axis.plot(
        hourly["time_local"],
        hourly["discharge_cfs"],
        linewidth=2,
        label="Observed hourly discharge",
    )

    axis.axhline(
        assessment["baseline_median_cfs"],
        linestyle="--",
        linewidth=1.5,
        label="Pre-event median discharge",
    )

    axis.axhline(
        assessment["baseline_threshold_cfs"],
        linestyle=":",
        linewidth=1.5,
        label="125% baseline screening threshold",
    )

    axis.axvline(
        CURRENT_WINDOW_END,
        linestyle="--",
        linewidth=1.5,
        label="Current candidate window end",
    )

    axis.scatter(
        assessment["peak_timestamp"],
        assessment["peak_discharge_cfs"],
        zorder=3,
        label="Florence discharge peak",
    )

    if assessment["return_timestamp"] is not None:
        axis.axvline(
            assessment["return_timestamp"],
            linestyle="-.",
            linewidth=1.5,
            label="Screened return near baseline",
        )

    axis.set_title(
        "Extended Streamflow Recession Screening After Hurricane Florence\n"
        "USGS 02092500 — Trent River near Trenton, North Carolina"
    )
    axis.set_xlabel("Date and time, Eastern Time")
    axis.set_ylabel("Observed discharge, cubic feet per second")
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved extended recession assessment figure: {FIGURE_FILE}")


def main() -> None:
    """Run extended streamflow screening workflow."""
    create_directories()
    payload = download_extended_streamflow()
    hourly = parse_to_hourly_streamflow(payload)
    assessment = assess_recession(hourly)
    create_summary(hourly, assessment)
    create_figure(hourly, assessment)

    print()
    print("Streamflow recession-window assessment completed successfully.")


if __name__ == "__main__":
    main()