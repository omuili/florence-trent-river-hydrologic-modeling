"""Diagnose the final Florence-response modeling window.

Purpose:
    Combine final verified-boundary MRMS rainfall and extended USGS streamflow
    to determine whether the September 10 through October 10, 2018 period is
    suitable for pre-Michael Hurricane Florence response modeling.

Important limitation:
    The observed discharge record does not return to pre-event flow before
    the pre-Michael cutoff. Therefore, runoff-depth and runoff-ratio results
    are screening metrics for the available analysis interval, not complete
    event water-balance estimates.

Inputs:
    - Final verified-boundary hourly MRMS rainfall time series
    - Extended hourly USGS discharge time series

Outputs:
    - Daily rainfall and streamflow diagnostic CSV
    - Final modeling-window diagnostic summary
    - Rainfall and discharge diagnostic figure
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Inputs and outputs
# ---------------------------------------------------------------------

RAINFALL_FILE = Path(
    "data/processed/"
    "mrms_trent_river_basin_average_rainfall_20180910_20181010.csv"
)

STREAMFLOW_FILE = Path(
    "data/processed/"
    "usgs_02092500_discharge_extended_hourly_20180910_20181010.csv"
)

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
PROCESSED_DIR = Path("data/processed")

SUMMARY_FILE = RESULTS_DIR / "final_modeling_window_diagnostic_summary.txt"
DAILY_CSV = PROCESSED_DIR / "final_modeling_window_daily_rainfall_streamflow_diagnostic.csv"
FIGURE_FILE = FIGURES_DIR / "final_modeling_window_rainfall_streamflow_diagnostic.png"

SITE_NAME = "Trent River near Trenton, NC"
SITE_NUMBER = "02092500"

# Published USGS station drainage area.
DRAINAGE_AREA_SQMI = 168.0

# Key modeling-period boundaries in Eastern Time.
WINDOW_START = pd.Timestamp("2018-09-10 00:00:00", tz="America/New_York")
EARLY_RESPONSE_END = pd.Timestamp("2018-09-23 00:00:00", tz="America/New_York")
WINDOW_END = pd.Timestamp("2018-10-11 00:00:00", tz="America/New_York")

BASELINE_START = pd.Timestamp("2018-09-10 00:00:00", tz="America/New_York")
BASELINE_END = pd.Timestamp("2018-09-12 12:00:00", tz="America/New_York")

# Threshold for listing a notable post-Sep.-22 rainfall day.
DAILY_RAINFALL_PULSE_THRESHOLD_IN = 0.25


def create_directories() -> None:
    """Create required output directories."""
    for directory in [RESULTS_DIR, FIGURES_DIR, PROCESSED_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_rainfall() -> pd.DataFrame:
    """Load final verified-boundary hourly MRMS rainfall."""
    if not RAINFALL_FILE.exists():
        raise FileNotFoundError(f"Rainfall file not found: {RAINFALL_FILE}")

    rainfall = pd.read_csv(RAINFALL_FILE)

    rainfall["timestamp_local"] = pd.to_datetime(
        rainfall["timestamp_local"],
        utc=True,
    ).dt.tz_convert("America/New_York")

    rainfall["basin_mean_precip_mm"] = pd.to_numeric(
        rainfall["basin_mean_precip_mm"],
        errors="coerce",
    )

    rainfall["rainfall_inches"] = rainfall["basin_mean_precip_mm"] / 25.4

    rainfall = rainfall[
        (rainfall["timestamp_local"] >= WINDOW_START)
        & (rainfall["timestamp_local"] < WINDOW_END)
    ].copy()

    return rainfall[["timestamp_local", "rainfall_inches"]].dropna()


def load_streamflow() -> pd.DataFrame:
    """Load extended hourly USGS discharge."""
    if not STREAMFLOW_FILE.exists():
        raise FileNotFoundError(f"Streamflow file not found: {STREAMFLOW_FILE}")

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
    ].copy()

    return streamflow[["time_local", "discharge_cfs"]].dropna()


def merge_hourly_data(
    rainfall: pd.DataFrame,
    streamflow: pd.DataFrame,
) -> pd.DataFrame:
    """Combine hourly rainfall and discharge records."""
    rainfall = rainfall.rename(columns={"timestamp_local": "time_local"})

    merged = pd.merge(
        rainfall,
        streamflow,
        on="time_local",
        how="inner",
    ).sort_values("time_local")

    expected_hours = int((WINDOW_END - WINDOW_START).total_seconds() / 3600)

    if len(merged) != expected_hours:
        print(
            f"Warning: expected {expected_hours} merged hourly observations, "
            f"but found {len(merged)}."
        )

    return merged.reset_index(drop=True)


def compute_interval_statistics(hourly: pd.DataFrame) -> dict:
    """Calculate rainfall, discharge-volume, and screening runoff metrics."""
    baseline = hourly[
        (hourly["time_local"] >= BASELINE_START)
        & (hourly["time_local"] < BASELINE_END)
    ]

    baseline_median_cfs = float(baseline["discharge_cfs"].median())

    early_response = hourly[hourly["time_local"] < EARLY_RESPONSE_END]
    later_recession = hourly[hourly["time_local"] >= EARLY_RESPONSE_END]

    total_rainfall_in = float(hourly["rainfall_inches"].sum())
    early_rainfall_in = float(early_response["rainfall_inches"].sum())
    later_rainfall_in = float(later_recession["rainfall_inches"].sum())

    peak_row = hourly.loc[hourly["discharge_cfs"].idxmax()]

    basin_area_sqft = DRAINAGE_AREA_SQMI * (5280.0 ** 2)
    seconds_per_hour = 3600.0

    total_volume_cuft = float(
        (hourly["discharge_cfs"] * seconds_per_hour).sum()
    )
    total_flow_depth_in = (
        total_volume_cuft / basin_area_sqft
    ) * 12.0

    hourly_excess_cfs = np.maximum(
        hourly["discharge_cfs"] - baseline_median_cfs,
        0.0,
    )
    excess_volume_cuft = float((hourly_excess_cfs * seconds_per_hour).sum())
    excess_flow_depth_in = (
        excess_volume_cuft / basin_area_sqft
    ) * 12.0

    interval_excess_runoff_ratio = excess_flow_depth_in / total_rainfall_in

    endpoint_discharge_cfs = float(hourly.iloc[-1]["discharge_cfs"])
    endpoint_to_baseline_ratio = endpoint_discharge_cfs / baseline_median_cfs

    return {
        "baseline_median_cfs": baseline_median_cfs,
        "total_rainfall_in": total_rainfall_in,
        "early_rainfall_in": early_rainfall_in,
        "later_rainfall_in": later_rainfall_in,
        "later_rainfall_percent": (later_rainfall_in / total_rainfall_in) * 100,
        "peak_discharge_cfs": float(peak_row["discharge_cfs"]),
        "peak_timestamp": peak_row["time_local"],
        "total_flow_depth_in": float(total_flow_depth_in),
        "excess_flow_depth_in": float(excess_flow_depth_in),
        "interval_excess_runoff_ratio": float(interval_excess_runoff_ratio),
        "endpoint_discharge_cfs": endpoint_discharge_cfs,
        "endpoint_to_baseline_ratio": endpoint_to_baseline_ratio,
    }


def create_daily_diagnostic(hourly: pd.DataFrame) -> pd.DataFrame:
    """Create daily rainfall and discharge summaries."""
    daily = (
        hourly.set_index("time_local")
        .resample("1D")
        .agg(
            daily_rainfall_inches=("rainfall_inches", "sum"),
            daily_mean_discharge_cfs=("discharge_cfs", "mean"),
            daily_max_discharge_cfs=("discharge_cfs", "max"),
        )
        .reset_index()
    )

    daily["period"] = np.where(
        daily["time_local"] < EARLY_RESPONSE_END,
        "Florence rainfall and early response",
        "Post-Sep-22 recession screening",
    )

    daily.to_csv(DAILY_CSV, index=False)
    print(f"Saved daily diagnostic CSV: {DAILY_CSV}")

    return daily


def create_summary(statistics: dict, daily: pd.DataFrame) -> None:
    """Write final modeling-window diagnostic summary."""
    later_pulses = daily[
        (daily["time_local"] >= EARLY_RESPONSE_END)
        & (daily["daily_rainfall_inches"] >= DAILY_RAINFALL_PULSE_THRESHOLD_IN)
    ].copy()

    if later_pulses.empty:
        pulse_text = (
            "No post-September-22 day received at least "
            f"{DAILY_RAINFALL_PULSE_THRESHOLD_IN:.2f} inches of "
            "basin-average rainfall."
        )
    else:
        pulse_lines = []
        for _, row in later_pulses.iterrows():
            pulse_lines.append(
                f"- {row['time_local']:%Y-%m-%d}: "
                f"{row['daily_rainfall_inches']:.2f} inches"
            )
        pulse_text = "\n".join(pulse_lines)

    summary = f"""Final Modeling-Window Diagnostic Summary
========================================

Project:
Rainfall-to-Streamflow Modeling of Hurricane Florence Flooding
in the Trent River Watershed, North Carolina

Study site:
{SITE_NAME}

USGS station:
{SITE_NUMBER}

Verified modeling inputs
------------------------
Watershed boundary: USGS StreamStats verified basin
Published drainage area used for volume conversion: {DRAINAGE_AREA_SQMI:.2f} square miles
Rainfall product: MRMS GaugeCorr_QPE_01H
Analysis window: {WINDOW_START} through {WINDOW_END}, exclusive
Number of hourly timesteps analyzed: {len(daily) * 24}

Rainfall-period comparison
--------------------------
Total basin-average rainfall through October 10: {statistics["total_rainfall_in"]:.2f} inches
Rainfall through September 22: {statistics["early_rainfall_in"]:.2f} inches
Rainfall from September 23 through October 10: {statistics["later_rainfall_in"]:.2f} inches
Later-period rainfall as percentage of analysis-window total: {statistics["later_rainfall_percent"]:.2f} percent

Post-September-22 rainfall days receiving at least
{DAILY_RAINFALL_PULSE_THRESHOLD_IN:.2f} inches:
{pulse_text}

Observed discharge response
---------------------------
Pre-event median discharge: {statistics["baseline_median_cfs"]:,.0f} cubic feet per second
Peak observed discharge: {statistics["peak_discharge_cfs"]:,.0f} cubic feet per second
Peak observed discharge timestamp: {statistics["peak_timestamp"]}
Discharge at analysis-window endpoint: {statistics["endpoint_discharge_cfs"]:,.0f} cubic feet per second
Endpoint discharge divided by pre-event median: {statistics["endpoint_to_baseline_ratio"]:.2f}

Interval water-balance screening
--------------------------------
Observed total streamflow depth over analysis interval: {statistics["total_flow_depth_in"]:.2f} inches
Observed flow depth above constant pre-event median discharge: {statistics["excess_flow_depth_in"]:.2f} inches
Screening ratio of above-baseline flow depth to rainfall depth: {statistics["interval_excess_runoff_ratio"]:.3f}

Interpretation and modeling decision
------------------------------------
The September 10 through October 10 period captures the dominant Florence
rainfall forcing and the available pre-Michael river response. Because the
river does not return to pre-event baseline discharge before the cutoff,
the above-baseline streamflow depth and runoff ratio are incomplete
interval-based screening metrics rather than complete direct-runoff or
event water-balance estimates.

This interval is suitable for calibration of peak discharge, time to peak,
and hydrograph shape over the observed pre-Michael response period, provided
all rainfall pulses identified above are included in model forcing. It is
not suitable for claiming a complete Florence runoff volume or a fully
closed event water balance.
"""

    SUMMARY_FILE.write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print(f"Saved final modeling-window summary: {SUMMARY_FILE}")


def create_figure(daily: pd.DataFrame) -> None:
    """Plot daily rainfall and daily maximum discharge for the analysis interval."""
    figure, (rain_axis, flow_axis) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2]},
    )

    rain_axis.bar(
        daily["time_local"],
        daily["daily_rainfall_inches"],
        width=0.8,
    )
    rain_axis.invert_yaxis()
    rain_axis.set_ylabel("Daily rainfall\n(inches)")
    rain_axis.set_title(
        "Rainfall and Streamflow Diagnostic for Florence Response Window\n"
        "Trent River near Trenton, North Carolina"
    )
    rain_axis.grid(True, alpha=0.3)

    flow_axis.plot(
        daily["time_local"],
        daily["daily_max_discharge_cfs"],
        linewidth=2,
        marker="o",
        label="Daily maximum discharge",
    )
    flow_axis.axvline(
        EARLY_RESPONSE_END,
        linestyle="--",
        linewidth=1.5,
        label="End of early-response window",
    )
    flow_axis.set_yscale("log")
    flow_axis.set_xlabel("Date, Eastern Time")
    flow_axis.set_ylabel("Daily maximum discharge\n(cfs, logarithmic scale)")
    flow_axis.grid(True, alpha=0.3)
    flow_axis.legend()

    figure.tight_layout()
    figure.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved final modeling-window diagnostic figure: {FIGURE_FILE}")


def main() -> None:
    """Run the final modeling-window diagnostic workflow."""
    create_directories()

    rainfall = load_rainfall()
    streamflow = load_streamflow()
    hourly = merge_hourly_data(rainfall, streamflow)

    statistics = compute_interval_statistics(hourly)
    daily = create_daily_diagnostic(hourly)
    create_summary(statistics, daily)
    create_figure(daily)

    print()
    print("Final modeling-window diagnostic completed successfully.")


if __name__ == "__main__":
    main()