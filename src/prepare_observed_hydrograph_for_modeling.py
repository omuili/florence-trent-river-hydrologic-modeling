"""Prepare an interpolation-assisted observed hydrograph for modeling.

Purpose:
    Fill short, bounded hourly gaps in the USGS discharge record using
    time interpolation while preserving a clear distinction between
    observed and interpolated flow values.

Important use rule:
    - Measured USGS values remain the authoritative observations.
    - Interpolated values are used only for continuous hydrograph
      visualization and interval-volume screening.
    - Model-performance reporting should distinguish observed-timestamp
      metrics from interpolation-assisted diagnostics.

Inputs:
    - Complete verified-boundary MRMS rainfall series
    - Extended hourly USGS discharge series

Outputs:
    - Hourly observed/interpolated discharge CSV
    - Gap-filling and corrected water-balance summary
    - Hydrograph quality-control figure
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

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

OUTPUT_HYDROGRAPH_CSV = (
    PROCESSED_DIR
    / "usgs_02092500_observed_and_interpolated_hourly_flow_20180910_20181010.csv"
)

OUTPUT_SUMMARY = (
    RESULTS_DIR
    / "observed_hydrograph_gap_filling_and_water_balance_summary.txt"
)

OUTPUT_FIGURE = (
    FIGURES_DIR
    / "observed_hydrograph_gap_filling_quality_control.png"
)

SITE_NAME = "Trent River near Trenton, NC"
SITE_NUMBER = "02092500"

DRAINAGE_AREA_SQMI = 168.0
SQUARE_FEET_PER_SQUARE_MILE = 5280.0 ** 2
SECONDS_PER_HOUR = 3600.0

WINDOW_START = pd.Timestamp("2018-09-10 00:00:00", tz="America/New_York")
WINDOW_END = pd.Timestamp("2018-10-11 00:00:00", tz="America/New_York")

EARLY_RESPONSE_END = pd.Timestamp("2018-09-23 00:00:00", tz="America/New_York")

BASELINE_START = pd.Timestamp("2018-09-10 00:00:00", tz="America/New_York")
BASELINE_END = pd.Timestamp("2018-09-12 12:00:00", tz="America/New_York")

# Maximum consecutive missing hourly discharge values permitted for interpolation.
MAX_MISSING_RUN_HOURS = 6


def create_directories() -> None:
    """Create output folders."""
    for directory in [PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_rainfall() -> pd.DataFrame:
    """Load complete verified-boundary MRMS rainfall series."""
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

    rainfall = rainfall.sort_values("time_local").drop_duplicates("time_local")

    expected_rain_hours = int((WINDOW_END - WINDOW_START).total_seconds() / 3600)

    if len(rainfall) != expected_rain_hours:
        raise ValueError(
            f"Expected {expected_rain_hours} rainfall hours, found {len(rainfall)}."
        )

    if rainfall["rainfall_inches"].isna().any():
        raise ValueError("Rainfall input contains missing values.")

    return rainfall


def load_streamflow_on_complete_grid() -> pd.DataFrame:
    """Load hourly discharge on an inclusive endpoint grid for integration."""
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
        & (streamflow["time_local"] <= WINDOW_END)
    ][["time_local", "discharge_cfs"]].copy()

    streamflow = streamflow.sort_values("time_local").drop_duplicates("time_local")

    complete_times = pd.date_range(
        start=WINDOW_START,
        end=WINDOW_END,
        freq="1h",
        inclusive="both",
    )

    complete = pd.DataFrame({"time_local": complete_times})
    complete = complete.merge(streamflow, on="time_local", how="left")

    complete = complete.rename(columns={"discharge_cfs": "discharge_cfs_observed"})

    return complete


def identify_missing_runs(flow: pd.DataFrame) -> pd.DataFrame:
    """Identify contiguous missing-discharge runs and their bounding values."""
    missing = flow["discharge_cfs_observed"].isna()

    if not missing.any():
        return pd.DataFrame(
            columns=[
                "gap_start",
                "gap_end",
                "missing_hours",
                "bounded_before",
                "bounded_after",
                "eligible_for_interpolation",
            ]
        )

    groups = missing.ne(missing.shift()).cumsum()
    records: list[dict] = []

    for _, group in flow[missing].groupby(groups[missing]):
        first_index = int(group.index.min())
        last_index = int(group.index.max())

        bounded_before = (
            first_index > 0
            and pd.notna(flow.loc[first_index - 1, "discharge_cfs_observed"])
        )
        bounded_after = (
            last_index < len(flow) - 1
            and pd.notna(flow.loc[last_index + 1, "discharge_cfs_observed"])
        )

        missing_hours = len(group)
        eligible = (
            bounded_before
            and bounded_after
            and missing_hours <= MAX_MISSING_RUN_HOURS
        )

        records.append(
            {
                "gap_start": group["time_local"].min(),
                "gap_end": group["time_local"].max(),
                "missing_hours": missing_hours,
                "bounded_before": bounded_before,
                "bounded_after": bounded_after,
                "eligible_for_interpolation": eligible,
            }
        )

    return pd.DataFrame(records)


def interpolate_short_bounded_gaps(
    flow: pd.DataFrame,
    gaps: pd.DataFrame,
) -> pd.DataFrame:
    """Linearly interpolate only short internal discharge gaps."""
    if not gaps.empty and not gaps["eligible_for_interpolation"].all():
        unacceptable = gaps[~gaps["eligible_for_interpolation"]]
        raise ValueError(
            "One or more discharge gaps are unbounded or exceed the "
            f"{MAX_MISSING_RUN_HOURS}-hour interpolation rule:\n{unacceptable}"
        )

    completed = flow.copy()
    completed = completed.set_index("time_local")

    completed["discharge_cfs_for_screening"] = completed[
        "discharge_cfs_observed"
    ].interpolate(method="time", limit_area="inside")

    completed["flow_value_status"] = np.where(
        completed["discharge_cfs_observed"].notna(),
        "observed",
        "interpolated_short_bounded_gap",
    )

    completed = completed.reset_index()

    if completed["discharge_cfs_for_screening"].isna().any():
        raise ValueError(
            "Discharge still contains missing values after interpolation."
        )

    return completed


def trapezoidal_volume_cuft(flow_values: np.ndarray) -> float:
    """Calculate volume from hourly discharge values using trapezoidal integration."""
    return float(np.trapz(flow_values, dx=SECONDS_PER_HOUR))


def calculate_water_balance(
    rainfall: pd.DataFrame,
    flow: pd.DataFrame,
) -> dict:
    """Calculate corrected rainfall and interpolation-assisted flow diagnostics."""
    observed_baseline = flow[
        (flow["time_local"] >= BASELINE_START)
        & (flow["time_local"] < BASELINE_END)
        & (flow["discharge_cfs_observed"].notna())
    ]

    baseline_median_cfs = float(
        observed_baseline["discharge_cfs_observed"].median()
    )

    total_rainfall_in = float(rainfall["rainfall_inches"].sum())

    early_rainfall_in = float(
        rainfall.loc[
            rainfall["time_local"] < EARLY_RESPONSE_END,
            "rainfall_inches",
        ].sum()
    )

    later_rainfall_in = float(
        rainfall.loc[
            rainfall["time_local"] >= EARLY_RESPONSE_END,
            "rainfall_inches",
        ].sum()
    )

    screening_flow = flow["discharge_cfs_for_screening"].to_numpy()

    total_volume_cuft = trapezoidal_volume_cuft(screening_flow)

    above_baseline_flow = np.maximum(screening_flow - baseline_median_cfs, 0.0)
    above_baseline_volume_cuft = trapezoidal_volume_cuft(above_baseline_flow)

    basin_area_sqft = DRAINAGE_AREA_SQMI * SQUARE_FEET_PER_SQUARE_MILE

    total_flow_depth_in = (total_volume_cuft / basin_area_sqft) * 12.0
    above_baseline_depth_in = (
        above_baseline_volume_cuft / basin_area_sqft
    ) * 12.0

    screening_runoff_ratio = above_baseline_depth_in / total_rainfall_in

    observed_peak = flow.loc[flow["discharge_cfs_observed"].idxmax()]
    endpoint = flow.iloc[-1]

    return {
        "baseline_median_cfs": baseline_median_cfs,
        "total_rainfall_in": total_rainfall_in,
        "early_rainfall_in": early_rainfall_in,
        "later_rainfall_in": later_rainfall_in,
        "observed_peak_cfs": float(observed_peak["discharge_cfs_observed"]),
        "observed_peak_time": observed_peak["time_local"],
        "endpoint_flow_cfs": float(endpoint["discharge_cfs_for_screening"]),
        "endpoint_to_baseline_ratio": float(
            endpoint["discharge_cfs_for_screening"] / baseline_median_cfs
        ),
        "total_flow_depth_in": float(total_flow_depth_in),
        "above_baseline_depth_in": float(above_baseline_depth_in),
        "screening_runoff_ratio": float(screening_runoff_ratio),
    }


def save_hydrograph(flow: pd.DataFrame) -> None:
    """Save observed and interpolation-assisted flow series."""
    flow.to_csv(OUTPUT_HYDROGRAPH_CSV, index=False)
    print(f"Saved modeling hydrograph CSV: {OUTPUT_HYDROGRAPH_CSV}")


def create_summary(
    rainfall: pd.DataFrame,
    flow: pd.DataFrame,
    gaps: pd.DataFrame,
    statistics: dict,
) -> None:
    """Write gap-filling and corrected screening summary."""
    observed_hours = int(flow["discharge_cfs_observed"].notna().sum())
    interpolated_hours = int(flow["discharge_cfs_observed"].isna().sum())

    gap_lines = []
    for _, row in gaps.iterrows():
        gap_lines.append(
            f"- {row['gap_start']} through {row['gap_end']}: "
            f"{int(row['missing_hours'])} interpolated hours"
        )

    gap_text = "\n".join(gap_lines) if gap_lines else "No interpolation was required."

    summary = f"""Observed Hydrograph Gap-Filling and Water-Balance Summary
=========================================================

Project:
Rainfall-to-Streamflow Modeling of Hurricane Florence Flooding
in the Trent River Watershed, North Carolina

Study site:
{SITE_NAME}

USGS station:
{SITE_NUMBER}

Analysis window
---------------
{WINDOW_START} through {WINDOW_END}, exclusive

Discharge data preparation
--------------------------
Hourly discharge values directly observed by USGS: {observed_hours}
Hourly discharge values filled by interpolation: {interpolated_hours}
Number of short bounded gaps filled: {len(gaps)}
Maximum consecutive missing-hour run allowed: {MAX_MISSING_RUN_HOURS}

Gap-filling rule
----------------
Only short internal gaps bounded by observed discharge values were filled.
Linear time interpolation was used to create an hourly screening hydrograph.
Measured USGS discharge remains the authoritative observed record; interpolated
values are intended for continuous hydrograph display and interval-volume
screening only.

Interpolated periods
--------------------
{gap_text}

Corrected rainfall totals from complete MRMS record
---------------------------------------------------
Total basin-average rainfall through October 10: {statistics["total_rainfall_in"]:.2f} inches
Rainfall through September 22: {statistics["early_rainfall_in"]:.2f} inches
Rainfall from September 23 through October 10: {statistics["later_rainfall_in"]:.2f} inches

Observed peak retained from measured USGS record
------------------------------------------------
Peak discharge: {statistics["observed_peak_cfs"]:,.0f} cubic feet per second
Peak timestamp: {statistics["observed_peak_time"]}

Interpolation-assisted water-balance screening
----------------------------------------------
Pre-event median observed discharge: {statistics["baseline_median_cfs"]:,.0f} cubic feet per second
Discharge at analysis-window endpoint: {statistics["endpoint_flow_cfs"]:,.0f} cubic feet per second
Endpoint discharge divided by pre-event median: {statistics["endpoint_to_baseline_ratio"]:.2f}
Total streamflow depth over analysis interval: {statistics["total_flow_depth_in"]:.2f} inches
Flow depth above constant pre-event median: {statistics["above_baseline_depth_in"]:.2f} inches
Screening ratio of above-baseline flow depth to rainfall depth: {statistics["screening_runoff_ratio"]:.3f}

Modeling interpretation
-----------------------
The complete MRMS rainfall record must be used as model forcing. The
interpolation-assisted observed hydrograph may be used for visual assessment
and screening water-balance calculations because all filled gaps are short
and bounded by measured observations. Model performance conclusions should
report measured-peak reproduction and observed-timestamp comparisons
separately from interpolation-assisted continuous-hydrograph diagnostics.

Because discharge remains above pre-event median flow at the pre-Michael
cutoff, the screening runoff ratio should not be described as a complete
Florence event runoff coefficient.
"""

    OUTPUT_SUMMARY.write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print(f"Saved gap-filling summary: {OUTPUT_SUMMARY}")


def create_quality_control_figure(flow: pd.DataFrame) -> None:
    """Plot observed and interpolated discharge during the gap-affected period."""
    plot_start = pd.Timestamp("2018-09-14 00:00:00", tz="America/New_York")
    plot_end = pd.Timestamp("2018-09-21 00:00:00", tz="America/New_York")

    focused = flow[
        (flow["time_local"] >= plot_start)
        & (flow["time_local"] < plot_end)
    ].copy()

    observed = focused[focused["discharge_cfs_observed"].notna()]
    interpolated = focused[focused["discharge_cfs_observed"].isna()]

    figure, axis = plt.subplots(figsize=(12, 6))

    axis.plot(
        focused["time_local"],
        focused["discharge_cfs_for_screening"],
        linewidth=1.8,
        label="Interpolation-assisted hourly hydrograph",
    )

    axis.scatter(
        observed["time_local"],
        observed["discharge_cfs_observed"],
        s=18,
        label="Observed USGS values",
        zorder=3,
    )

    axis.scatter(
        interpolated["time_local"],
        interpolated["discharge_cfs_for_screening"],
        s=18,
        marker="x",
        label="Interpolated hourly values",
        zorder=3,
    )

    axis.set_title(
        "Observed and Interpolation-Assisted Discharge During Florence Flood Response\n"
        "USGS 02092500 — Trent River near Trenton, North Carolina"
    )
    axis.set_xlabel("Date and time, Eastern Time")
    axis.set_ylabel("Discharge, cubic feet per second")
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved gap-filling quality-control figure: {OUTPUT_FIGURE}")


def main() -> None:
    """Run hydrograph preparation and corrected water-balance screening."""
    create_directories()

    rainfall = load_rainfall()
    raw_flow_grid = load_streamflow_on_complete_grid()
    gaps = identify_missing_runs(raw_flow_grid)

    if not gaps.empty:
        print("Identified discharge gaps:")
        print(gaps.to_string(index=False))
        print()

    completed_flow = interpolate_short_bounded_gaps(raw_flow_grid, gaps)
    statistics = calculate_water_balance(rainfall, completed_flow)

    save_hydrograph(completed_flow)
    create_summary(rainfall, completed_flow, gaps, statistics)
    create_quality_control_figure(completed_flow)

    print()
    print("Observed hydrograph preparation completed successfully.")


if __name__ == "__main__":
    main()