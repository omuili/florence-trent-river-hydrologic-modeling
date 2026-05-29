"""Prepare final HEC-HMS time-series inputs for Nahunta Swamp modeling.

Final model site:
    USGS 02091000 - Nahunta Swamp near Shine, North Carolina

Model configuration supported by this script:
    - U.S. Customary project units
    - One-hour computation interval
    - MRMS basin-average rainfall as a Specified Hyetograph
    - USGS hourly mean discharge as observed-flow comparison series

Quality-control approach:
    - Rainfall values are complete and are retained without modification.
    - Missing hourly discharge values are filled only when each missing run
      is exactly one hour and bounded by observed values.
    - Completed hourly flow is compared against official USGS daily mean
      discharge to verify that gap filling does not distort flow volume.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------

SITE_NUMBER = "02091000"
SITE_NAME = "Nahunta Swamp near Shine, NC"

DRAINAGE_AREA_SQMI = 80.40
SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86400.0
SQUARE_FEET_PER_SQUARE_MILE = 5280.0 ** 2

WINDOW_START = pd.Timestamp("2018-09-10 00:00:00", tz="America/New_York")
WINDOW_END = pd.Timestamp("2018-10-11 00:00:00", tz="America/New_York")

BASELINE_START = pd.Timestamp("2018-09-10 00:00:00", tz="America/New_York")
BASELINE_END = pd.Timestamp("2018-09-12 12:00:00", tz="America/New_York")

OFFICIAL_INSTANTANEOUS_PEAK_CFS = 6060.0
OFFICIAL_INSTANTANEOUS_PEAK_TIME = pd.Timestamp(
    "2018-09-15 04:30:00",
    tz="America/New_York",
)

EXPECTED_HOURS = 744
MAX_INTERPOLATABLE_GAP_HOURS = 1

# ---------------------------------------------------------------------
# Input files
# ---------------------------------------------------------------------

RAINFALL_FILE = Path(
    "data/processed/final_model/nahunta_swamp/"
    "nahunta_mrms_hourly_basin_average_rainfall_20180910_20181010.csv"
)

HOURLY_FLOW_FILE = Path(
    "data/processed/final_model/nahunta_swamp/"
    "nahunta_usgs_hourly_observed_discharge_20180910_20181010.csv"
)

DAILY_FLOW_FILE = Path(
    "data/processed/site_screening/nahunta_swamp/"
    "nahunta_usgs_daily_mean_discharge_20180910_20181010.csv"
)

# ---------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------

OUTPUT_DIR = Path("data/processed/final_model/nahunta_swamp/hec_hms_inputs")
RESULTS_DIR = Path("results/final_model/nahunta_swamp")
FIGURES_DIR = Path("figures/final_model/nahunta_swamp")

PRECIPITATION_INPUT_FILE = (
    OUTPUT_DIR / "hec_hms_nahunta_hourly_precipitation_inches.csv"
)

OBSERVED_FLOW_INPUT_FILE = (
    OUTPUT_DIR / "hec_hms_nahunta_hourly_observed_flow_cfs.csv"
)

FLOW_QC_FILE = (
    OUTPUT_DIR / "nahunta_hourly_flow_observed_interpolated_qc.csv"
)

DAILY_VALIDATION_FILE = (
    OUTPUT_DIR / "nahunta_hourly_vs_official_daily_flow_validation.csv"
)

SUMMARY_FILE = RESULTS_DIR / "nahunta_hec_hms_input_preparation_summary.txt"
FIGURE_FILE = FIGURES_DIR / "nahunta_hec_hms_input_time_series_quality_control.png"


def create_directories() -> None:
    """Create required output directories."""
    for directory in [OUTPUT_DIR, RESULTS_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_rainfall() -> pd.DataFrame:
    """Load and validate complete hourly MRMS rainfall forcing."""
    if not RAINFALL_FILE.exists():
        raise FileNotFoundError(f"Rainfall file not found: {RAINFALL_FILE}")

    rainfall = pd.read_csv(RAINFALL_FILE)

    rainfall["time_local"] = pd.to_datetime(
        rainfall["timestamp_local"],
        utc=True,
    ).dt.tz_convert("America/New_York")

    rainfall["precipitation_inches"] = pd.to_numeric(
        rainfall["basin_mean_precip_inches"],
        errors="coerce",
    )

    rainfall = rainfall[
        (rainfall["time_local"] >= WINDOW_START)
        & (rainfall["time_local"] < WINDOW_END)
    ][["time_local", "precipitation_inches"]].copy()

    rainfall = rainfall.sort_values("time_local").drop_duplicates("time_local")

    if len(rainfall) != EXPECTED_HOURS:
        raise ValueError(
            f"Expected {EXPECTED_HOURS} hourly rainfall values, "
            f"found {len(rainfall)}."
        )

    if rainfall["precipitation_inches"].isna().any():
        raise ValueError("Rainfall forcing contains missing values.")

    return rainfall.reset_index(drop=True)


def load_hourly_flow() -> pd.DataFrame:
    """Load hourly observed discharge and place it on the expected hourly grid."""
    if not HOURLY_FLOW_FILE.exists():
        raise FileNotFoundError(f"Hourly flow file not found: {HOURLY_FLOW_FILE}")

    flow = pd.read_csv(HOURLY_FLOW_FILE)

    flow["time_local"] = pd.to_datetime(
        flow["time_local"],
        utc=True,
    ).dt.tz_convert("America/New_York")

    flow["hourly_mean_discharge_cfs"] = pd.to_numeric(
        flow["hourly_mean_discharge_cfs"],
        errors="coerce",
    )

    expected_times = pd.date_range(
        start=WINDOW_START,
        end=WINDOW_END,
        freq="1h",
        inclusive="left",
    )

    expected = pd.DataFrame({"time_local": expected_times})

    flow = expected.merge(
        flow[["time_local", "hourly_mean_discharge_cfs"]],
        on="time_local",
        how="left",
    )

    return flow


def identify_missing_runs(flow: pd.DataFrame) -> pd.DataFrame:
    """Identify contiguous missing-flow runs."""
    missing = flow["hourly_mean_discharge_cfs"].isna()

    if not missing.any():
        return pd.DataFrame(
            columns=["gap_start", "gap_end", "missing_hours", "bounded"]
        )

    run_id = missing.ne(missing.shift()).cumsum()
    records: list[dict] = []

    for _, group in flow[missing].groupby(run_id[missing]):
        first_index = int(group.index.min())
        last_index = int(group.index.max())

        bounded_before = (
            first_index > 0
            and pd.notna(flow.loc[first_index - 1, "hourly_mean_discharge_cfs"])
        )
        bounded_after = (
            last_index < len(flow) - 1
            and pd.notna(flow.loc[last_index + 1, "hourly_mean_discharge_cfs"])
        )

        records.append(
            {
                "gap_start": group["time_local"].min(),
                "gap_end": group["time_local"].max(),
                "missing_hours": len(group),
                "bounded": bool(bounded_before and bounded_after),
            }
        )

    return pd.DataFrame(records)


def complete_hourly_flow(
    flow: pd.DataFrame,
    gaps: pd.DataFrame,
) -> pd.DataFrame:
    """Interpolate only isolated, internally bounded one-hour gaps."""
    if not gaps.empty:
        invalid_gaps = gaps[
            (gaps["missing_hours"] > MAX_INTERPOLATABLE_GAP_HOURS)
            | (~gaps["bounded"])
        ]

        if not invalid_gaps.empty:
            raise ValueError(
                "One or more hourly flow gaps are not eligible for "
                "one-hour bounded interpolation:\n"
                f"{invalid_gaps.to_string(index=False)}"
            )

    completed = flow.copy()
    completed["flow_status"] = np.where(
        completed["hourly_mean_discharge_cfs"].notna(),
        "observed_hourly_mean",
        "interpolated_isolated_one_hour_gap",
    )

    completed = completed.set_index("time_local")
    completed["discharge_cfs_for_model_comparison"] = completed[
        "hourly_mean_discharge_cfs"
    ].interpolate(
        method="time",
        limit=MAX_INTERPOLATABLE_GAP_HOURS,
        limit_area="inside",
    )
    completed = completed.reset_index()

    if completed["discharge_cfs_for_model_comparison"].isna().any():
        raise ValueError(
            "Completed hourly flow series still contains missing values."
        )

    return completed


def load_daily_flow() -> pd.DataFrame:
    """Load approved USGS daily mean discharge used for validation."""
    if not DAILY_FLOW_FILE.exists():
        raise FileNotFoundError(f"Daily flow file not found: {DAILY_FLOW_FILE}")

    daily = pd.read_csv(DAILY_FLOW_FILE)

    daily["date"] = pd.to_datetime(daily["date"])
    daily["daily_mean_discharge_cfs"] = pd.to_numeric(
        daily["daily_mean_discharge_cfs"],
        errors="coerce",
    )

    return daily[["date", "daily_mean_discharge_cfs", "approval_status"]].dropna(
        subset=["daily_mean_discharge_cfs"]
    )


def validate_completed_flow_against_daily(
    completed_flow: pd.DataFrame,
    official_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Compare completed hourly flow aggregated daily against official daily means."""
    hourly_daily = (
        completed_flow.set_index("time_local")[
            "discharge_cfs_for_model_comparison"
        ]
        .resample("1D")
        .mean()
        .reset_index()
        .rename(
            columns={
                "time_local": "date_with_timezone",
                "discharge_cfs_for_model_comparison": "daily_mean_from_completed_hourly_cfs",
            }
        )
    )

    hourly_daily["date"] = (
        hourly_daily["date_with_timezone"].dt.tz_localize(None)
    )
    hourly_daily = hourly_daily.drop(columns="date_with_timezone")

    validation = official_daily.merge(hourly_daily, on="date", how="left")

    validation["difference_cfs"] = (
        validation["daily_mean_from_completed_hourly_cfs"]
        - validation["daily_mean_discharge_cfs"]
    )

    validation["absolute_difference_cfs"] = validation["difference_cfs"].abs()

    validation["percent_difference"] = (
        validation["difference_cfs"]
        / validation["daily_mean_discharge_cfs"]
        * 100
    )

    validation.to_csv(DAILY_VALIDATION_FILE, index=False)

    return validation


def depth_from_hourly_mean_flow(discharge_cfs: pd.Series) -> float:
    """Convert hourly mean discharge values to basin-equivalent flow depth in inches."""
    volume_cuft = float(discharge_cfs.sum() * SECONDS_PER_HOUR)
    basin_area_sqft = DRAINAGE_AREA_SQMI * SQUARE_FEET_PER_SQUARE_MILE
    return (volume_cuft / basin_area_sqft) * 12.0


def depth_from_daily_mean_flow(discharge_cfs: pd.Series) -> float:
    """Convert daily mean discharge values to basin-equivalent flow depth in inches."""
    volume_cuft = float(discharge_cfs.sum() * SECONDS_PER_DAY)
    basin_area_sqft = DRAINAGE_AREA_SQMI * SQUARE_FEET_PER_SQUARE_MILE
    return (volume_cuft / basin_area_sqft) * 12.0


def save_hec_hms_input_tables(
    rainfall: pd.DataFrame,
    completed_flow: pd.DataFrame,
) -> None:
    """Save clean U.S.-Customary time series for HEC-HMS entry."""
    precipitation_input = rainfall.copy()
    precipitation_input["date"] = precipitation_input["time_local"].dt.strftime(
        "%d%b%Y"
    ).str.upper()
    precipitation_input["time"] = precipitation_input["time_local"].dt.strftime(
        "%H:%M"
    )

    precipitation_input = precipitation_input[
        ["date", "time", "precipitation_inches"]
    ]

    precipitation_input.to_csv(PRECIPITATION_INPUT_FILE, index=False)

    observed_flow_input = completed_flow.copy()
    observed_flow_input["date"] = observed_flow_input["time_local"].dt.strftime(
        "%d%b%Y"
    ).str.upper()
    observed_flow_input["time"] = observed_flow_input["time_local"].dt.strftime(
        "%H:%M"
    )

    observed_flow_input = observed_flow_input[
        ["date", "time", "discharge_cfs_for_model_comparison", "flow_status"]
    ]

    observed_flow_input.to_csv(OBSERVED_FLOW_INPUT_FILE, index=False)
    completed_flow.to_csv(FLOW_QC_FILE, index=False)

    print(f"Saved HEC-HMS precipitation input: {PRECIPITATION_INPUT_FILE}")
    print(f"Saved HEC-HMS observed-flow input: {OBSERVED_FLOW_INPUT_FILE}")
    print(f"Saved flow-status QC file: {FLOW_QC_FILE}")


def create_summary(
    rainfall: pd.DataFrame,
    completed_flow: pd.DataFrame,
    gaps: pd.DataFrame,
    validation: pd.DataFrame,
    official_daily: pd.DataFrame,
) -> None:
    """Write the HEC-HMS input-preparation quality summary."""
    baseline = completed_flow[
        (completed_flow["time_local"] >= BASELINE_START)
        & (completed_flow["time_local"] < BASELINE_END)
    ]

    baseline_median_cfs = float(
        baseline["discharge_cfs_for_model_comparison"].median()
    )

    rainfall_total_inches = float(rainfall["precipitation_inches"].sum())
    interpolated_hours = int(
        (completed_flow["flow_status"] == "interpolated_isolated_one_hour_gap").sum()
    )

    rainfall_with_flow_status = rainfall.merge(
        completed_flow[["time_local", "flow_status"]],
        on="time_local",
        how="left",
    )

    rainfall_during_interpolated_hours = float(
        rainfall_with_flow_status.loc[
            rainfall_with_flow_status["flow_status"]
            == "interpolated_isolated_one_hour_gap",
            "precipitation_inches",
        ].sum()
    )

    hourly_total_depth_inches = depth_from_hourly_mean_flow(
        completed_flow["discharge_cfs_for_model_comparison"]
    )
    daily_total_depth_inches = depth_from_daily_mean_flow(
        official_daily["daily_mean_discharge_cfs"]
    )

    hourly_above_baseline_depth_inches = depth_from_hourly_mean_flow(
        np.maximum(
            completed_flow["discharge_cfs_for_model_comparison"] - baseline_median_cfs,
            0.0,
        )
    )

    daily_above_baseline_depth_inches = depth_from_daily_mean_flow(
        np.maximum(
            official_daily["daily_mean_discharge_cfs"] - baseline_median_cfs,
            0.0,
        )
    )

    maximum_abs_daily_difference = float(
        validation["absolute_difference_cfs"].max()
    )
    mean_abs_daily_difference = float(
        validation["absolute_difference_cfs"].mean()
    )

    maximum_difference_row = validation.loc[
        validation["absolute_difference_cfs"].idxmax()
    ]

    gap_lines = []
    for _, row in gaps.iterrows():
        gap_lines.append(
            f"- {row['gap_start']}: one interpolated hourly mean discharge value"
        )

    gaps_text = "\n".join(gap_lines) if gap_lines else "No hourly gaps required filling."

    summary = f"""Nahunta HEC-HMS Input Preparation Summary
========================================

Final model site:
{SITE_NAME}

USGS station:
{SITE_NUMBER}

Model input configuration
-------------------------
Unit system: U.S. Customary
Computation interval recommended for baseline model: 1 hour
Drainage area: {DRAINAGE_AREA_SQMI:.2f} square miles
Precipitation method: Specified Hyetograph
Rainfall time series units: incremental inches per hour
Observed-flow time series units: cubic feet per second

Simulation period
-----------------
{WINDOW_START} through {WINDOW_END}, exclusive
Number of hourly precipitation values: {len(rainfall)}
Number of hourly observed/completed flow values: {len(completed_flow)}

Rainfall forcing
----------------
Complete basin-average MRMS precipitation depth: {rainfall_total_inches:.2f} inches
Missing rainfall values: {int(rainfall["precipitation_inches"].isna().sum())}

Observed-flow completion
------------------------
Observed hourly flow values retained: {len(completed_flow) - interpolated_hours}
Interpolated isolated one-hour flow values: {interpolated_hours}
Rainfall occurring during interpolated-flow hours: {rainfall_during_interpolated_hours:.2f} inches
Maximum allowed consecutive interpolated gap: {MAX_INTERPOLATABLE_GAP_HOURS} hour

Interpolated hourly periods
---------------------------
{gaps_text}

Official peak benchmark
-----------------------
Instantaneous USGS Florence peak retained separately from hourly calibration series:
{OFFICIAL_INSTANTANEOUS_PEAK_CFS:,.0f} cfs at {OFFICIAL_INSTANTANEOUS_PEAK_TIME}

Daily-flow validation of completed hourly series
------------------------------------------------
Official USGS daily records used for validation: {len(official_daily)}
Maximum absolute difference between completed-hourly daily mean
and official USGS daily mean: {maximum_abs_daily_difference:,.2f} cfs
Date of maximum difference: {maximum_difference_row["date"]:%Y-%m-%d}
Mean absolute daily difference: {mean_abs_daily_difference:,.2f} cfs

Water-balance consistency check
-------------------------------
Completed-hourly total flow depth: {hourly_total_depth_inches:.2f} inches
Official-daily total flow depth: {daily_total_depth_inches:.2f} inches
Completed-hourly above-baseline flow depth: {hourly_above_baseline_depth_inches:.2f} inches
Official-daily above-baseline flow depth: {daily_above_baseline_depth_inches:.2f} inches
Completed-hourly above-baseline depth / rainfall depth:
{hourly_above_baseline_depth_inches / rainfall_total_inches:.3f}
Official-daily above-baseline depth / rainfall depth:
{daily_above_baseline_depth_inches / rainfall_total_inches:.3f}

Modeling decision
-----------------
The Nahunta Swamp rainfall and discharge datasets are suitable for developing
the baseline HEC-HMS event-reconstruction model. Hourly discharge gaps are
isolated and explicitly flagged after interpolation. The measured
instantaneous peak of 6,060 cfs remains a separate benchmark, while the
completed hourly discharge series will be used for one-hour hydrograph
comparison and initial calibration diagnostics.
"""

    SUMMARY_FILE.write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print(f"Saved HEC-HMS preparation summary: {SUMMARY_FILE}")


def create_figure(
    rainfall: pd.DataFrame,
    completed_flow: pd.DataFrame,
) -> None:
    """Plot prepared precipitation and observed/interpolated discharge inputs."""
    observed = completed_flow[
        completed_flow["flow_status"] == "observed_hourly_mean"
    ]
    interpolated = completed_flow[
        completed_flow["flow_status"] == "interpolated_isolated_one_hour_gap"
    ]

    figure, (rain_axis, flow_axis) = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2]},
    )

    rain_axis.bar(
        rainfall["time_local"],
        rainfall["precipitation_inches"],
        width=0.035,
    )
    rain_axis.invert_yaxis()
    rain_axis.set_ylabel("Hourly rainfall\n(inches)")
    rain_axis.set_title(
        "Prepared HEC-HMS Inputs for Hurricane Florence Event Reconstruction\n"
        "Nahunta Swamp near Shine, North Carolina"
    )
    rain_axis.grid(True, alpha=0.3)

    flow_axis.plot(
        completed_flow["time_local"],
        completed_flow["discharge_cfs_for_model_comparison"],
        linewidth=1.6,
        label="Hourly flow series for model comparison",
    )

    flow_axis.scatter(
        observed["time_local"],
        observed["hourly_mean_discharge_cfs"],
        s=9,
        label="Observed hourly means",
        zorder=3,
    )

    flow_axis.scatter(
        interpolated["time_local"],
        interpolated["discharge_cfs_for_model_comparison"],
        marker="x",
        s=28,
        label="Interpolated isolated hours",
        zorder=4,
    )

    flow_axis.axhline(
        OFFICIAL_INSTANTANEOUS_PEAK_CFS,
        linestyle="--",
        linewidth=1.2,
        label="Official instantaneous peak benchmark: 6,060 cfs",
    )

    flow_axis.set_xlabel("Date and time, Eastern Time")
    flow_axis.set_ylabel("Discharge\n(cfs)")
    flow_axis.grid(True, alpha=0.3)
    flow_axis.legend()

    figure.tight_layout()
    figure.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved HEC-HMS input QC figure: {FIGURE_FILE}")


def main() -> None:
    """Prepare and validate final HEC-HMS input time series."""
    create_directories()

    rainfall = load_rainfall()
    hourly_flow = load_hourly_flow()
    gaps = identify_missing_runs(hourly_flow)
    completed_flow = complete_hourly_flow(hourly_flow, gaps)
    official_daily = load_daily_flow()

    validation = validate_completed_flow_against_daily(
        completed_flow,
        official_daily,
    )

    save_hec_hms_input_tables(rainfall, completed_flow)
    create_summary(
        rainfall,
        completed_flow,
        gaps,
        validation,
        official_daily,
    )
    create_figure(rainfall, completed_flow)

    print()
    print("Nahunta HEC-HMS input preparation completed successfully.")


if __name__ == "__main__":
    main()