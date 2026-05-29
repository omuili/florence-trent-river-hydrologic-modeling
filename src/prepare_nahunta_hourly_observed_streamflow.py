"""Prepare hourly observed streamflow for final Nahunta Swamp modeling.

Final modeling candidate:
    USGS 02091000 - Nahunta Swamp near Shine, North Carolina

Purpose:
    Download continuous USGS discharge observations for the pre-Michael
    Hurricane Florence response period, aggregate them to hourly values,
    inspect data coverage and missing periods, and compare the observed
    streamflow response against processed MRMS hourly rainfall.

Outputs:
    - Raw USGS continuous-response JSON
    - Parsed continuous observations CSV
    - Hourly observed discharge CSV
    - Hourly rainfall-streamflow comparison figure
    - Data-quality summary
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SITE_ID = "USGS-02091000"
SITE_NUMBER = "02091000"
SITE_NAME = "Nahunta Swamp near Shine, NC"

PARAMETER_CODE = "00060"  # Discharge, cubic feet per second

# September 10 through October 10, 2018, Eastern Time.
START_LOCAL = pd.Timestamp("2018-09-10 00:00:00", tz="America/New_York")
END_LOCAL = pd.Timestamp("2018-10-11 00:00:00", tz="America/New_York")

START_UTC = "2018-09-10T04:00:00Z"
END_UTC = "2018-10-11T04:00:00Z"

BASELINE_START = pd.Timestamp("2018-09-10 00:00:00", tz="America/New_York")
BASELINE_END = pd.Timestamp("2018-09-12 12:00:00", tz="America/New_York")

OFFICIAL_FLORENCE_PEAK_CFS = 6060.0
OFFICIAL_FLORENCE_PEAK_DATE = "2018-09-15"

API_URL = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/continuous/items"

RAINFALL_FILE = Path(
    "data/processed/final_model/nahunta_swamp/"
    "nahunta_mrms_hourly_basin_average_rainfall_20180910_20181010.csv"
)

RAW_DIR = Path("data/raw/streamflow/nahunta_final")
PROCESSED_DIR = Path("data/processed/final_model/nahunta_swamp")
RESULTS_DIR = Path("results/final_model/nahunta_swamp")
FIGURES_DIR = Path("figures/final_model/nahunta_swamp")

RAW_JSON_FILE = RAW_DIR / "nahunta_usgs_continuous_discharge_20180910_20181010.json"
CONTINUOUS_CSV_FILE = PROCESSED_DIR / "nahunta_usgs_continuous_discharge_20180910_20181010.csv"
HOURLY_CSV_FILE = PROCESSED_DIR / "nahunta_usgs_hourly_observed_discharge_20180910_20181010.csv"
SUMMARY_FILE = RESULTS_DIR / "nahunta_hourly_streamflow_quality_summary.txt"
FIGURE_FILE = FIGURES_DIR / "nahunta_hourly_rainfall_streamflow_quality_control.png"


def create_directories() -> None:
    """Create required output directories."""
    for directory in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def download_continuous_discharge() -> dict:
    """Download continuous USGS discharge observations."""
    params = {
        "f": "json",
        "monitoring_location_id": SITE_ID,
        "parameter_code": PARAMETER_CODE,
        "time": f"{START_UTC}/{END_UTC}",
        "limit": 10000,
    }

    print("Requesting continuous USGS discharge for Nahunta Swamp...")
    response = requests.get(API_URL, params=params, timeout=120)
    print(f"Request URL: {response.url}")
    response.raise_for_status()

    payload = response.json()
    features = payload.get("features", [])

    if not features:
        raise ValueError("USGS returned no continuous discharge observations.")

    with RAW_JSON_FILE.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print(f"Downloaded {len(features):,} continuous observations.")
    print(f"Saved raw response: {RAW_JSON_FILE}")

    return payload


def parse_continuous_observations(payload: dict) -> pd.DataFrame:
    """Parse continuous observations from the USGS response."""
    records: list[dict] = []

    for feature in payload["features"]:
        properties = feature.get("properties", {})

        records.append(
            {
                "time_utc": properties.get("time"),
                "discharge_cfs": properties.get("value"),
                "unit": properties.get("unit_of_measure"),
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
        .reset_index(drop=True)
    )

    observations.to_csv(CONTINUOUS_CSV_FILE, index=False)
    print(f"Saved parsed continuous observations: {CONTINUOUS_CSV_FILE}")

    return observations


def create_hourly_series(observations: pd.DataFrame) -> pd.DataFrame:
    """Aggregate continuous observations to an hourly observed-flow series."""
    hourly_values = (
        observations.set_index("time_local")["discharge_cfs"]
        .resample("1h")
        .agg(["mean", "max", "count"])
        .rename(
            columns={
                "mean": "hourly_mean_discharge_cfs",
                "max": "hourly_max_discharge_cfs",
                "count": "observation_count_in_hour",
            }
        )
    )

    expected_hours = pd.date_range(
        start=START_LOCAL,
        end=END_LOCAL,
        freq="1h",
        inclusive="left",
    )

    hourly = pd.DataFrame(index=expected_hours)
    hourly.index.name = "time_local"
    hourly = hourly.join(hourly_values).reset_index()

    hourly["hour_has_observed_discharge"] = hourly[
        "hourly_mean_discharge_cfs"
    ].notna()

    hourly.to_csv(HOURLY_CSV_FILE, index=False)
    print(f"Saved hourly observed discharge series: {HOURLY_CSV_FILE}")

    return hourly


def identify_missing_hourly_periods(hourly: pd.DataFrame) -> pd.DataFrame:
    """Identify contiguous periods with no hourly observed discharge."""
    missing = ~hourly["hour_has_observed_discharge"]

    if not missing.any():
        return pd.DataFrame(columns=["gap_start", "gap_end", "missing_hours"])

    groups = missing.ne(missing.shift()).cumsum()
    records: list[dict] = []

    for _, group in hourly[missing].groupby(groups[missing]):
        records.append(
            {
                "gap_start": group["time_local"].min(),
                "gap_end": group["time_local"].max(),
                "missing_hours": len(group),
            }
        )

    return pd.DataFrame(records)


def load_rainfall() -> pd.DataFrame:
    """Load final-model Nahunta MRMS rainfall time series."""
    if not RAINFALL_FILE.exists():
        raise FileNotFoundError(f"Rainfall input not found: {RAINFALL_FILE}")

    rainfall = pd.read_csv(RAINFALL_FILE)

    rainfall["time_local"] = pd.to_datetime(
        rainfall["timestamp_local"],
        utc=True,
    ).dt.tz_convert("America/New_York")

    rainfall["basin_mean_precip_inches"] = pd.to_numeric(
        rainfall["basin_mean_precip_inches"],
        errors="coerce",
    )

    return rainfall[["time_local", "basin_mean_precip_inches"]]


def create_summary(
    observations: pd.DataFrame,
    hourly: pd.DataFrame,
    gaps: pd.DataFrame,
    rainfall: pd.DataFrame,
) -> None:
    """Write the final hourly streamflow quality-control summary."""
    valid_hourly = hourly.dropna(subset=["hourly_mean_discharge_cfs"])

    observed_peak = observations.loc[observations["discharge_cfs"].idxmax()]

    baseline = valid_hourly[
        (valid_hourly["time_local"] >= BASELINE_START)
        & (valid_hourly["time_local"] < BASELINE_END)
    ]
    baseline_median_cfs = float(baseline["hourly_mean_discharge_cfs"].median())

    expected_hours = len(hourly)
    observed_hours = int(hourly["hour_has_observed_discharge"].sum())
    missing_hours = int((~hourly["hour_has_observed_discharge"]).sum())

    rainfall_total_inches = float(rainfall["basin_mean_precip_inches"].sum())

    merged = rainfall.merge(
        hourly[["time_local", "hour_has_observed_discharge"]],
        on="time_local",
        how="left",
    )

    rainfall_during_missing_flow_inches = float(
        merged.loc[
            merged["hour_has_observed_discharge"] == False,
            "basin_mean_precip_inches",
        ].sum()
    )

    if gaps.empty:
        gaps_text = "No missing hourly discharge periods were found."
        maximum_gap_hours = 0
    else:
        maximum_gap_hours = int(gaps["missing_hours"].max())
        gap_lines = []
        for _, row in gaps.iterrows():
            gap_lines.append(
                f"- {row['gap_start']} through {row['gap_end']}: "
                f"{int(row['missing_hours'])} hours"
            )
        gaps_text = "\n".join(gap_lines)

    peak_difference_cfs = observed_peak["discharge_cfs"] - OFFICIAL_FLORENCE_PEAK_CFS
    peak_percent_difference = (
        peak_difference_cfs / OFFICIAL_FLORENCE_PEAK_CFS
    ) * 100

    summary = f"""Nahunta Hourly Observed Streamflow Quality Summary
================================================

Final modeling candidate:
{SITE_NAME}

USGS station:
{SITE_NUMBER}

Analysis window:
{START_LOCAL} through {END_LOCAL}, exclusive

MRMS rainfall forcing
---------------------
Complete basin-average rainfall depth: {rainfall_total_inches:.2f} inches

Continuous discharge retrieval
------------------------------
Number of continuous USGS observations downloaded: {len(observations):,}
First continuous observation: {observations["time_local"].min()}
Last continuous observation: {observations["time_local"].max()}

Hourly observed discharge coverage
----------------------------------
Expected hourly timesteps: {expected_hours}
Hours with at least one observed discharge value: {observed_hours}
Hours with no observed discharge value: {missing_hours}
Longest missing hourly period: {maximum_gap_hours} hours
Rainfall occurring during missing-discharge hours: {rainfall_during_missing_flow_inches:.2f} inches

Observed peak comparison
------------------------
Maximum discharge in retrieved continuous record: {observed_peak["discharge_cfs"]:,.0f} cfs
Timestamp of retrieved maximum: {observed_peak["time_local"]}
USGS Florence-report peak reference: {OFFICIAL_FLORENCE_PEAK_CFS:,.0f} cfs on {OFFICIAL_FLORENCE_PEAK_DATE}
Difference from report peak: {peak_difference_cfs:,.0f} cfs
Percent difference from report peak: {peak_percent_difference:.2f} percent

Pre-event discharge screening
-----------------------------
Median hourly discharge before substantial Florence rainfall:
{baseline_median_cfs:,.0f} cfs

Missing hourly discharge periods
--------------------------------
{gaps_text}

Decision rule
-------------
If hourly coverage is complete or missing periods are short and do not
remove a substantial portion of storm rainfall or the observed peak, this
series may be used for HEC-HMS hydrograph calibration and performance
assessment. If major gaps overlap storm rainfall or the peak response,
calibration metrics must be restricted or the site reconsidered.
"""

    SUMMARY_FILE.write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print(f"Saved hourly streamflow summary: {SUMMARY_FILE}")


def create_figure(
    hourly: pd.DataFrame,
    rainfall: pd.DataFrame,
) -> None:
    """Plot hourly rainfall and observed discharge for quality control."""
    figure, (rain_axis, flow_axis) = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2]},
    )

    rain_axis.bar(
        rainfall["time_local"],
        rainfall["basin_mean_precip_inches"],
        width=0.035,
    )
    rain_axis.invert_yaxis()
    rain_axis.set_ylabel("Hourly rainfall\n(inches)")
    rain_axis.set_title(
        "Hurricane Florence Rainfall and Observed Streamflow Quality Control\n"
        "USGS 02091000 — Nahunta Swamp near Shine, North Carolina"
    )
    rain_axis.grid(True, alpha=0.3)

    flow_axis.plot(
        hourly["time_local"],
        hourly["hourly_mean_discharge_cfs"],
        linewidth=2,
        label="Hourly observed discharge",
    )
    flow_axis.set_xlabel("Date and time, Eastern Time")
    flow_axis.set_ylabel("Discharge\n(cfs)")
    flow_axis.grid(True, alpha=0.3)
    flow_axis.legend()

    figure.tight_layout()
    figure.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved hourly rainfall-streamflow figure: {FIGURE_FILE}")


def main() -> None:
    """Run the Nahunta hourly discharge quality-control workflow."""
    create_directories()

    payload = download_continuous_discharge()
    observations = parse_continuous_observations(payload)
    hourly = create_hourly_series(observations)
    gaps = identify_missing_hourly_periods(hourly)
    rainfall = load_rainfall()

    create_summary(observations, hourly, gaps, rainfall)
    create_figure(hourly, rainfall)

    print()
    print("Nahunta hourly streamflow quality-control workflow completed successfully.")


if __name__ == "__main__":
    main()