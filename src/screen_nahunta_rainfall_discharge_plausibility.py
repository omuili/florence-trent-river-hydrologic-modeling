"""Screen Nahunta Swamp near Shine for hydrologic modeling suitability.

Candidate site:
    USGS 02091000 - Nahunta Swamp near Shine, North Carolina

Purpose:
    Determine whether Hurricane Florence rainfall and official USGS daily
    discharge yield a physically plausible event-response screening result
    before investing time in HEC-HMS model construction.

Inputs:
    - Accepted point-specific NLDI split-catchment basin boundary
    - Existing local MRMS GaugeCorr_QPE_01H rainfall grids

Downloads:
    - Official USGS daily mean discharge for the screening period

Outputs:
    - Hourly basin-average MRMS rainfall CSV
    - Daily rainfall/discharge diagnostic CSV
    - Raw and processed USGS daily discharge files
    - Screening summary
    - Rainfall/discharge diagnostic figure
"""

from __future__ import annotations

import gzip
import json
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.mask import mask
from shapely.geometry import mapping


# ---------------------------------------------------------------------
# Candidate-site configuration
# ---------------------------------------------------------------------

SITE_ID = "USGS-02091000"
SITE_NUMBER = "02091000"
SITE_NAME = "Nahunta Swamp near Shine, NC"

PUBLISHED_DRAINAGE_AREA_SQMI = 80.40
SCREENING_BASIN_AREA_SQMI = 80.40

PARAMETER_CODE = "00060"   # Discharge
STATISTIC_ID = "00003"      # Daily mean

WINDOW_START = pd.Timestamp("2018-09-10 00:00:00", tz="America/New_York")
WINDOW_END = pd.Timestamp("2018-10-11 00:00:00", tz="America/New_York")

BASELINE_START = pd.Timestamp("2018-09-10")
BASELINE_END = pd.Timestamp("2018-09-13")

EXPECTED_RAINFALL_FILES = 744

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

BASIN_FILE = Path(
    "data/raw/watershed/nahunta_screening/"
    "nahunta_swamp_usgs_02091000_nldi_splitcatchment_basin.geojson"
)

RAINFALL_DIR = Path("data/raw/rainfall/mrms_gaugecorr_qpe_01h")

OUTPUT_DIR = Path("data/processed/site_screening/nahunta_swamp")
RAW_STREAMFLOW_DIR = Path("data/raw/streamflow/nahunta_screening")
RESULTS_DIR = Path("results/site_screening/nahunta_swamp")
FIGURES_DIR = Path("figures/site_screening/nahunta_swamp")

RAINFALL_CSV = OUTPUT_DIR / "nahunta_mrms_hourly_basin_average_rainfall_20180910_20181010.csv"
DAILY_DISCHARGE_CSV = OUTPUT_DIR / "nahunta_usgs_daily_mean_discharge_20180910_20181010.csv"
DAILY_DIAGNOSTIC_CSV = OUTPUT_DIR / "nahunta_daily_rainfall_discharge_screening.csv"

RAW_DAILY_JSON = RAW_STREAMFLOW_DIR / "nahunta_usgs_daily_mean_discharge_20180910_20181010.json"

SUMMARY_FILE = RESULTS_DIR / "nahunta_site_plausibility_screening_summary.txt"
FIGURE_FILE = FIGURES_DIR / "nahunta_daily_rainfall_discharge_screening.png"

USGS_DAILY_API = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items"

SQUARE_METERS_PER_SQUARE_MILE = 2_589_988.110336
SQUARE_FEET_PER_SQUARE_MILE = 5280.0 ** 2
SECONDS_PER_DAY = 86400.0


def create_directories() -> None:
    """Create output directories."""
    for directory in [
        OUTPUT_DIR,
        RAW_STREAMFLOW_DIR,
        RESULTS_DIR,
        FIGURES_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def parse_rainfall_timestamp(file_path: Path) -> datetime:
    """Parse UTC timestamp from an MRMS filename."""
    match = re.search(r"_(\d{8}-\d{6})\.grib2\.gz$", file_path.name)

    if not match:
        raise ValueError(f"Unable to parse timestamp from: {file_path.name}")

    timestamp = datetime.strptime(match.group(1), "%Y%m%d-%H%M%S")
    return timestamp.replace(tzinfo=timezone.utc)


def get_rainfall_files() -> list[Path]:
    """Return the 744 local MRMS files for the screening period."""
    files = sorted(
        RAINFALL_DIR.glob("*.grib2.gz"),
        key=parse_rainfall_timestamp,
    )

    selected: list[Path] = []

    for file_path in files:
        timestamp_local = pd.Timestamp(
            parse_rainfall_timestamp(file_path)
        ).tz_convert("America/New_York")

        if WINDOW_START <= timestamp_local < WINDOW_END:
            selected.append(file_path)

    if len(selected) != EXPECTED_RAINFALL_FILES:
        raise ValueError(
            f"Expected {EXPECTED_RAINFALL_FILES} rainfall files for the "
            f"screening period, but found {len(selected)}."
        )

    return selected


def read_and_validate_basin() -> gpd.GeoDataFrame:
    """Read accepted Nahunta boundary and independently verify polygon area."""
    if not BASIN_FILE.exists():
        raise FileNotFoundError(f"Boundary file not found: {BASIN_FILE}")

    basin = gpd.read_file(BASIN_FILE)

    if basin.empty:
        raise ValueError("Nahunta basin boundary file is empty.")

    if basin.crs is None:
        basin = basin.set_crs("EPSG:4326")

    calculated_area_sqmi = (
        basin.to_crs("EPSG:5070").geometry.area.sum()
        / SQUARE_METERS_PER_SQUARE_MILE
    )

    percent_difference = (
        (calculated_area_sqmi - PUBLISHED_DRAINAGE_AREA_SQMI)
        / PUBLISHED_DRAINAGE_AREA_SQMI
        * 100
    )

    print("Nahunta boundary validation")
    print("---------------------------")
    print(f"Published drainage area: {PUBLISHED_DRAINAGE_AREA_SQMI:.2f} mi²")
    print(f"Calculated polygon area: {calculated_area_sqmi:.2f} mi²")
    print(f"Percent difference: {percent_difference:.2f}%")
    print()

    if abs(percent_difference) > 2.0:
        raise ValueError(
            "Nahunta boundary area exceeds the 2% screening tolerance. "
            "Do not proceed with rainfall processing."
        )

    return basin


def decompress_to_temporary_grib(compressed_file: Path) -> Path:
    """Decompress one MRMS .gz file to a temporary GRIB2 file."""
    temporary_file = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False)
    temporary_path = Path(temporary_file.name)
    temporary_file.close()

    with gzip.open(compressed_file, "rb") as source:
        with temporary_path.open("wb") as destination:
            shutil.copyfileobj(source, destination)

    return temporary_path


def process_rainfall(
    rainfall_files: list[Path],
    basin: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Construct hourly basin-average rainfall series for Nahunta."""
    records: list[dict] = []

    print(f"Processing {len(rainfall_files)} MRMS grids for Nahunta Swamp...")

    for index, compressed_file in enumerate(rainfall_files, start=1):
        timestamp_utc = parse_rainfall_timestamp(compressed_file)
        temporary_grib = decompress_to_temporary_grib(compressed_file)

        try:
            with rasterio.open(temporary_grib) as source:
                basin_aligned = basin.to_crs(source.crs)
                geometries = [mapping(geometry) for geometry in basin_aligned.geometry]

                clipped, _ = mask(
                    source,
                    geometries,
                    crop=True,
                    filled=False,
                    indexes=1,
                )

                rainfall = np.ma.array(clipped, copy=False)
                rainfall = np.ma.masked_invalid(rainfall)
                rainfall = np.ma.masked_where(rainfall < 0, rainfall)

                valid_values = rainfall.compressed()

                if valid_values.size == 0:
                    raise ValueError(
                        f"No valid clipped rainfall cells for {compressed_file.name}."
                    )

                records.append(
                    {
                        "timestamp_utc": timestamp_utc,
                        "valid_watershed_cells": int(valid_values.size),
                        "basin_mean_precip_mm": float(valid_values.mean()),
                        "basin_max_precip_mm": float(valid_values.max()),
                    }
                )

        finally:
            temporary_grib.unlink(missing_ok=True)

        if index == 1 or index % 50 == 0 or index == len(rainfall_files):
            print(f"[{index:03d}/{len(rainfall_files):03d}] processed")

    hourly = pd.DataFrame(records)
    hourly["timestamp_utc"] = pd.to_datetime(hourly["timestamp_utc"], utc=True)
    hourly["timestamp_local"] = (
        hourly["timestamp_utc"].dt.tz_convert("America/New_York")
    )
    hourly["basin_mean_precip_inches"] = hourly["basin_mean_precip_mm"] / 25.4
    hourly["cumulative_precip_inches"] = (
        hourly["basin_mean_precip_inches"].cumsum()
    )

    hourly.to_csv(RAINFALL_CSV, index=False)
    print(f"Saved Nahunta rainfall CSV: {RAINFALL_CSV}")

    return hourly


def download_daily_discharge() -> pd.DataFrame:
    """Download and parse official USGS daily mean discharge for Nahunta."""
    params = {
        "f": "json",
        "monitoring_location_id": SITE_ID,
        "parameter_code": PARAMETER_CODE,
        "statistic_id": STATISTIC_ID,
        "time": "2018-09-10/2018-10-10",
        "limit": 1000,
    }

    print()
    print("Requesting official USGS daily mean discharge for Nahunta...")
    response = requests.get(USGS_DAILY_API, params=params, timeout=120)
    print(f"Request URL: {response.url}")
    response.raise_for_status()

    payload = response.json()
    features = payload.get("features", [])

    if not features:
        raise ValueError("USGS returned no daily discharge values for Nahunta.")

    with RAW_DAILY_JSON.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    records: list[dict] = []

    for feature in features:
        properties = feature.get("properties", {})

        records.append(
            {
                "date": properties.get("time"),
                "daily_mean_discharge_cfs": properties.get("value"),
                "unit": properties.get("unit_of_measure"),
                "approval_status": properties.get("approval_status"),
                "qualifier": properties.get("qualifier"),
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
        .drop_duplicates("date")
        .dropna(subset=["daily_mean_discharge_cfs"])
        .reset_index(drop=True)
    )

    if len(daily) != 31:
        print(f"Warning: expected 31 daily discharge values, found {len(daily)}.")

    daily.to_csv(DAILY_DISCHARGE_CSV, index=False)
    print(f"Saved Nahunta daily discharge CSV: {DAILY_DISCHARGE_CSV}")

    return daily


def build_daily_diagnostic(
    hourly_rainfall: pd.DataFrame,
    daily_discharge: pd.DataFrame,
) -> pd.DataFrame:
    """Create daily rainfall and discharge screening table."""
    daily_rainfall = (
        hourly_rainfall.set_index("timestamp_local")["basin_mean_precip_inches"]
        .resample("1D")
        .sum()
        .reset_index()
    )
    daily_rainfall["date"] = daily_rainfall["timestamp_local"].dt.tz_localize(None)
    daily_rainfall = daily_rainfall.drop(columns="timestamp_local")

    daily = daily_rainfall.merge(daily_discharge, on="date", how="left")
    daily.to_csv(DAILY_DIAGNOSTIC_CSV, index=False)

    print(f"Saved Nahunta daily diagnostic CSV: {DAILY_DIAGNOSTIC_CSV}")
    return daily


def calculate_screening_statistics(daily: pd.DataFrame) -> dict:
    """Calculate water-balance plausibility screening statistics."""
    total_rainfall_in = float(daily["basin_mean_precip_inches"].sum())

    baseline = daily[
        (daily["date"] >= BASELINE_START)
        & (daily["date"] < BASELINE_END)
    ]

    baseline_mean_cfs = float(baseline["daily_mean_discharge_cfs"].mean())

    maximum_daily_row = daily.loc[daily["daily_mean_discharge_cfs"].idxmax()]

    basin_area_sqft = SCREENING_BASIN_AREA_SQMI * SQUARE_FEET_PER_SQUARE_MILE

    total_discharge_volume_cuft = float(
        (daily["daily_mean_discharge_cfs"] * SECONDS_PER_DAY).sum()
    )

    above_baseline_cfs = np.maximum(
        daily["daily_mean_discharge_cfs"] - baseline_mean_cfs,
        0.0,
    )
    above_baseline_volume_cuft = float(
        (above_baseline_cfs * SECONDS_PER_DAY).sum()
    )

    total_discharge_depth_in = (
        total_discharge_volume_cuft / basin_area_sqft
    ) * 12.0

    above_baseline_discharge_depth_in = (
        above_baseline_volume_cuft / basin_area_sqft
    ) * 12.0

    runoff_ratio = above_baseline_discharge_depth_in / total_rainfall_in

    later_rainfall_in = float(
        daily.loc[
            daily["date"] >= pd.Timestamp("2018-09-23"),
            "basin_mean_precip_inches",
        ].sum()
    )

    return {
        "total_rainfall_in": total_rainfall_in,
        "later_rainfall_in": later_rainfall_in,
        "baseline_mean_cfs": baseline_mean_cfs,
        "maximum_daily_mean_discharge_cfs": float(
            maximum_daily_row["daily_mean_discharge_cfs"]
        ),
        "maximum_daily_mean_discharge_date": maximum_daily_row["date"],
        "total_discharge_depth_in": total_discharge_depth_in,
        "above_baseline_discharge_depth_in": above_baseline_discharge_depth_in,
        "runoff_ratio": runoff_ratio,
    }


def create_summary(
    hourly_rainfall: pd.DataFrame,
    daily: pd.DataFrame,
    statistics: dict,
) -> None:
    """Write site-screening decision summary."""
    approval_statuses = ", ".join(
        sorted(daily["approval_status"].dropna().astype(str).unique())
    )

    cells_min = int(hourly_rainfall["valid_watershed_cells"].min())
    cells_max = int(hourly_rainfall["valid_watershed_cells"].max())

    ratio = statistics["runoff_ratio"]

    if ratio <= 1.0:
        decision = """PASS INITIAL WATER-BALANCE PLAUSIBILITY SCREEN.

The official daily mean above-baseline discharge depth does not exceed
the complete MRMS basin-average rainfall depth. Nahunta Swamp is a
defensible candidate for the HEC-HMS event-reconstruction workflow,
subject to final inspection of the rainfall/discharge figure and later
documentation of modeling uncertainties."""
    else:
        decision = """FAIL INITIAL WATER-BALANCE PLAUSIBILITY SCREEN.

The official daily mean above-baseline discharge depth exceeds the complete
MRMS basin-average rainfall depth. Do not proceed to HEC-HMS modeling for
this candidate without investigating precipitation and discharge uncertainty."""
    
    summary = f"""Nahunta Swamp Site Plausibility Screening Summary
================================================

Candidate modeling site:
{SITE_NAME}

USGS station:
{SITE_NUMBER}

Boundary verification
---------------------
Boundary source: USGS NLDI point-specific basin, splitCatchment=true
Published drainage area: {PUBLISHED_DRAINAGE_AREA_SQMI:.2f} square miles
Calculated boundary area from prior validation: 79.80 square miles
Boundary difference from published area: -0.75 percent

Screening period
----------------
September 10, 2018 through October 10, 2018
Hourly rainfall grids processed: {len(hourly_rainfall)}
Valid watershed rainfall cells per hour: {cells_min} to {cells_max}
Daily discharge records analyzed: {len(daily)}
USGS daily discharge approval statuses returned: {approval_statuses}

Rainfall forcing
----------------
Complete MRMS basin-average rainfall depth: {statistics["total_rainfall_in"]:.2f} inches
Rainfall from September 23 through October 10: {statistics["later_rainfall_in"]:.2f} inches

Official daily discharge response
---------------------------------
Pre-event daily mean baseline discharge: {statistics["baseline_mean_cfs"]:,.0f} cfs
Maximum daily mean discharge: {statistics["maximum_daily_mean_discharge_cfs"]:,.0f} cfs
Date of maximum daily mean discharge: {statistics["maximum_daily_mean_discharge_date"]:%Y-%m-%d}
Total discharge depth over screening interval: {statistics["total_discharge_depth_in"]:.2f} inches
Above-baseline discharge depth over screening interval: {statistics["above_baseline_discharge_depth_in"]:.2f} inches

Water-balance plausibility metric
---------------------------------
Above-baseline discharge depth / rainfall depth: {statistics["runoff_ratio"]:.3f}

Screening decision
------------------
{decision}

Interpretation note
-------------------
This is an initial site-selection screen, not a complete calibrated model
assessment. Because the pre-Michael window may truncate part of the recession
limb, a physically plausible ratio below one supports proceeding, but does
not by itself validate a hydrologic model.
"""

    SUMMARY_FILE.write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print(f"Saved Nahunta screening summary: {SUMMARY_FILE}")


def create_figure(daily: pd.DataFrame, statistics: dict) -> None:
    """Create rainfall and daily-discharge screening figure."""
    figure, (rain_axis, flow_axis) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2]},
    )

    rain_axis.bar(
        daily["date"],
        daily["basin_mean_precip_inches"],
        width=0.8,
    )
    rain_axis.invert_yaxis()
    rain_axis.set_ylabel("Daily rainfall\n(inches)")
    rain_axis.set_title(
        "Hurricane Florence Site Screening: Rainfall and Daily Mean Discharge\n"
        "USGS 02091000 — Nahunta Swamp near Shine, North Carolina"
    )
    rain_axis.grid(True, alpha=0.3)

    flow_axis.plot(
        daily["date"],
        daily["daily_mean_discharge_cfs"],
        linewidth=2,
        marker="o",
        label="USGS daily mean discharge",
    )
    flow_axis.set_xlabel("Date")
    flow_axis.set_ylabel("Daily mean discharge\n(cfs)")
    flow_axis.grid(True, alpha=0.3)
    flow_axis.legend()

    annotation = (
        f"MRMS rainfall = {statistics['total_rainfall_in']:.2f} in\n"
        f"Above-baseline flow depth = {statistics['above_baseline_discharge_depth_in']:.2f} in\n"
        f"Screening ratio = {statistics['runoff_ratio']:.3f}"
    )

    flow_axis.text(
        0.98,
        0.95,
        annotation,
        transform=flow_axis.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    figure.tight_layout()
    figure.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved Nahunta screening figure: {FIGURE_FILE}")


def main() -> None:
    """Run complete Nahunta site-plausibility screening."""
    create_directories()

    basin = read_and_validate_basin()
    rainfall_files = get_rainfall_files()
    hourly_rainfall = process_rainfall(rainfall_files, basin)

    daily_discharge = download_daily_discharge()
    daily_diagnostic = build_daily_diagnostic(hourly_rainfall, daily_discharge)
    statistics = calculate_screening_statistics(daily_diagnostic)

    create_summary(hourly_rainfall, daily_diagnostic, statistics)
    create_figure(daily_diagnostic, statistics)

    print()
    print("Nahunta site plausibility screening completed successfully.")


if __name__ == "__main__":
    main()