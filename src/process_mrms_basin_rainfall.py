"""Construct a basin-average Hurricane Florence rainfall time series.

Study area:
    Trent River watershed upstream of USGS 02092500, North Carolina

Inputs:
    - 120 hourly MRMS GaugeCorr_QPE_01H compressed GRIB2 rainfall files
    - Trent River watershed GeoJSON boundary
    - Processed USGS observed discharge CSV

Outputs:
    - Hourly basin-average rainfall CSV
    - Rainfall-processing summary text file
    - Combined rainfall and observed streamflow figure

Method:
    Each hourly MRMS precipitation grid is clipped to the watershed polygon.
    Valid non-negative rainfall grid cells inside the watershed are averaged
    to obtain an hourly basin-average precipitation estimate.
"""

from __future__ import annotations

import gzip
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
from rasterio.mask import mask
from shapely.affinity import translate
from shapely.geometry import mapping


# ---------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------

RAINFALL_DIR = Path("data/raw/rainfall/mrms_gaugecorr_qpe_01h")
WATERSHED_FILE = Path("data/raw/watershed/trent_river_usgs_02092500_basin.geojson")
STREAMFLOW_FILE = Path(
    "data/processed/usgs_02092500_discharge_processed_2018-09-10_to_2018-09-22.csv"
)

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

OUTPUT_RAINFALL_CSV = (
    PROCESSED_DIR
    / "mrms_trent_river_basin_average_rainfall_20180913_20180917.csv"
)

OUTPUT_SUMMARY = RESULTS_DIR / "mrms_basin_average_rainfall_summary.txt"
OUTPUT_FIGURE = FIGURES_DIR / "mrms_basin_rainfall_and_observed_streamflow.png"

EXPECTED_NUMBER_OF_FILES = 120


def create_directories() -> None:
    """Create required output directories."""
    for directory in [PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def parse_timestamp_from_filename(file_path: Path) -> datetime:
    """Extract the UTC timestamp embedded in an MRMS filename."""
    match = re.search(r"_(\d{8}-\d{6})\.grib2\.gz$", file_path.name)

    if not match:
        raise ValueError(f"Could not parse timestamp from filename: {file_path.name}")

    timestamp = datetime.strptime(match.group(1), "%Y%m%d-%H%M%S")
    return timestamp.replace(tzinfo=timezone.utc)


def get_rainfall_files() -> list[Path]:
    """Return the local MRMS rainfall files ordered by timestamp."""
    rainfall_files = sorted(
        RAINFALL_DIR.glob("*.grib2.gz"),
        key=parse_timestamp_from_filename,
    )

    if not rainfall_files:
        raise FileNotFoundError(
            f"No MRMS rainfall files were found in {RAINFALL_DIR}."
        )

    if len(rainfall_files) != EXPECTED_NUMBER_OF_FILES:
        print(
            f"Warning: expected {EXPECTED_NUMBER_OF_FILES} rainfall files, "
            f"but found {len(rainfall_files)}."
        )

    return rainfall_files


def read_watershed() -> gpd.GeoDataFrame:
    """Read the Trent River watershed polygon."""
    watershed = gpd.read_file(WATERSHED_FILE)

    if watershed.empty:
        raise ValueError("Watershed boundary file is empty.")

    if watershed.crs is None:
        watershed = watershed.set_crs("EPSG:4326")

    return watershed


def decompress_to_temporary_grib(compressed_file: Path) -> Path:
    """Temporarily decompress a single MRMS GRIB2 file."""
    temporary_file = tempfile.NamedTemporaryFile(
        suffix=".grib2",
        delete=False,
    )
    temporary_path = Path(temporary_file.name)
    temporary_file.close()

    with gzip.open(compressed_file, "rb") as source:
        with temporary_path.open("wb") as destination:
            shutil.copyfileobj(source, destination)

    return temporary_path


def align_watershed_to_raster(
    watershed: gpd.GeoDataFrame,
    raster_crs,
    raster_bounds,
) -> gpd.GeoDataFrame:
    """Project watershed geometry to the raster CRS and adjust longitude if needed."""
    aligned_watershed = watershed.copy()

    if raster_crs is not None:
        aligned_watershed = aligned_watershed.to_crs(raster_crs)

    watershed_bounds = aligned_watershed.total_bounds

    raster_uses_positive_longitudes = raster_bounds.left > 0
    watershed_uses_negative_longitudes = watershed_bounds[0] < 0

    if raster_uses_positive_longitudes and watershed_uses_negative_longitudes:
        shifted_watershed = aligned_watershed.copy()
        shifted_watershed.geometry = shifted_watershed.geometry.apply(
            lambda geometry: translate(geometry, xoff=360)
        )

        shifted_bounds = shifted_watershed.total_bounds

        if (
            shifted_bounds[0] <= raster_bounds.right
            and shifted_bounds[2] >= raster_bounds.left
        ):
            aligned_watershed = shifted_watershed

    return aligned_watershed


def process_one_grid(
    compressed_file: Path,
    watershed: gpd.GeoDataFrame,
) -> tuple[dict, dict]:
    """Clip one rainfall grid and calculate watershed rainfall statistics."""
    timestamp_utc = parse_timestamp_from_filename(compressed_file)
    temporary_grib = decompress_to_temporary_grib(compressed_file)

    try:
        with rasterio.open(temporary_grib) as source:
            aligned_watershed = align_watershed_to_raster(
                watershed,
                source.crs,
                source.bounds,
            )

            geometries = [
                mapping(geometry) for geometry in aligned_watershed.geometry
            ]

            clipped_array, _ = mask(
                source,
                geometries,
                crop=True,
                filled=False,
                indexes=1,
            )

            rainfall = np.ma.array(clipped_array, copy=False)
            rainfall = np.ma.masked_invalid(rainfall)
            rainfall = np.ma.masked_where(rainfall < 0, rainfall)

            valid_values = rainfall.compressed()

            if valid_values.size == 0:
                raise ValueError(
                    f"No valid rainfall cells remained after clipping {compressed_file.name}."
                )

            band_tags = source.tags(1)

            record = {
                "timestamp_utc": timestamp_utc,
                "timestamp_local": timestamp_utc.astimezone(),
                "source_filename": compressed_file.name,
                "valid_watershed_cells": int(valid_values.size),
                "basin_mean_precip_mm": float(valid_values.mean()),
                "basin_median_precip_mm": float(np.median(valid_values)),
                "basin_max_precip_mm": float(valid_values.max()),
                "basin_min_precip_mm": float(valid_values.min()),
            }

            metadata = {
                "driver": source.driver,
                "crs": str(source.crs),
                "width": source.width,
                "height": source.height,
                "nodata": source.nodata,
                "band_tags": band_tags,
            }

            return record, metadata

    finally:
        temporary_grib.unlink(missing_ok=True)


def process_all_rainfall_grids(
    rainfall_files: list[Path],
    watershed: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Generate hourly basin-average rainfall statistics for all files."""
    records: list[dict] = []
    first_metadata: dict = {}

    print(f"Processing {len(rainfall_files)} MRMS rainfall grids...")

    for index, rainfall_file in enumerate(rainfall_files, start=1):
        record, metadata = process_one_grid(rainfall_file, watershed)
        records.append(record)

        if index == 1:
            first_metadata = metadata

        print(
            f"[{index:03d}/{len(rainfall_files):03d}] "
            f"{record['timestamp_utc']:%Y-%m-%d %H:%M UTC} — "
            f"mean rainfall = {record['basin_mean_precip_mm']:.3f} mm"
        )

    rainfall = pd.DataFrame(records)
    rainfall["timestamp_utc"] = pd.to_datetime(rainfall["timestamp_utc"], utc=True)
    rainfall["timestamp_local"] = (
        rainfall["timestamp_utc"].dt.tz_convert("America/New_York")
    )

    rainfall = rainfall.sort_values("timestamp_utc").reset_index(drop=True)

    rainfall["cumulative_basin_mean_precip_mm"] = (
        rainfall["basin_mean_precip_mm"].cumsum()
    )
    rainfall["basin_mean_precip_inches"] = (
        rainfall["basin_mean_precip_mm"] / 25.4
    )
    rainfall["cumulative_basin_mean_precip_inches"] = (
        rainfall["cumulative_basin_mean_precip_mm"] / 25.4
    )

    return rainfall, first_metadata


def save_rainfall_csv(rainfall: pd.DataFrame) -> None:
    """Save the processed basin-average rainfall time series."""
    rainfall.to_csv(OUTPUT_RAINFALL_CSV, index=False)
    print(f"\nSaved basin-average rainfall CSV: {OUTPUT_RAINFALL_CSV}")


def load_streamflow_hourly() -> pd.DataFrame:
    """Load USGS streamflow and aggregate observations to hourly mean discharge."""
    if not STREAMFLOW_FILE.exists():
        raise FileNotFoundError(
            f"Observed streamflow file not found: {STREAMFLOW_FILE}"
        )

    streamflow = pd.read_csv(STREAMFLOW_FILE)

    streamflow["time_local"] = pd.to_datetime(
        streamflow["time_local"],
        utc=True,
    ).dt.tz_convert("America/New_York")

    streamflow["discharge_cfs"] = pd.to_numeric(
        streamflow["discharge_cfs"],
        errors="coerce",
    )

    hourly_streamflow = (
        streamflow.set_index("time_local")["discharge_cfs"]
        .resample("1h")
        .mean()
        .reset_index()
    )

    return hourly_streamflow


def create_summary(
    rainfall: pd.DataFrame,
    streamflow_hourly: pd.DataFrame,
    metadata: dict,
) -> None:
    """Write rainfall statistics and storm-response summary."""
    rainfall_peak = rainfall.loc[rainfall["basin_mean_precip_mm"].idxmax()]
    streamflow_peak = streamflow_hourly.loc[
        streamflow_hourly["discharge_cfs"].idxmax()
    ]

    total_precip_mm = rainfall["basin_mean_precip_mm"].sum()
    total_precip_inches = total_precip_mm / 25.4

    min_valid_cells = int(rainfall["valid_watershed_cells"].min())
    max_valid_cells = int(rainfall["valid_watershed_cells"].max())

    lag_hours = (
        streamflow_peak["time_local"] - rainfall_peak["timestamp_local"]
    ).total_seconds() / 3600

    band_tags_text = "\n".join(
        f"{key}: {value}" for key, value in metadata.get("band_tags", {}).items()
    )
    if not band_tags_text:
        band_tags_text = "No GRIB band tags returned by Rasterio."

    summary = f"""MRMS Basin-Average Rainfall Processing Summary
==============================================

Project:
Rainfall-to-Streamflow Modeling of Hurricane Florence Flooding
in the Trent River Watershed, North Carolina

Rainfall product:
MRMS GaugeCorr_QPE_01H

Processing method:
Hourly MRMS rainfall grids were clipped to the USGS-derived Trent River
watershed polygon. Non-negative valid grid-cell values within the polygon
were averaged for each hourly timestep.

Processed rainfall record
-------------------------
Number of hourly grids processed: {len(rainfall)}
First rainfall timestamp: {rainfall["timestamp_local"].min()}
Last rainfall timestamp: {rainfall["timestamp_local"].max()}
Valid watershed cells per hour: {min_valid_cells} to {max_valid_cells}

Storm rainfall results
----------------------
Total basin-average precipitation: {total_precip_mm:,.2f} mm
Total basin-average precipitation: {total_precip_inches:,.2f} inches
Maximum hourly basin-average precipitation: {rainfall_peak["basin_mean_precip_mm"]:,.3f} mm
Maximum hourly rainfall timestamp: {rainfall_peak["timestamp_local"]}
Maximum grid-cell rainfall observed in any hour: {rainfall["basin_max_precip_mm"].max():,.3f} mm

Observed streamflow comparison
------------------------------
Observed peak discharge: {streamflow_peak["discharge_cfs"]:,.0f} cubic feet per second
Observed peak discharge timestamp: {streamflow_peak["time_local"]}
Descriptive lag from maximum hourly basin rainfall to peak discharge: {lag_hours:,.1f} hours

Important interpretation note
-----------------------------
The lag calculation compares the single wettest basin-average rainfall hour
with the observed discharge peak. It is a descriptive screening metric only;
formal hydrologic interpretation should consider cumulative rainfall,
antecedent wetness, routing, and model calibration.

Raster metadata from first processed MRMS file
----------------------------------------------
Driver: {metadata.get("driver")}
Coordinate reference system: {metadata.get("crs")}
Raster dimensions: {metadata.get("width")} columns x {metadata.get("height")} rows
NoData value: {metadata.get("nodata")}

GRIB band tags
--------------
{band_tags_text}

Quality note
------------
USACE HEC-HMS guidance recommends evaluating gridded precipitation quality
against other rainfall observations before formal hydrologic calibration.
A rain-gauge comparison will be considered as a later quality-control step.
"""

    OUTPUT_SUMMARY.write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print(f"Saved rainfall-processing summary: {OUTPUT_SUMMARY}")


def create_combined_figure(
    rainfall: pd.DataFrame,
    streamflow_hourly: pd.DataFrame,
) -> None:
    """Plot basin-average rainfall with observed river discharge."""
    figure, (rain_axis, flow_axis) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2]},
    )

    rain_axis.bar(
        rainfall["timestamp_local"],
        rainfall["basin_mean_precip_mm"],
        width=0.035,
    )
    rain_axis.invert_yaxis()
    rain_axis.set_ylabel("Hourly rainfall\n(mm)")
    rain_axis.set_title(
        "Hurricane Florence Rainfall and Observed Streamflow Response\n"
        "Trent River Watershed near Trenton, North Carolina"
    )
    rain_axis.grid(True, alpha=0.3)

    flow_axis.plot(
        streamflow_hourly["time_local"],
        streamflow_hourly["discharge_cfs"],
        linewidth=2,
        label="Observed USGS discharge",
    )

    streamflow_peak = streamflow_hourly.loc[
        streamflow_hourly["discharge_cfs"].idxmax()
    ]

    flow_axis.scatter(
        streamflow_peak["time_local"],
        streamflow_peak["discharge_cfs"],
        zorder=3,
    )
    flow_axis.annotate(
        f'Peak = {streamflow_peak["discharge_cfs"]:,.0f} cfs\n'
        f'{streamflow_peak["time_local"]:%Y-%m-%d %H:%M %Z}',
        xy=(streamflow_peak["time_local"], streamflow_peak["discharge_cfs"]),
        xytext=(25, -55),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )

    flow_axis.set_xlabel("Date and time, Eastern Time")
    flow_axis.set_ylabel("Observed discharge\n(cubic feet per second)")
    flow_axis.grid(True, alpha=0.3)
    flow_axis.legend()

    figure.tight_layout()
    figure.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved combined rainfall-streamflow figure: {OUTPUT_FIGURE}")


def main() -> None:
    """Run full basin-average rainfall processing workflow."""
    create_directories()

    rainfall_files = get_rainfall_files()
    watershed = read_watershed()

    rainfall, metadata = process_all_rainfall_grids(rainfall_files, watershed)
    save_rainfall_csv(rainfall)

    streamflow_hourly = load_streamflow_hourly()
    create_summary(rainfall, streamflow_hourly, metadata)
    create_combined_figure(rainfall, streamflow_hourly)

    print()
    print("MRMS basin-average rainfall processing completed successfully.")


if __name__ == "__main__":
    main()