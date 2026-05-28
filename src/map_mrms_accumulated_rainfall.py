"""Map accumulated MRMS Hurricane Florence rainfall over the Trent River watershed.

Inputs:
    - Hourly MRMS GaugeCorr_QPE_01H GRIB2 files
    - Trent River watershed GeoJSON boundary

Outputs:
    - Accumulated storm rainfall map in inches
    - Spatial rainfall summary text file
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
import rasterio
from rasterio.mask import mask
from rasterio.transform import array_bounds
from shapely.geometry import Point, mapping


RAINFALL_DIR = Path("data/raw/rainfall/mrms_gaugecorr_qpe_01h")
WATERSHED_FILE = Path("data/raw/watershed/trent_river_usgs_02092500_basin.geojson")

FIGURES_DIR = Path("figures")
RESULTS_DIR = Path("results")

OUTPUT_FIGURE = FIGURES_DIR / "mrms_accumulated_storm_rainfall_trent_river.png"
OUTPUT_SUMMARY = RESULTS_DIR / "mrms_accumulated_rainfall_spatial_summary.txt"

GAUGE_LONGITUDE = -77.46138889
GAUGE_LATITUDE = 35.06416667


def parse_timestamp(file_path: Path) -> datetime:
    """Parse the timestamp from an MRMS GRIB2 filename."""
    match = re.search(r"_(\d{8}-\d{6})\.grib2\.gz$", file_path.name)
    if not match:
        raise ValueError(f"Unable to parse timestamp from {file_path.name}")

    timestamp = datetime.strptime(match.group(1), "%Y%m%d-%H%M%S")
    return timestamp.replace(tzinfo=timezone.utc)


def decompress_to_temp(compressed_file: Path) -> Path:
    """Decompress one MRMS file to a temporary GRIB2 path."""
    temporary_file = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False)
    temporary_path = Path(temporary_file.name)
    temporary_file.close()

    with gzip.open(compressed_file, "rb") as source:
        with temporary_path.open("wb") as destination:
            shutil.copyfileobj(source, destination)

    return temporary_path


def main() -> None:
    """Accumulate hourly rainfall grids and generate a watershed map."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rainfall_files = sorted(RAINFALL_DIR.glob("*.grib2.gz"), key=parse_timestamp)
    if not rainfall_files:
        raise FileNotFoundError(f"No rainfall files found in {RAINFALL_DIR}")

    watershed = gpd.read_file(WATERSHED_FILE)
    if watershed.crs is None:
        watershed = watershed.set_crs("EPSG:4326")

    gauge = gpd.GeoDataFrame(
        {"site": ["USGS 02092500"]},
        geometry=[Point(GAUGE_LONGITUDE, GAUGE_LATITUDE)],
        crs="EPSG:4326",
    )

    accumulated_mm = None
    output_transform = None
    output_crs = None

    for index, rainfall_file in enumerate(rainfall_files, start=1):
        temporary_grib = decompress_to_temp(rainfall_file)

        try:
            with rasterio.open(temporary_grib) as source:
                aligned_watershed = watershed.to_crs(source.crs)
                geometries = [mapping(geometry) for geometry in aligned_watershed.geometry]

                clipped, transform = mask(
                    source,
                    geometries,
                    crop=True,
                    filled=False,
                    indexes=1,
                )

                hourly_mm = np.ma.array(clipped, copy=False)
                hourly_mm = np.ma.masked_invalid(hourly_mm)
                hourly_mm = np.ma.masked_where(hourly_mm < 0, hourly_mm)

                if accumulated_mm is None:
                    accumulated_mm = hourly_mm.filled(0).astype(float)
                    output_transform = transform
                    output_crs = source.crs
                else:
                    accumulated_mm += hourly_mm.filled(0)

            print(f"[{index:03d}/{len(rainfall_files):03d}] Added {rainfall_file.name}")

        finally:
            temporary_grib.unlink(missing_ok=True)

    if accumulated_mm is None or output_transform is None:
        raise RuntimeError("No rainfall grids were accumulated.")

    basin_mask = accumulated_mm <= 0
    accumulated_inches = np.ma.masked_where(basin_mask, accumulated_mm / 25.4)

    valid_values = accumulated_inches.compressed()
    if valid_values.size == 0:
        raise ValueError("No valid accumulated rainfall cells were produced.")

    watershed_projected = watershed.to_crs(output_crs)
    gauge_projected = gauge.to_crs(output_crs)

    left, bottom, right, top = array_bounds(
        accumulated_inches.shape[0],
        accumulated_inches.shape[1],
        output_transform,
    )

    figure, axis = plt.subplots(figsize=(9, 7))

    image = axis.imshow(
        accumulated_inches,
        extent=(left, right, bottom, top),
        origin="upper",
    )

    watershed_projected.boundary.plot(
        ax=axis,
        linewidth=1.0,
        label="Watershed boundary",
    )

    gauge_projected.plot(
        ax=axis,
        marker="*",
        markersize=120,
        label="USGS 02092500",
    )

    axis.set_title(
        "MRMS-Estimated Hurricane Florence Accumulated Rainfall\n"
        "Trent River Watershed near Trenton, North Carolina"
    )
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.legend()

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Accumulated precipitation, inches")

    figure.tight_layout()
    figure.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(figure)

    summary = f"""MRMS Accumulated Rainfall Spatial Summary
=========================================

Project:
Rainfall-to-Streamflow Modeling of Hurricane Florence Flooding
in the Trent River Watershed, North Carolina

Number of hourly rainfall grids accumulated: {len(rainfall_files)}

Spatial accumulated-rainfall statistics
---------------------------------------
Mean watershed-cell accumulated rainfall: {valid_values.mean():,.2f} inches
Median watershed-cell accumulated rainfall: {np.median(valid_values):,.2f} inches
Minimum watershed-cell accumulated rainfall: {valid_values.min():,.2f} inches
Maximum watershed-cell accumulated rainfall: {valid_values.max():,.2f} inches

Interpretation note
-------------------
This map provides an initial spatial assessment of MRMS-estimated storm
rainfall within the Trent River watershed. It will be used to evaluate
whether a basin-average hyetograph is an appropriate first modeling
representation and to identify any notable within-basin rainfall gradients.
"""

    OUTPUT_SUMMARY.write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print(f"Saved accumulated rainfall map: {OUTPUT_FIGURE}")
    print(f"Saved accumulated rainfall summary: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()