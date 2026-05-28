"""Inspect and clip one MRMS rainfall grid to the Trent River watershed.

Purpose:
    Confirm that locally downloaded MRMS GRIB2 rainfall data can be read
    correctly and spatially intersect the watershed boundary before
    processing the full Hurricane Florence rainfall time series.

Inputs:
    - One compressed MRMS GaugeCorr_QPE_01H GRIB2 file
    - Trent River watershed GeoJSON boundary

Outputs:
    - Console inspection report
    - One clipped-rainfall preview figure
    - One text summary file
"""

from __future__ import annotations

import gzip
import shutil
import tempfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.affinity import translate
from shapely.geometry import mapping


# ---------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------

RAINFALL_DIR = Path("data/raw/rainfall/mrms_gaugecorr_qpe_01h")
WATERSHED_FILE = Path("data/raw/watershed/trent_river_usgs_02092500_basin.geojson")

# Choose an hour during the main Florence rainfall period.
PREFERRED_TEST_PATTERN = "*20180914-120000.grib2.gz"

FIGURES_DIR = Path("figures")
RESULTS_DIR = Path("results")


def select_test_file() -> Path:
    """Select one MRMS file for inspection."""
    preferred_files = sorted(RAINFALL_DIR.glob(PREFERRED_TEST_PATTERN))

    if preferred_files:
        return preferred_files[0]

    fallback_files = sorted(RAINFALL_DIR.glob("*.grib2.gz"))

    if not fallback_files:
        raise FileNotFoundError(
            f"No MRMS .grib2.gz files were found in {RAINFALL_DIR}."
        )

    print(
        "Preferred Florence test hour was not found; "
        f"using the first available file: {fallback_files[0].name}"
    )
    return fallback_files[0]


def decompress_to_temporary_grib(compressed_file: Path) -> Path:
    """Decompress one .grib2.gz file into a temporary GRIB2 file."""
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


def load_watershed_for_raster(
    watershed_file: Path,
    raster_crs,
    raster_bounds,
) -> gpd.GeoDataFrame:
    """Read the watershed boundary and align it with the rainfall raster."""
    watershed = gpd.read_file(watershed_file)

    if watershed.empty:
        raise ValueError("Watershed boundary file is empty.")

    if watershed.crs is None:
        watershed = watershed.set_crs("EPSG:4326")

    if raster_crs is not None:
        watershed = watershed.to_crs(raster_crs)

    watershed_bounds = watershed.total_bounds

    # Some GRIB2 rasters use longitude coordinates from 0 to 360.
    # If the watershed is represented using negative longitude values
    # and does not overlap the raster, try shifting it eastward by 360°.
    raster_uses_positive_longitudes = raster_bounds.left > 0
    watershed_uses_negative_longitudes = watershed_bounds[0] < 0

    if raster_uses_positive_longitudes and watershed_uses_negative_longitudes:
        shifted_geometries = watershed.geometry.apply(
            lambda geometry: translate(geometry, xoff=360)
        )
        shifted_watershed = watershed.copy()
        shifted_watershed.geometry = shifted_geometries

        shifted_bounds = shifted_watershed.total_bounds

        if (
            shifted_bounds[0] <= raster_bounds.right
            and shifted_bounds[2] >= raster_bounds.left
        ):
            print("Shifted watershed longitudes by +360 degrees to match GRIB2 grid.")
            watershed = shifted_watershed

    return watershed


def inspect_and_clip_rainfall(compressed_file: Path) -> None:
    """Inspect one MRMS grid, clip it to the basin, and summarize rainfall."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    temporary_grib = decompress_to_temporary_grib(compressed_file)

    try:
        with rasterio.open(temporary_grib) as source:
            print()
            print("MRMS raster inspection")
            print("----------------------")
            print(f"Source file: {compressed_file.name}")
            print(f"Driver: {source.driver}")
            print(f"Coordinate reference system: {source.crs}")
            print(f"Raster size: {source.width} columns x {source.height} rows")
            print(f"Number of raster bands: {source.count}")
            print(f"Raster bounds: {source.bounds}")
            print(f"NoData value: {source.nodata}")

            watershed = load_watershed_for_raster(
                WATERSHED_FILE,
                source.crs,
                source.bounds,
            )

            geometries = [mapping(geometry) for geometry in watershed.geometry]

            clipped_array, clipped_transform = mask(
                source,
                geometries,
                crop=True,
                filled=False,
                indexes=1,
            )

            # Rainfall values should be non-negative.
            # Negative encoded values, where present, are treated as invalid.
            rainfall = np.ma.masked_where(clipped_array < 0, clipped_array)
            valid_values = rainfall.compressed()

            if valid_values.size == 0:
                raise ValueError(
                    "The clipped rainfall grid contains no valid non-negative cells."
                )

            basin_mean_mm = float(valid_values.mean())
            basin_max_mm = float(valid_values.max())
            basin_total_valid_cells = int(valid_values.size)

            summary = f"""MRMS Single-Grid Inspection Summary
===================================

Project:
Rainfall-to-Streamflow Modeling of Hurricane Florence Flooding
in the Trent River Watershed, North Carolina

Rainfall grid inspected:
{compressed_file.name}

Raster metadata
---------------
Driver: {source.driver}
Coordinate reference system: {source.crs}
Raster width: {source.width}
Raster height: {source.height}
Raster bands: {source.count}
NoData value: {source.nodata}

Watershed-clipped rainfall results
----------------------------------
Valid watershed grid cells: {basin_total_valid_cells:,}
Basin-average hourly rainfall: {basin_mean_mm:,.3f} mm
Maximum watershed grid-cell rainfall: {basin_max_mm:,.3f} mm

Interpretation
--------------
This test confirms that the MRMS GRIB2 rainfall product can be read
within the Python workflow and clipped to the Trent River watershed.
The next step will apply this procedure to all 120 hourly rainfall grids
to construct a basin-average Hurricane Florence precipitation time series.
"""

            summary_file = RESULTS_DIR / "mrms_single_grid_inspection_summary.txt"
            summary_file.write_text(summary, encoding="utf-8")

            print()
            print(summary)
            print(f"Saved inspection summary: {summary_file}")

            figure, axis = plt.subplots(figsize=(8, 6))

            image = axis.imshow(
                rainfall,
                extent=(
                    clipped_transform.c,
                    clipped_transform.c + clipped_transform.a * rainfall.shape[1],
                    clipped_transform.f + clipped_transform.e * rainfall.shape[0],
                    clipped_transform.f,
                ),
            )

            axis.set_title(
                "MRMS Hourly Rainfall Clipped to Trent River Watershed\n"
                f"{compressed_file.name}"
            )
            axis.set_xlabel("Raster x-coordinate")
            axis.set_ylabel("Raster y-coordinate")

            colorbar = figure.colorbar(image, ax=axis)
            colorbar.set_label("Hourly precipitation, millimeters")

            figure.tight_layout()

            figure_file = FIGURES_DIR / "mrms_single_grid_clipped_rainfall_preview.png"
            figure.savefig(figure_file, dpi=300, bbox_inches="tight")
            plt.close(figure)

            print(f"Saved clipped rainfall preview figure: {figure_file}")

    finally:
        temporary_grib.unlink(missing_ok=True)


def main() -> None:
    """Run the one-grid MRMS inspection and clipping test."""
    if not WATERSHED_FILE.exists():
        raise FileNotFoundError(
            f"Watershed file not found: {WATERSHED_FILE}. "
            "Run the watershed boundary workflow first."
        )

    selected_file = select_test_file()
    print(f"Selected MRMS test file: {selected_file}")

    inspect_and_clip_rainfall(selected_file)

    print()
    print("MRMS single-grid inspection completed successfully.")


if __name__ == "__main__":
    main()