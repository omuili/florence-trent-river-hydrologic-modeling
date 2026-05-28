"""Download and summarize the contributing watershed boundary for USGS 02092500.

Study site:
    USGS 02092500 - Trent River near Trenton, North Carolina

Outputs:
    - Watershed boundary GeoJSON
    - Watershed boundary map
    - Basin-area summary text file

Data source:
    U.S. Geological Survey Network Linked Data Index (NLDI) Basin Endpoint

Notes:
    The script first requests a point-specific basin using splitCatchment=true.
    If the USGS service cannot complete that geoprocessing operation, the script
    automatically falls back to the standard whole-catchment upstream basin.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import requests
from shapely.geometry import Point


# ---------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------

SITE_ID = "USGS-02092500"
SITE_NUMBER = "02092500"
SITE_NAME = "Trent River near Trenton, NC"

# Published station coordinates and drainage area from USGS metadata.
GAUGE_LONGITUDE = -77.46138889
GAUGE_LATITUDE = 35.06416667
PUBLISHED_DRAINAGE_AREA_SQMI = 168.0

NLDI_BASIN_URL = (
    "https://api.water.usgs.gov/nldi/linked-data/"
    f"nwissite/{SITE_ID}/basin"
)

RAW_WATERSHED_DIR = Path("data/raw/watershed")
FIGURES_DIR = Path("figures")
RESULTS_DIR = Path("results")


def create_directories() -> None:
    """Create required output directories."""
    for directory in [RAW_WATERSHED_DIR, FIGURES_DIR, RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def request_basin(params: dict[str, str], method_label: str) -> dict[str, Any] | None:
    """Attempt a single watershed-boundary request.

    Returns a GeoJSON payload if successful; otherwise returns None and
    prints enough diagnostic information for reproducibility.
    """
    print(f"\nTrying delineation method: {method_label}")

    try:
        response = requests.get(NLDI_BASIN_URL, params=params, timeout=180)
        print(f"Request URL: {response.url}")
        print(f"HTTP status: {response.status_code}")

        if response.status_code != 200:
            print(
                f"USGS NLDI could not complete this method: "
                f"{response.reason}. Trying fallback method..."
            )
            return None

        payload = response.json()

        if not payload.get("features"):
            print("Response contained no watershed polygon. Trying fallback method...")
            return None

        return payload

    except requests.RequestException as error:
        print(f"Request failed: {error}. Trying fallback method...")
        return None


def download_basin_geojson() -> tuple[Path, str]:
    """Download a watershed boundary with robust fallback options."""
    delineation_attempts = [
        (
            {
                "f": "json",
                "simplified": "false",
                "splitCatchment": "true",
            },
            "Precise point-specific basin: full resolution with splitCatchment=true",
        ),
        (
            {
                "f": "json",
                "simplified": "false",
                "splitCatchment": "false",
            },
            "Standard upstream basin: full resolution with splitCatchment=false",
        ),
        (
            {
                "f": "json",
                "simplified": "true",
                "splitCatchment": "false",
            },
            "Standard upstream basin: simplified geometry with splitCatchment=false",
        ),
    ]

    for params, method_label in delineation_attempts:
        payload = request_basin(params, method_label)

        if payload is not None:
            output_file = (
                RAW_WATERSHED_DIR / "trent_river_usgs_02092500_basin.geojson"
            )

            with output_file.open("w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2)

            print(f"\nSuccessful delineation method: {method_label}")
            print(f"Saved watershed boundary: {output_file}")

            return output_file, method_label

    raise RuntimeError(
        "All USGS NLDI basin requests failed. The service may be experiencing "
        "a temporary problem for this station. If this persists, we will use "
        "an alternate watershed-boundary source."
    )


def read_and_measure_basin(geojson_file: Path) -> tuple[gpd.GeoDataFrame, float]:
    """Read the watershed polygon and calculate area in square miles."""
    basin = gpd.read_file(geojson_file)

    if basin.empty:
        raise ValueError("The downloaded watershed boundary file is empty.")

    if basin.crs is None:
        basin = basin.set_crs("EPSG:4326")

    # NAD83 / Conus Albers is appropriate for equal-area calculation in CONUS.
    basin_equal_area = basin.to_crs("EPSG:5070")

    square_meters_per_square_mile = 2_589_988.110336
    basin_area_sqmi = (
        basin_equal_area.geometry.area.sum() / square_meters_per_square_mile
    )

    return basin, float(basin_area_sqmi)


def create_summary(basin_area_sqmi: float, method_label: str) -> Path:
    """Create a text summary comparing mapped and published drainage areas."""
    difference_sqmi = basin_area_sqmi - PUBLISHED_DRAINAGE_AREA_SQMI
    percent_difference = (
        difference_sqmi / PUBLISHED_DRAINAGE_AREA_SQMI
    ) * 100

    method_note = (
        "This method delineates the watershed specifically to the gauge location."
        if "splitCatchment=true" in method_label
        else (
            "The precise split-catchment request failed at the USGS service, so this "
            "boundary represents the standard upstream NHDPlus basin assembled from "
            "whole catchments. It remains appropriate for initial basin-average "
            "rainfall processing, subject to the area comparison below."
        )
    )

    summary = f"""Watershed Boundary Acquisition Summary
======================================

Project:
Rainfall-to-Streamflow Modeling of Hurricane Florence Flooding
in the Trent River Watershed, North Carolina

Study site:
{SITE_NAME}

USGS monitoring location:
{SITE_ID}

Boundary source:
U.S. Geological Survey Network Linked Data Index (NLDI) basin endpoint

Successful delineation method:
{method_label}

Method note:
{method_note}

Drainage-area comparison
------------------------
Calculated polygon area: {basin_area_sqmi:,.2f} square miles
Published USGS drainage area: {PUBLISHED_DRAINAGE_AREA_SQMI:,.2f} square miles
Difference: {difference_sqmi:,.2f} square miles
Percent difference: {percent_difference:,.2f} percent

Interpretation note
-------------------
The downloaded polygon represents the contributing upstream watershed used
for initial spatial averaging of Hurricane Florence precipitation. If the
calculated area differs materially from the published 168-square-mile USGS
drainage area, an alternative delineation method will be evaluated before
formal model calibration.
"""

    output_file = RESULTS_DIR / "watershed_boundary_summary.txt"
    output_file.write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print(f"Saved watershed summary: {output_file}")

    return output_file


def plot_watershed_boundary(basin: gpd.GeoDataFrame) -> Path:
    """Plot the watershed boundary and selected USGS gauge location."""
    gauge = gpd.GeoDataFrame(
        {"site_name": [SITE_NAME]},
        geometry=[Point(GAUGE_LONGITUDE, GAUGE_LATITUDE)],
        crs="EPSG:4326",
    )

    basin_projected = basin.to_crs("EPSG:3857")
    gauge_projected = gauge.to_crs("EPSG:3857")

    figure, axis = plt.subplots(figsize=(8, 8))

    basin_projected.boundary.plot(
        ax=axis,
        linewidth=1.5,
        label="Watershed boundary",
    )
    gauge_projected.plot(
        ax=axis,
        marker="*",
        markersize=140,
        label="USGS stream gauge",
    )

    axis.set_title(
        "Trent River Watershed Upstream of Trenton, North Carolina\n"
        "USGS 02092500"
    )
    axis.set_xlabel("Projected east-west coordinate")
    axis.set_ylabel("Projected north-south coordinate")
    axis.legend()
    axis.grid(True, alpha=0.3)

    figure.tight_layout()

    output_file = FIGURES_DIR / "trent_river_watershed_boundary.png"
    figure.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved watershed boundary figure: {output_file}")
    return output_file


def main() -> None:
    """Run watershed boundary acquisition, validation, and mapping."""
    create_directories()
    geojson_file, method_label = download_basin_geojson()
    basin, basin_area_sqmi = read_and_measure_basin(geojson_file)
    create_summary(basin_area_sqmi, method_label)
    plot_watershed_boundary(basin)

    print()
    print("Watershed boundary workflow completed successfully.")


if __name__ == "__main__":
    main()