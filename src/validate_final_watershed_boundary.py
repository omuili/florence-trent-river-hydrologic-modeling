"""Validate and select the final watershed boundary for hydrologic modeling.

Purpose:
    Compare the preliminary USGS NLDI fallback basin against the verified
    USGS StreamStats basin delineated at gauge 02092500.

Final modeling decision:
    Use the StreamStats basin because its reported drainage area is
    168 square miles, matching the published USGS station drainage area.

Outputs:
    - Final verified watershed GeoJSON used by rainfall and HEC-HMS workflows
    - Boundary comparison summary
    - Overlay figure comparing NLDI fallback and StreamStats boundaries
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.ops import unary_union


# ---------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------

STREAMSTATS_FILE = Path(
    "data/raw/watershed/streamstats/"
    "trent_river_usgs_02092500_streamstats_basin.geojson"
)

NLDI_PRELIMINARY_FILE = Path(
    "data/raw/watershed/"
    "preliminary_nldi_fallback_trent_river_usgs_02092500_basin.geojson"
)

FINAL_BOUNDARY_FILE = Path(
    "data/processed/verified_streamstats_trent_river_basin.geojson"
)

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

SUMMARY_FILE = RESULTS_DIR / "watershed_boundary_validation_summary.txt"
FIGURE_FILE = FIGURES_DIR / "watershed_boundary_comparison_nldi_vs_streamstats.png"

SITE_NAME = "Trent River near Trenton, NC"
SITE_NUMBER = "02092500"

# Official USGS gauge metadata/reference value.
PUBLISHED_DRAINAGE_AREA_SQMI = 168.0

# Published station coordinates used only for map reference.
GAUGE_LONGITUDE = -77.46138889
GAUGE_LATITUDE = 35.06416667

SQUARE_METERS_PER_SQUARE_MILE = 2_589_988.110336


def create_directories() -> None:
    """Create output directories if required."""
    FINAL_BOUNDARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def read_single_geometry(file_path: Path, source_name: str) -> gpd.GeoDataFrame:
    """Read one boundary file and consolidate all features into one geometry."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required boundary file was not found:\n{file_path}\n"
            "Check the filename in the project folder before rerunning."
        )

    boundary = gpd.read_file(file_path)

    if boundary.empty:
        raise ValueError(f"Boundary file contains no features: {file_path}")

    if boundary.crs is None:
        boundary = boundary.set_crs("EPSG:4326")

    boundary = boundary[boundary.geometry.notna()].copy()
    boundary["geometry"] = boundary.geometry.make_valid()

    merged_geometry = unary_union(boundary.geometry.tolist())

    return gpd.GeoDataFrame(
        {"boundary_source": [source_name]},
        geometry=[merged_geometry],
        crs=boundary.crs,
    )


def area_square_miles(boundary: gpd.GeoDataFrame) -> float:
    """Calculate boundary area in square miles using an equal-area projection."""
    equal_area = boundary.to_crs("EPSG:5070")
    area_sqmi = equal_area.geometry.area.iloc[0] / SQUARE_METERS_PER_SQUARE_MILE
    return float(area_sqmi)


def compare_boundaries(
    streamstats: gpd.GeoDataFrame,
    nldi: gpd.GeoDataFrame,
) -> dict[str, float]:
    """Calculate area and overlap statistics for two watershed boundaries."""
    streamstats_equal_area = streamstats.to_crs("EPSG:5070")
    nldi_equal_area = nldi.to_crs("EPSG:5070")

    streamstats_geometry = streamstats_equal_area.geometry.iloc[0]
    nldi_geometry = nldi_equal_area.geometry.iloc[0]

    streamstats_area = (
        streamstats_geometry.area / SQUARE_METERS_PER_SQUARE_MILE
    )
    nldi_area = nldi_geometry.area / SQUARE_METERS_PER_SQUARE_MILE

    intersection_area = (
        streamstats_geometry.intersection(nldi_geometry).area
        / SQUARE_METERS_PER_SQUARE_MILE
    )
    nldi_only_area = (
        nldi_geometry.difference(streamstats_geometry).area
        / SQUARE_METERS_PER_SQUARE_MILE
    )
    streamstats_only_area = (
        streamstats_geometry.difference(nldi_geometry).area
        / SQUARE_METERS_PER_SQUARE_MILE
    )

    percent_streamstats_overlap = (intersection_area / streamstats_area) * 100

    return {
        "streamstats_area_sqmi": float(streamstats_area),
        "nldi_area_sqmi": float(nldi_area),
        "intersection_area_sqmi": float(intersection_area),
        "nldi_only_area_sqmi": float(nldi_only_area),
        "streamstats_only_area_sqmi": float(streamstats_only_area),
        "percent_streamstats_overlap": float(percent_streamstats_overlap),
    }


def save_final_boundary(streamstats: gpd.GeoDataFrame) -> None:
    """Save StreamStats geometry as the final verified modeling boundary."""
    final_boundary = streamstats.to_crs("EPSG:4326")
    final_boundary["modeling_status"] = "verified_final_boundary"
    final_boundary["verification_basis"] = (
        "USGS StreamStats DRNAREA reported 168 square miles, matching "
        "published USGS drainage area for station 02092500."
    )

    final_boundary.to_file(FINAL_BOUNDARY_FILE, driver="GeoJSON")
    print(f"Saved final verified modeling boundary: {FINAL_BOUNDARY_FILE}")


def create_summary(statistics: dict[str, float]) -> None:
    """Write the boundary-validation summary."""
    streamstats_difference = (
        statistics["streamstats_area_sqmi"] - PUBLISHED_DRAINAGE_AREA_SQMI
    )
    nldi_difference = (
        statistics["nldi_area_sqmi"] - PUBLISHED_DRAINAGE_AREA_SQMI
    )

    summary = f"""Final Watershed Boundary Validation Summary
===========================================

Project:
Rainfall-to-Streamflow Modeling of Hurricane Florence Flooding
in the Trent River Watershed, North Carolina

Study site:
{SITE_NAME}

USGS station:
{SITE_NUMBER}

Boundary-selection decision
---------------------------
Final modeling boundary: USGS StreamStats delineated basin
Reason: The StreamStats report returned DRNAREA = 168 square miles,
which exactly matches the published USGS drainage area for station
02092500. The NLDI basin remains preserved only as a preliminary
quality-control comparison boundary.

Area comparison calculated from exported geometries
---------------------------------------------------
Published USGS drainage area: {PUBLISHED_DRAINAGE_AREA_SQMI:,.2f} square miles
StreamStats exported polygon area: {statistics["streamstats_area_sqmi"]:,.2f} square miles
StreamStats difference from published value: {streamstats_difference:,.2f} square miles
NLDI fallback polygon area: {statistics["nldi_area_sqmi"]:,.2f} square miles
NLDI difference from published value: {nldi_difference:,.2f} square miles

Boundary overlap comparison
---------------------------
Intersection area: {statistics["intersection_area_sqmi"]:,.2f} square miles
Area included only by NLDI fallback: {statistics["nldi_only_area_sqmi"]:,.2f} square miles
Area included only by StreamStats: {statistics["streamstats_only_area_sqmi"]:,.2f} square miles
Percentage of StreamStats basin overlapping NLDI boundary: {statistics["percent_streamstats_overlap"]:,.2f} percent

Modeling implication
--------------------
All final basin-average rainfall calculations and later HEC-HMS model
inputs will use the verified StreamStats basin boundary. Preliminary
rainfall summaries created with the NLDI fallback geometry are retained
for quality-control documentation only and are not treated as final
model inputs.
"""

    SUMMARY_FILE.write_text(summary, encoding="utf-8")
    print()
    print(summary)
    print(f"Saved validation summary: {SUMMARY_FILE}")


def plot_boundary_comparison(
    streamstats: gpd.GeoDataFrame,
    nldi: gpd.GeoDataFrame,
) -> None:
    """Create an overlay plot of the preliminary and verified boundaries."""
    streamstats_plot = streamstats.to_crs("EPSG:4326")
    nldi_plot = nldi.to_crs("EPSG:4326")

    gauge = gpd.GeoDataFrame(
        {"site_name": [SITE_NAME]},
        geometry=[Point(GAUGE_LONGITUDE, GAUGE_LATITUDE)],
        crs="EPSG:4326",
    )

    figure, axis = plt.subplots(figsize=(9, 8))

    nldi_plot.boundary.plot(
        ax=axis,
        linewidth=2,
        linestyle="--",
        label="Preliminary NLDI fallback boundary",
    )

    streamstats_plot.boundary.plot(
        ax=axis,
        linewidth=2,
        label="Verified StreamStats boundary",
    )

    gauge.plot(
        ax=axis,
        marker="*",
        markersize=120,
        label="USGS 02092500",
    )

    axis.set_title(
        "Watershed Boundary Validation Comparison\n"
        "Trent River near Trenton, North Carolina"
    )
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved boundary-comparison figure: {FIGURE_FILE}")


def main() -> None:
    """Run the final boundary validation and selection workflow."""
    create_directories()

    streamstats = read_single_geometry(
        STREAMSTATS_FILE,
        "USGS StreamStats verified basin",
    )
    nldi = read_single_geometry(
        NLDI_PRELIMINARY_FILE,
        "USGS NLDI preliminary fallback basin",
    )

    statistics = compare_boundaries(streamstats, nldi)

    save_final_boundary(streamstats)
    create_summary(statistics)
    plot_boundary_comparison(streamstats, nldi)

    print()
    print("Final watershed boundary validation completed successfully.")


if __name__ == "__main__":
    main()