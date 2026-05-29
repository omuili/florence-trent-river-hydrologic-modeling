"""Attempt precise official watershed delineation for Nahunta Swamp near Shine, NC.

Candidate final modeling site:
    USGS 02091000 - Nahunta Swamp near Shine, North Carolina

Acceptance rule:
    The boundary is accepted only if:
    1. The official USGS NLDI point-specific split-catchment request succeeds; and
    2. Its calculated polygon area is within 2 percent of the published
       USGS drainage area of 80.4 square miles.

If this precision request fails, the script stops without accepting a
fallback boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import requests


SITE_ID = "USGS-02091000"
SITE_NUMBER = "02091000"
SITE_NAME = "Nahunta Swamp near Shine, NC"

PUBLISHED_DRAINAGE_AREA_SQMI = 80.4
ACCEPTABLE_PERCENT_DIFFERENCE = 2.0
SQUARE_METERS_PER_SQUARE_MILE = 2_589_988.110336

NLDI_BASIN_URL = (
    "https://api.water.usgs.gov/nldi/linked-data/"
    f"nwissite/{SITE_ID}/basin"
)

OUTPUT_DIR = Path("data/raw/watershed/nahunta_screening")
RESULTS_DIR = Path("results/site_screening/nahunta_swamp")
OUTPUT_BOUNDARY = OUTPUT_DIR / "nahunta_swamp_usgs_02091000_nldi_splitcatchment_basin.geojson"
OUTPUT_SUMMARY = RESULTS_DIR / "nahunta_precise_boundary_screening_summary.txt"


def main() -> None:
    """Request, validate, and conditionally accept a precise watershed boundary."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    params = {
        "f": "json",
        "simplified": "false",
        "splitCatchment": "true",
    }

    print("Requesting point-specific basin boundary from USGS NLDI...")
    response = requests.get(NLDI_BASIN_URL, params=params, timeout=180)

    print(f"Request URL: {response.url}")
    print(f"HTTP status: {response.status_code}")

    if response.status_code != 200:
        summary = f"""Nahunta Precise Boundary Screening Summary
==========================================

Candidate site:
{SITE_NAME}

USGS station:
{SITE_NUMBER}

Published drainage area:
{PUBLISHED_DRAINAGE_AREA_SQMI:.2f} square miles

Boundary method attempted:
USGS NLDI basin endpoint with splitCatchment=true

Result:
Request failed with HTTP status {response.status_code}: {response.reason}

Decision:
No boundary accepted. Do not process rainfall or discharge for Nahunta
until an outlet-specific verified watershed boundary is obtained.
"""
        OUTPUT_SUMMARY.write_text(summary, encoding="utf-8")
        print()
        print(summary)
        print(f"Saved screening summary: {OUTPUT_SUMMARY}")
        return

    payload = response.json()

    if not payload.get("features"):
        raise ValueError("USGS NLDI returned no basin geometry.")

    with OUTPUT_BOUNDARY.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    basin = gpd.read_file(OUTPUT_BOUNDARY)

    if basin.crs is None:
        basin = basin.set_crs("EPSG:4326")

    area_sqmi = (
        basin.to_crs("EPSG:5070").geometry.area.sum()
        / SQUARE_METERS_PER_SQUARE_MILE
    )

    difference_sqmi = area_sqmi - PUBLISHED_DRAINAGE_AREA_SQMI
    percent_difference = (
        difference_sqmi / PUBLISHED_DRAINAGE_AREA_SQMI
    ) * 100

    accepted = abs(percent_difference) <= ACCEPTABLE_PERCENT_DIFFERENCE

    decision = (
        "Accept this point-specific boundary for initial Nahunta screening."
        if accepted
        else (
            "Do not accept this boundary. Its calculated area is outside the "
            "2 percent validation tolerance."
        )
    )

    summary = f"""Nahunta Precise Boundary Screening Summary
==========================================

Candidate site:
{SITE_NAME}

USGS station:
{SITE_NUMBER}

Boundary method:
USGS NLDI basin endpoint with splitCatchment=true

Drainage-area validation
------------------------
Published USGS drainage area: {PUBLISHED_DRAINAGE_AREA_SQMI:.2f} square miles
Calculated polygon area: {area_sqmi:.2f} square miles
Difference: {difference_sqmi:.2f} square miles
Percent difference: {percent_difference:.2f} percent
Acceptance tolerance: plus or minus {ACCEPTABLE_PERCENT_DIFFERENCE:.2f} percent

Decision
--------
{decision}

Output boundary file:
{OUTPUT_BOUNDARY}
"""

    OUTPUT_SUMMARY.write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print(f"Saved screening summary: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()