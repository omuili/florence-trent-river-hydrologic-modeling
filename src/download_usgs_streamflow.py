"""Download and visualize Hurricane Florence streamflow observations.

Study site:
    USGS-02092500 - Trent River near Trenton, North Carolina

Study period:
    September 10 through September 22, 2018, local Eastern time

Output files:
    - Raw USGS API response in JSON format
    - Processed discharge CSV
    - Quality-control summary text file
    - Event hydrograph PNG figure

Data source:
    Modernized USGS Water Data API, Continuous Values endpoint
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests


# ---------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------

SITE_ID = "USGS-02092500"
SITE_NUMBER = "02092500"
SITE_NAME = "Trent River near Trenton, NC"
PARAMETER_CODE = "00060"  # Discharge, cubic feet per second

# September 10, 00:00 EDT through September 23, 00:00 EDT,
# expressed in UTC for the USGS API.
START_UTC = "2018-09-10T04:00:00Z"
END_UTC = "2018-09-23T04:00:00Z"

STUDY_PERIOD_LABEL = "2018-09-10_to_2018-09-22"

API_URL = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/continuous/items"

RAW_DIR = Path("data/raw/streamflow")
PROCESSED_DIR = Path("data/processed")
FIGURES_DIR = Path("figures")
RESULTS_DIR = Path("results")


def create_directories() -> None:
    """Create output directories if they do not already exist."""
    for directory in [RAW_DIR, PROCESSED_DIR, FIGURES_DIR, RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def download_usgs_data() -> dict:
    """Download continuous discharge observations from the modern USGS API."""
    params = {
        "f": "json",
        "monitoring_location_id": SITE_ID,
        "parameter_code": PARAMETER_CODE,
        "time": f"{START_UTC}/{END_UTC}",
        "limit": 10000,
    }

    print("Requesting observed streamflow data from USGS...")
    response = requests.get(API_URL, params=params, timeout=90)
    print(f"Request URL: {response.url}")
    response.raise_for_status()

    payload = response.json()
    features = payload.get("features", [])

    if not features:
        raise ValueError(
            "The USGS API returned no streamflow observations for the requested period."
        )

    raw_file = RAW_DIR / f"usgs_{SITE_NUMBER}_continuous_{STUDY_PERIOD_LABEL}.json"
    with raw_file.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print(f"Downloaded {len(features):,} observations.")
    print(f"Saved raw API response: {raw_file}")

    return payload


def parse_streamflow_data(payload: dict) -> pd.DataFrame:
    """Parse USGS API observations into a clean pandas DataFrame."""
    records: list[dict] = []

    for feature in payload["features"]:
        properties = feature.get("properties", {})

        records.append(
            {
                "site_id": properties.get("monitoring_location_id", SITE_ID),
                "time_utc": properties.get("time"),
                "discharge_cfs": properties.get("value"),
                "unit": properties.get("unit_of_measure", "ft^3/s"),
                "parameter_code": properties.get("parameter_code", PARAMETER_CODE),
                "approval_status": properties.get("approval_status"),
                "qualifier": properties.get("qualifier"),
            }
        )

    dataframe = pd.DataFrame(records)

    dataframe["time_utc"] = pd.to_datetime(dataframe["time_utc"], utc=True)
    dataframe["time_local"] = dataframe["time_utc"].dt.tz_convert("America/New_York")
    dataframe["discharge_cfs"] = pd.to_numeric(
        dataframe["discharge_cfs"], errors="coerce"
    )

    dataframe = (
        dataframe.sort_values("time_local")
        .drop_duplicates(subset=["time_local"])
        .reset_index(drop=True)
    )

    dataframe.insert(1, "site_name", SITE_NAME)

    return dataframe[
        [
            "site_id",
            "site_name",
            "time_utc",
            "time_local",
            "discharge_cfs",
            "unit",
            "parameter_code",
            "approval_status",
            "qualifier",
        ]
    ]


def save_processed_data(dataframe: pd.DataFrame) -> Path:
    """Save clean streamflow observations as a CSV file."""
    output_file = (
        PROCESSED_DIR
        / f"usgs_{SITE_NUMBER}_discharge_processed_{STUDY_PERIOD_LABEL}.csv"
    )

    dataframe.to_csv(output_file, index=False)
    print(f"Saved processed streamflow CSV: {output_file}")

    return output_file


def create_summary(dataframe: pd.DataFrame) -> Path:
    """Create a short quality-control and flood-event summary."""
    valid_data = dataframe.dropna(subset=["discharge_cfs"])

    if valid_data.empty:
        raise ValueError("All retrieved discharge values are missing.")

    peak_row = valid_data.loc[valid_data["discharge_cfs"].idxmax()]

    summary = f"""USGS Streamflow Download and Quality-Control Summary
=================================================

Project:
Rainfall-to-Streamflow Modeling of Hurricane Florence Flooding
in the Trent River Watershed, North Carolina

Study site:
{SITE_NAME}

USGS monitoring location:
{SITE_ID}

Parameter:
Discharge, cubic feet per second ({PARAMETER_CODE})

Study period:
September 10 through September 22, 2018, Eastern Time

Quality-control summary
-----------------------
Number of downloaded observations: {len(dataframe):,}
Valid discharge observations: {valid_data["discharge_cfs"].count():,}
Missing discharge observations: {dataframe["discharge_cfs"].isna().sum():,}
Duplicate timestamps after processing: {dataframe["time_local"].duplicated().sum():,}
First observation: {dataframe["time_local"].min()}
Last observation: {dataframe["time_local"].max()}

Maximum discharge in downloaded record
--------------------------------------
Peak discharge: {peak_row["discharge_cfs"]:,.0f} cubic feet per second
Peak timestamp: {peak_row["time_local"]}

Interpretation note
-------------------
This observed discharge record will be used as the target hydrograph
for later rainfall-runoff modeling and synthetic storm scenario analysis.
The official USGS Hurricane Florence flood assessment reported an
estimated peak discharge of approximately 67,700 cubic feet per second
at this station on September 16, 2018.
"""

    output_file = RESULTS_DIR / "streamflow_download_summary.txt"
    output_file.write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print(f"Saved summary file: {output_file}")

    return output_file


def plot_hydrograph(dataframe: pd.DataFrame) -> Path:
    """Create a hydrograph of observed streamflow during Hurricane Florence."""
    valid_data = dataframe.dropna(subset=["discharge_cfs"])
    peak_row = valid_data.loc[valid_data["discharge_cfs"].idxmax()]

    figure, axis = plt.subplots(figsize=(11, 6))

    axis.plot(
        valid_data["time_local"],
        valid_data["discharge_cfs"],
        linewidth=2,
        label="Observed USGS discharge",
    )

    axis.scatter(
        peak_row["time_local"],
        peak_row["discharge_cfs"],
        zorder=3,
        label="Observed event peak",
    )

    axis.annotate(
        f'Peak = {peak_row["discharge_cfs"]:,.0f} cfs\n'
        f'{peak_row["time_local"]:%Y-%m-%d %H:%M %Z}',
        xy=(peak_row["time_local"], peak_row["discharge_cfs"]),
        xytext=(25, -55),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )

    axis.set_title(
        "Observed Streamflow During Hurricane Florence\n"
        "USGS 02092500 — Trent River near Trenton, North Carolina"
    )
    axis.set_xlabel("Date and time, Eastern Time")
    axis.set_ylabel("Discharge, cubic feet per second")
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()

    output_file = FIGURES_DIR / "observed_streamflow_hurricane_florence_trent_river.png"
    figure.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved hydrograph figure: {output_file}")

    return output_file


def main() -> None:
    """Execute the streamflow download, processing, summary, and plot workflow."""
    create_directories()
    payload = download_usgs_data()
    streamflow = parse_streamflow_data(payload)
    save_processed_data(streamflow)
    create_summary(streamflow)
    plot_hydrograph(streamflow)

    print()
    print("Streamflow workflow completed successfully.")


if __name__ == "__main__":
    main()