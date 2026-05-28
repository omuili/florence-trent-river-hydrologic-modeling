"""Download hourly MRMS gauge-corrected precipitation for Hurricane Florence.

Study site:
    Trent River watershed upstream of USGS 02092500, North Carolina

Rainfall product:
    MRMS GaugeCorr_QPE_01H
    Hourly gauge-corrected quantitative precipitation estimate in GRIB2 format

Initial storm window:
    2018-09-13 00:00 UTC through 2018-09-18 00:00 UTC, exclusive

Outputs:
    - Compressed hourly MRMS GRIB2 rainfall files
    - CSV manifest documenting downloaded files
    - Text summary of rainfall download status

Data archive:
    Iowa Environmental Mesonet MRMS archive, referenced by USACE HEC-HMS
    guidance for importing MRMS QPE precipitation data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

import pandas as pd
import requests


# ---------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------

PRODUCT_NAME = "GaugeCorr_QPE_01H"
BASE_URL = "https://mtarchive.geol.iastate.edu"

START_UTC = datetime(2018, 9, 13, 0, 0, tzinfo=timezone.utc)
END_UTC = datetime(2018, 9, 18, 0, 0, tzinfo=timezone.utc)

RAINFALL_DIR = Path("data/raw/rainfall/mrms_gaugecorr_qpe_01h")
RESULTS_DIR = Path("results")

REQUEST_TIMEOUT_SECONDS = 120
MAX_RETRIES = 3


def create_directories() -> None:
    """Create required output directories."""
    RAINFALL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_hourly_times(start: datetime, end: datetime) -> list[datetime]:
    """Generate UTC hourly timestamps from start inclusive to end exclusive."""
    timestamps: list[datetime] = []
    current = start

    while current < end:
        timestamps.append(current)
        current += timedelta(hours=1)

    return timestamps


def build_archive_url(timestamp: datetime) -> tuple[str, str]:
    """Construct archive URL and filename for an hourly MRMS file."""
    date_path = timestamp.strftime("%Y/%m/%d")
    timestamp_string = timestamp.strftime("%Y%m%d-%H0000")
    filename = f"{PRODUCT_NAME}_00.00_{timestamp_string}.grib2.gz"

    url = (
        f"{BASE_URL}/{date_path}/mrms/ncep/"
        f"{PRODUCT_NAME}/{filename}"
    )

    return url, filename


def download_file(url: str, destination: Path) -> tuple[str, int]:
    """Download one MRMS file with retries.

    Returns:
        A status label and file size in bytes.
    """
    if destination.exists() and destination.stat().st_size > 0:
        return "already_exists", destination.stat().st_size

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)

            if response.status_code == 404:
                return "not_found", 0

            response.raise_for_status()
            destination.write_bytes(response.content)

            return "downloaded", destination.stat().st_size

        except requests.RequestException as error:
            if attempt == MAX_RETRIES:
                print(f"Failed after {MAX_RETRIES} attempts: {url}")
                print(f"Error: {error}")
                return "failed", 0

            print(f"Retrying file after request failure: attempt {attempt + 1}")
            time.sleep(2)

    return "failed", 0


def run_access_test() -> None:
    """Download one known Florence-period file to verify archive access."""
    test_time = datetime(2018, 9, 14, 0, 0, tzinfo=timezone.utc)
    url, filename = build_archive_url(test_time)
    destination = RAINFALL_DIR / filename

    print("Running single-file MRMS archive access test...")
    print(f"Test URL: {url}")

    status, size_bytes = download_file(url, destination)

    if status in {"downloaded", "already_exists"}:
        print("MRMS access test successful.")
        print(f"File: {destination}")
        print(f"Size: {size_bytes / 1024:,.1f} KB")
    else:
        raise RuntimeError(
            f"MRMS access test failed with status '{status}'. "
            "Do not proceed to the full download until this is resolved."
        )


def download_event_rainfall() -> pd.DataFrame:
    """Download all hourly MRMS rainfall files in the Florence storm window."""
    timestamps = generate_hourly_times(START_UTC, END_UTC)
    records: list[dict] = []

    print(f"Downloading {len(timestamps)} hourly MRMS rainfall files...")
    print(f"Product: {PRODUCT_NAME}")
    print(f"UTC period: {START_UTC} to {END_UTC}, exclusive")
    print()

    for index, timestamp in enumerate(timestamps, start=1):
        url, filename = build_archive_url(timestamp)
        destination = RAINFALL_DIR / filename

        status, size_bytes = download_file(url, destination)

        records.append(
            {
                "timestamp_utc": timestamp.isoformat(),
                "product": PRODUCT_NAME,
                "filename": filename,
                "local_path": str(destination),
                "source_url": url,
                "status": status,
                "size_bytes": size_bytes,
            }
        )

        print(
            f"[{index:03d}/{len(timestamps):03d}] "
            f"{timestamp:%Y-%m-%d %H:%M UTC} — {status}"
        )

    manifest = pd.DataFrame(records)
    return manifest


def save_manifest_and_summary(manifest: pd.DataFrame) -> None:
    """Save download manifest and concise event-rainfall summary."""
    manifest_file = RESULTS_DIR / "mrms_rainfall_download_manifest.csv"
    manifest.to_csv(manifest_file, index=False)

    expected_files = len(manifest)
    available_files = int(
        manifest["status"].isin(["downloaded", "already_exists"]).sum()
    )
    missing_files = int((manifest["status"] == "not_found").sum())
    failed_files = int((manifest["status"] == "failed").sum())
    total_size_mb = manifest["size_bytes"].sum() / (1024 ** 2)

    summary = f"""MRMS Hurricane Florence Rainfall Download Summary
================================================

Project:
Rainfall-to-Streamflow Modeling of Hurricane Florence Flooding
in the Trent River Watershed, North Carolina

Rainfall product:
MRMS {PRODUCT_NAME}

Product interpretation:
Hourly gauge-corrected quantitative precipitation estimate

Requested UTC period:
{START_UTC} through {END_UTC}, exclusive

Download summary
----------------
Expected hourly files: {expected_files}
Successfully available files: {available_files}
Files not found: {missing_files}
Failed downloads: {failed_files}
Total compressed data size: {total_size_mb:,.2f} MB

Next processing step
--------------------
The downloaded GRIB2 rainfall grids will be clipped to the Trent River
watershed polygon and spatially averaged to construct an hourly
basin-average rainfall time series for comparison with observed USGS
streamflow and later HEC-HMS model input.
"""

    summary_file = RESULTS_DIR / "mrms_rainfall_download_summary.txt"
    summary_file.write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print(f"Saved rainfall download manifest: {manifest_file}")
    print(f"Saved rainfall download summary: {summary_file}")


def main() -> None:
    """Run either a one-file test or the complete event rainfall download."""
    parser = argparse.ArgumentParser(
        description="Download MRMS Hurricane Florence precipitation grids."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Download one known file to verify archive access.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the complete 120-hour Florence rainfall window.",
    )

    args = parser.parse_args()

    create_directories()

    if args.test:
        run_access_test()
    elif args.download:
        manifest = download_event_rainfall()
        save_manifest_and_summary(manifest)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()