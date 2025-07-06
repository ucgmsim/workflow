#!/usr/bin/env python
"""Filter GCMT earthquake solutions.

This script parses earthquakes.json, stores the data in a Pandas DataFrame,
and sorts the earthquakes according to depth.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_earthquake_data(json_file_path: Path) -> dict[str, Any]:
    """Load earthquake data from JSON file.

    Parameters
    ----------
    json_file_path : Path
        Path to the earthquakes.json file.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing earthquake data.
    """
    with open(json_file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_earthquakes_to_dataframe(earthquake_data: dict[str, Any]) -> pd.DataFrame:
    """Parse earthquake data into a pandas DataFrame.

    Parameters
    ----------
    earthquake_data : Dict[str, Any]
        Dictionary containing earthquake data from JSON.

    Returns
    -------
    pd.DataFrame
        DataFrame with earthquake information including depth, magnitude, location, etc.
    """
    records = []

    for event_id, event_data in earthquake_data.items():
        # Extract basic event information
        record = {
            "event_id": event_id,
            "date": event_data["date"],
            "magnitude": event_data["magnitude"],
            "depth": event_data["location"]["depth"],
            "latitude": event_data["location"]["latitude"],
            "longitude": event_data["location"]["longitude"],
        }

        # Extract nodal plane information
        # Most events have 2 nodal planes, we'll take the first one as primary
        if event_data["nodalPlanes"]:
            primary_plane = event_data["nodalPlanes"][0]
            record.update(
                {
                    "dip": primary_plane["dip"],
                    "rake": primary_plane["rake"],
                    "strike": primary_plane["strike"],
                }
            )

            # If there's a second nodal plane, include it with suffix
            if len(event_data["nodalPlanes"]) > 1:
                secondary_plane = event_data["nodalPlanes"][1]
                record.update(
                    {
                        "dip_2": secondary_plane["dip"],
                        "rake_2": secondary_plane["rake"],
                        "strike_2": secondary_plane["strike"],
                    }
                )

        records.append(record)

    return pd.DataFrame(records)


def filter_and_sort_earthquakes(
    df: pd.DataFrame,
    min_magnitude: float | None = None,
    max_magnitude: float | None = None,
    min_depth: float | None = None,
    max_depth: float | None = None,
) -> pd.DataFrame:
    """Filter and sort earthquake DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing earthquake data.
    min_magnitude : float, optional
        Minimum magnitude filter.
    max_magnitude : float, optional
        Maximum magnitude filter.
    min_depth : float, optional
        Minimum depth filter (km).
    max_depth : float, optional
        Maximum depth filter (km).

    Returns
    -------
    pd.DataFrame
        Filtered and sorted DataFrame.
    """
    filtered_df = df.copy()

    # Apply filters if specified
    if min_magnitude is not None:
        filtered_df = filtered_df[filtered_df["magnitude"] >= min_magnitude]

    if max_magnitude is not None:
        filtered_df = filtered_df[filtered_df["magnitude"] <= max_magnitude]

    if min_depth is not None:
        filtered_df = filtered_df[filtered_df["depth"] >= min_depth]

    if max_depth is not None:
        filtered_df = filtered_df[filtered_df["depth"] <= max_depth]

    # Sort by depth (shallow to deep)
    filtered_df = filtered_df.sort_values("depth", ascending=True)

    return filtered_df.reset_index(drop=True)


def print_summary_statistics(df: pd.DataFrame) -> None:
    """Print summary statistics for the earthquake dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing earthquake data.
    """
    print("=== Earthquake Dataset Summary ===")
    print(f"Total earthquakes: {len(df)}")
    print()

    print("Magnitude statistics:")
    print(f"  Min: {df['magnitude'].min():.1f}")
    print(f"  Max: {df['magnitude'].max():.1f}")
    print(f"  Mean: {df['magnitude'].mean():.2f}")
    print(f"  Median: {df['magnitude'].median():.2f}")
    print()

    print("Depth statistics (km):")
    print(f"  Min: {df['depth'].min():.1f}")
    print(f"  Max: {df['depth'].max():.1f}")
    print(f"  Mean: {df['depth'].mean():.2f}")
    print(f"  Median: {df['depth'].median():.2f}")
    print()

    print("Depth distribution:")
    depth_bins = [0, 10, 30, 70, 150, float("inf")]
    depth_labels = ["0-10km", "10-30km", "30-70km", "70-150km", ">150km"]
    df["depth_category"] = pd.cut(
        df["depth"], bins=depth_bins, labels=depth_labels, right=False
    )
    depth_counts = df["depth_category"].value_counts().sort_index()
    for category, count in depth_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")


def main() -> None:
    """Main function to process earthquake data."""
    # Path to the earthquakes.json file
    json_file = Path(__file__).parent / "earthquakes.json"

    if not json_file.exists():
        print(f"Error: {json_file} not found!")
        return

    print(f"Loading earthquake data from {json_file}...")
    earthquake_data = load_earthquake_data(json_file)

    print("Parsing data into DataFrame...")
    df = parse_earthquakes_to_dataframe(earthquake_data)

    print("Filtering for depths between 4-6 km and sorting by magnitude...")
    sorted_df = filter_and_sort_earthquakes(df, min_depth=4.0, max_depth=6.0)
    # Sort by magnitude instead of depth
     = sorted_df.sort_values("magnitude", ascending=False).reset_index(
        drop=True
    )

    # Print summary statistics
    print_summary_statistics(sorted_df)

    print("\n=== Highest Magnitude Earthquakes (4-6 km depth) ===")
    print(
        sorted_df[["event_id", "date", "magnitude", "depth", "latitude", "longitude"]]
        .head(10)
        .to_string(index=False)
    )

    print("\n=== Lowest Magnitude Earthquakes (4-6 km depth) ===")
    print(
        sorted_df[["event_id", "date", "magnitude", "depth", "latitude", "longitude"]]
        .tail(10)
        .to_string(index=False)
    )

    # Save the sorted DataFrame to CSV
    output_file = Path(__file__).parent / "earthquakes_4_6km_depth_by_magnitude.csv"
    sorted_df.to_csv(output_file, index=False)
    print(f"\nFiltered earthquake data saved to: {output_file}")

    # Example: Filter for shallow, large earthquakes
    print("\n=== Large Shallow Earthquakes (M≥5.0, depth≤30km) ===")
    large_shallow = filter_and_sort_earthquakes(df, min_magnitude=5.0, max_depth=30)
    if len(large_shallow) > 0:
        print(
            large_shallow[
                ["event_id", "date", "magnitude", "depth", "latitude", "longitude"]
            ].to_string(index=False)
        )
    else:
        print("No earthquakes match these criteria.")


if __name__ == "__main__":
    main()
