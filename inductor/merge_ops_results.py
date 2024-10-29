import pandas as pd
import glob
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl import load_workbook
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge CSV files into a single Excel file."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input folder path containing CSV files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the merged Excel file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Get folder path and output path from arguments
    folder_path = args.input
    output_path = args.output

    # Ensure folder path ends with '/'
    if not folder_path.endswith("/"):
        folder_path += "/"

    # Specify file pattern for CSV files
    file_pattern = "*.csv"

    # Read all CSV files
    csv_files = glob.glob(folder_path + file_pattern)

    # Create an Excel writer object
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for file in csv_files:
            # Read each CSV file into a DataFrame with the correct delimiter
            df = pd.read_csv(file, delimiter=";")

            # Round numeric columns to 2 decimal places
            df = df.round(2)

            # Extract the middle part of the filename as the sheet name
            filename_parts = file.split("/")[-1].split("_")
            sheet_name = "_".join(filename_parts[1:-1])

            # Write each DataFrame to a separate sheet with the extracted name
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"All CSV files merged successfully into {output_path}")


if __name__ == "__main__":
    main()
