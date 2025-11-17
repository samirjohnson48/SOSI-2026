"""
This file creates full table of all assessed stocks (and unassessed stocks)

Inputs
    - ./input/updated_assessment_overview.xlsx: collection of spreadsheets of stock assessments for each area
    - ./input/ASFIS_sp_2024.csv: list of ASFIS species in 2024
    - ./input/ASFIS_changes_2024.xlsx: list of name updates for ASFIS species in 2024
    - ./input/deep_sea_name_map.json: mapping from name in data to ASFIS Scientific Name for Deep Sea sheet
    - ./input/corrected_scientific_names.json: mapping of corrected ASFIS Scientific Names for the report
    - ./input/overview_2025-02-04.xlsx: a corrected stock assessments sheet with changes made to Angolan stocks in Area 47

Outputs
    - ./output/clean_data/stock_assessments_w_unassessed.xlsx: complete list of assessed and unassessed stocks from sheets
    - ./output/clean_data/stock_assessments.xlsx: complete list of assessed stocks (Status is 'U', 'M' or 'O')
    - ./output/clean_data/special_group_stocks.xlsx: list of all assessed species within special groups (Deep Sea, Salmon, Sharks, and Tuna)

Output schema for stock assessments(primary key = [Area, ASFIS Scientific Name, Location]):
    - Area: The group of stocks which are found in separate sheets from input
        Most of the time, this is an FAO major fishing area (21, 27, etc.)
        However, this can include other types of aggregations, such as 
        Salmon, Tuna, Deep Sea, and Sharks.
    - ISSCAAP Code: ISSCAAP Code corresponding to the corresponding to the species of the stock
    - ASFIS Name: The current ASFIS common name corresponding to the species of the stock
    - ASFIS Scientific Name: The current ASFIS Scientific Name pertaining to the species of the stock
    - Location: The reported location of the stock
    - Tier: The tier of the assessment (1, 2, or 3)
    - Status: The status of the assessment, standardized to be U, M, O, or NaN 
            (Underfished, Maximally Sustainably Fished, Overfished, or Unknown, resp.)
    - Uncertainty: The uncertainty of the assessment, standardized to be L, M, or H (low, medium, high, resp.)
"""

# Silence the Pandas future warnings on output
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import pandas as pd
import numpy as np

from utils.stock_assessments import *


def main():
    # Define directories for input and output files
    parent_dir = os.getcwd()
    input_dir = os.path.join(parent_dir, "input")
    output_dir = os.path.join(parent_dir, os.path.join("output", "clean_data"))

    os.makedirs(output_dir, exist_ok=True)

    # Read in stock reference list
    stock_reference = pd.read_excel(
        os.path.join(input_dir, "stock_reference_list.xlsx")
    )

    # Add alpha3 codes to species
    asfis = get_asfis_mappings(input_dir, "ASFIS_sp_2024.csv")["ASFIS"]

    sn_to_code = dict(zip(asfis["Scientific_Name"], asfis["Alpha3_Code"]))
    name_to_code = dict(zip(asfis["English_name"], zip(asfis["Alpha3_Code"])))
    name_to_code = {k: v[0] for k, v in name_to_code.items()}

    alpha_count = asfis.groupby("Scientific_Name")["Alpha3_Code"].nunique()

    mult_sns = list(alpha_count[alpha_count > 1].index)

    mult_mask = stock_reference["ASFIS Scientific Name"].isin(mult_sns)

    stock_reference["Alpha3_Code"] = stock_reference["ASFIS Scientific Name"].map(
        sn_to_code
    )
    stock_reference.loc[mult_mask, "Alpha3_Code"] = stock_reference.loc[
        mult_mask, "ASFIS Name"
    ].map(name_to_code)

    # -- Validaiton of stock reference list --
    # Check uniqueness and non-nullity of primary key ASFIS Scientific Name, Location
    validate_primary_key(stock_reference)

    # Check that ASFIS Names and ISSCAAP Codes are consistent
    # for a given ASFIS Scientific Name
    validate_consistent_values(stock_reference)

    # Only keep assessed stocks for stock assessments list
    assessed_mask = stock_reference["Status"].isin(["U", "M", "O"])

    stock_assessments = stock_reference[assessed_mask].copy()

    print(f"Saving output files to {output_dir}")
    stock_assessments.to_excel(
        os.path.join(output_dir, "stock_assessments.xlsx"), index=False
    )


if __name__ == "__main__":
    main()
