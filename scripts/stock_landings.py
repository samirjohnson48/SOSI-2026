"""

"""

import os
import pandas as pd
import numpy as np
import json

from utils.stock_landings import *
from utils.stock_assessments import get_asfis_mappings
from utils.species_landings import format_fishstat


def main():
    # Define directories for input and output files
    parent_dir = os.getcwd()
    input_dir = os.path.join(parent_dir, "input")
    output_dir = os.path.join(parent_dir, os.path.join("output", "clean_data"))

    # Retrieve the species landings
    species_landings = pd.read_excel(os.path.join(output_dir, "species_landings.xlsx"))

    # Compute the missing species landings
    sp_map = pd.read_excel(os.path.join(input_dir, "SOS_species_map_landings.xlsx"))

    spl = compute_missing_species_landings(sp_map, species_landings)

    # Retrieve the weights
    weights = pd.read_excel(os.path.join(output_dir, "stock_weights.xlsx"))

    # --- First compute the default landings ---
    # These are used in the case of no available landings or proxy landings
    # Based on reallocation of NEI landings
    # Merge the two dataframes
    stock_landings = pd.merge(
        spl,
        weights,
        on=["FAO Area", "ASFIS Scientific Name"],
    )
    stock_landings = stock_landings.rename(columns={2021: "Species Landings 2021"})
    cols_to_keep = [
        "FAO Area",
        "ASFIS Scientific Name",
        "Location",
        "Species Landings 2021",
        "Normalized Weight",
    ]
    stock_landings = stock_landings[cols_to_keep]

    # Add Num Stocks column for computing landings
    stock_landings = compute_num_stocks(stock_landings)

    stock_landings["Stock Landings 2021"] = stock_landings.apply(
        compute_landings, axis=1
    )
    # stock_landings = stock_landings.drop(columns="Num Stocks")

    # For the remaining stocks with missing landings, we use the NEI species corresponding
    # to the stock's ISSCAAP Code. We split the landings according to the distribution
    # of status among stocks with landings
    # We retrieve the status of stocks for the distribution
    primary_key = ["ASFIS Scientific Name", "Location"]
    stock_assessments = pd.read_excel(
        os.path.join(output_dir, "stock_assessments.xlsx")
    )
    stock_assessments = stock_assessments[
        primary_key + ["ISSCAAP Code", "ASFIS Name", "Status", "Analysis Group"]
    ]
    stock_landings = pd.merge(stock_landings, stock_assessments, on=primary_key)

    # Retrieve ISSCAAP Code to NEI species mapping
    with open(os.path.join(input_dir, "NEI_to_ISSCAAP.json"), "r") as file:
        nei_to_isscaap = json.load(file)

    # Retrieve fishstat and ASFIS data for NEI landings
    fishstat = pd.read_csv(os.path.join(input_dir, "global_capture_production.csv"))
    mappings = get_asfis_mappings(input_dir, "ASFIS_sp_2024.csv")
    asfis = mappings["ASFIS"]
    code_to_scientific = dict(zip(asfis["Alpha3_Code"], asfis["Scientific_Name"]))
    code_to_name = dict(zip(asfis["Alpha3_Code"], asfis["English_name"]))

    fishstat = format_fishstat(fishstat, code_to_scientific)
    fishstat["ASFIS Name"] = fishstat["Alpha3_Code"].map(code_to_name)

    # Get list of analysis groups which are not in special groups
    sg_df = pd.read_excel(os.path.join(input_dir, "special_groups_species.xlsx"))
    sg_list = sg_df["Analysis Group"].unique()
    numerical_ags = [
        ag for ag in stock_landings["Analysis Group"].unique() if ag not in sg_list
    ]
    stock_landings = compute_missing_stock_landings(
        stock_landings, fishstat, numerical_ags, nei_to_isscaap
    )

    # Save assigned landings to output file
    cols_to_save = [
        "FAO Area",
        "ASFIS Scientific Name",
        "Location",
        "Stock Landings 2021",
    ]
    stock_landings = stock_landings[cols_to_save]

    stock_landings.to_excel(
        os.path.join(output_dir, "stock_landings.xlsx"), index=False
    )


if __name__ == "__main__":
    main()
