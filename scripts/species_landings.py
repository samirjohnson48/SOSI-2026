"""
This file creates a list of assessed stocks with their corresponding species landings from Fishstat

Species landings for a given stock are the summed landings of that stock's species in that stock's FAO major fishing area(s).

Inputs
    - ./input/global_capture_production.csv: Global capture production data (1950-2022) from Fishstat database
    - ./input/ASFIS_sp_2024.csv: list of ASFIS species in 2024
    - ./output/clean_data/stock_assessments.xlsx: Cleaned list of all assessed stocks
    
Output:
    - ./output/clean_data/species_landings.xlsx: list of all assessed stocks with species landings from 1950 - 2021
    
Output schema (primary key = [Area, ASFIS Scientific Name, Location]):
    - Area: The group of stocks which are found in separate sheets from input
        Most of the time, this is an FAO major fishing area (21, 27, etc.)
        However, this can include other types of aggregations, such as 
        Salmon, Tuna, Deep Sea, and Sharks.
    - ASFIS Scientific Name: The current ASFIS Scientific Name pertaining to the species of the stock
    - Location: The reported location of the stock
    - 1950, ..., 2021: Total landings for years 1950, ..., 2021 for the stock's species in that stock's area(s)
"""

import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm


from utils.species_landings import *
from utils.stock_assessments import get_asfis_mappings


def main():
    # Define directories for input and output files
    parent_dir = os.getcwd()
    input_dir = os.path.join(parent_dir, "input")
    output_dir = os.path.join(parent_dir, os.path.join("output", "clean_data"))

    # Retrieve fishstat data from input folder
    fishstat = pd.read_csv(os.path.join(input_dir, "global_capture_production.csv"))

    # Format fishstat data
    mappings = get_asfis_mappings(input_dir, "ASFIS_sp_2024.csv")
    asfis = mappings["ASFIS"]
    code_to_scientific = dict(zip(asfis["Alpha3_Code"], asfis["Scientific_Name"]))

    fishstat = format_fishstat(fishstat, code_to_scientific)

    # Retrieve assessed stocks from clean_data folder
    stock_assessments = pd.read_excel(
        os.path.join(output_dir, "stock_assessments.xlsx")
    )
    species_landings = stock_assessments[
        ["FAO Areas", "Alpha3_Code", "ASFIS Scientific Name", "Location"]
    ].copy()

    # Expand list of stocks across their FAO Major Fishing Area(s)
    species_landings = explode_stocks(species_landings)

    # Compute species landings for all assessed stocks
    year_start, year_end = 1950, 2021
    years = list(range(year_start, year_end + 1))

    pk = ["FAO Area", "Alpha3_Code", "ASFIS Scientific Name"]

    species_landings = species_landings.drop_duplicates(pk)[pk].copy()

    # Define species with multiple ASFIS entries
    mult_sns = [
        "Actinopterygii",
        "Clupeiformes (=Clupeoidei)",
        "Crustacea",
        "Elasmobranchii",
        "Gobiidae",
        "Mollusca",
        "Palaemonidae",
        "Perciformes (Others)",
        "Testudinata",
    ]

    print("Computing species landings...")
    tqdm.pandas()
    species_landings[years] = species_landings.progress_apply(
        compute_species_landings,
        args=(
            fishstat,
            mult_sns,
        ),
        axis=1,
    )

    # Substitute landings for certain stocks
    subs = [
        [47, ["Sardinella aurita", "Sardinella maderensis"], ["Sardinella spp"]],
        [
            34,
            ["Penaeus notialis, Penaeus monodon, Holthuispenaeopsis atlantic"],
            ["Penaeus spp"],
        ],
    ]

    species_landings = substitute_landings(species_landings, fishstat, subs, years)

    # Include landings from NEI species for certain stocks
    spl = {
        "Mullus spp": [37, "Mullus barbatus"],
        "Sardinella spp": [37, "Sardina pilchardus"],
        "Trachurus spp": [37, "Trachurus mediterraneus"],
    }

    species_landings = add_species_landings(species_landings, fishstat, spl, years)

    print("Species landings computed")

    # Save stocks with species landings
    file_path = os.path.join(output_dir, "species_landings.xlsx")
    print(f"Saving species landings data to {file_path}")
    species_landings.to_excel(file_path, index=False)


if __name__ == "__main__":
    main()
