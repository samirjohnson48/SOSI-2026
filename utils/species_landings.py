"""
This file includes all functions used for collection of species landings 
from the FAO Fishstat database

These functions are implemented in ./main/fishstat_landings.py
"""

import pandas as pd
import numpy as np


def format_fishstat(fishstat, code_to_scientific=[], year_start=1950, year_end=2022):
    fishstat = fishstat.drop(columns=["Unit"])

    rename_dict = {f"[{year}]": year for year in range(year_start, year_end + 1)}
    rename_dict["Country (ISO3 code)"] = "ISO3"
    rename_dict["ASFIS species (3-alpha code)"] = "Alpha3_Code"
    rename_dict["FAO major fishing area (Code)"] = "Area"
    rename_dict["Unit (Name)"] = "Unit"
    fishstat = fishstat.rename(columns=rename_dict)

    if code_to_scientific:
        fishstat["ASFIS Scientific Name"] = fishstat["Alpha3_Code"].map(
            code_to_scientific
        )

    return fishstat


def explode_stocks(species_landings, key="FAO Areas"):
    sl = species_landings.copy()

    sl[key] = sl[key].astype(str)

    sl["FAO Area"] = sl[key].apply(lambda x: x.split(", "))

    sl = sl.explode("FAO Area").reset_index(drop=True)

    sl["FAO Area"] = sl["FAO Area"].apply(
        lambda x: int(x) if x.isdigit() else print(f"{x} cannot be cast to type int")
    )

    sl = sl.drop(columns=key)

    return sl


def compute_species_landings(
    row,
    fishstat,
    mult_sns=[],
    year_start=1950,
    year_end=2021,
    key="ASFIS Scientific Name",
):
    scientific_name, area = row[key], row["FAO Area"]
    years = list(range(year_start, year_end + 1))

    area_mask = fishstat["Area"] == area

    # Create mask for scientific name
    # Handle species listed by commas
    fishstat_sn = fishstat["ASFIS Scientific Name"].unique()
    if ", " in scientific_name and scientific_name not in fishstat_sn:
        scientific_names = [
            sn for sn in scientific_name.split(", ") if sn in fishstat_sn
        ]
        sn_mask = fishstat["ASFIS Scientific Name"].isin(scientific_names)
    elif scientific_name in mult_sns:
        sn_mask = fishstat["Alpha3_Code"] == row["Alpha3_Code"]
    else:
        sn_mask = fishstat["ASFIS Scientific Name"] == scientific_name

    # If no matching scientific names, return missing values
    if sum(sn_mask) == 0:
        return pd.Series([np.nan] * len(years), index=years)

    return fishstat[area_mask & sn_mask][years].sum()


def substitute_landings(species_landings, fishstat, subs, years):
    sl = species_landings.copy()

    for sub in subs:
        area = sub[0]
        stocks = sub[1]
        sub_stocks = sub[2]
        n_stocks = len(stocks)

        sl_area_mask = sl["FAO Area"] == area
        sl_stocks_mask = sl["ASFIS Scientific Name"].isin(stocks)

        fs_area_mask = fishstat["Area"] == area
        fs_stocks_mask = fishstat["ASFIS Scientific Name"].isin(sub_stocks)

        landings = fishstat[fs_area_mask & fs_stocks_mask][years].sum() / n_stocks

        sl.loc[sl_area_mask & sl_stocks_mask, years] = landings.values

    return sl


def add_species_landings(species_landings, fishstat, spl, years):
    sl = species_landings.copy()

    for species_to_add, species_info in spl.items():
        # Area is the area of the species
        # species is the list of species where the extra landings are added
        # distribution is the weights over the list of species for the extra landings
        if len(species_info) == 3:
            area, species, distribution = species_info
        elif len(species_info) == 2:
            # If no distribution provided, use uniform distribution
            area, species = species_info

            # Make sure species is list if single species is given
            if not isinstance(species, list):
                species = [species]

            n = len(species)
            distribution = list(np.ones(n) / n)

        # Normalize distribution if not done already
        s = sum(distribution)
        if s != 1:
            distribution = [d / s for d in distribution]

        species_mask = fishstat["ASFIS Scientific Name"] == species_to_add
        area_mask = fishstat["Area"] == area

        landings_to_add = fishstat[species_mask & area_mask][years].sum()

        for sp, w in zip(species, distribution):
            lta = landings_to_add * w

            sl_species_mask = sl["ASFIS Scientific Name"] == sp
            sl_area_mask = sl["FAO Area"] == area

            sl.loc[sl_species_mask & sl_area_mask, years] += lta

    return sl
