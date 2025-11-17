"""

"""

import pandas as pd
import os
from tqdm import tqdm
import matplotlib as mpl

mpl.rcdefaults()

# Make PDF font editable
mpl.rcParams["pdf.fonttype"] = 42

from utils.capture_production_figures import *
from utils.species_landings import format_fishstat
from utils.stock_assessments import get_asfis_mappings


def main():
    # Define directories for input and output files
    parent_dir = os.getcwd()
    input_dir = os.path.join(parent_dir, "input")
    output_dir = os.path.join(parent_dir, "output", "figures")
    table_dir = os.path.join(parent_dir, "output", "aggregate_tables")

    os.makedirs(output_dir, exist_ok=True)

    # Retrieve fishstat data from input folder
    fishstat = pd.read_csv(os.path.join(input_dir, "global_capture_production.csv"))
    
    # Remove the entries reported by number
    number_mask = fishstat["Unit (Name)"] == "Number"
    fishstat = fishstat[~number_mask].reset_index(drop=True)

    # Format fishstat data
    mappings = get_asfis_mappings(input_dir, "ASFIS_sp_2025.csv")
    asfis = mappings["ASFIS"]
    code_to_scientific = dict(zip(asfis["Alpha3_Code"], asfis["Scientific_Name"]))
    code_to_isscaap = dict(zip(asfis["Alpha3_Code"], asfis["ISSCAAP_Group_Code"]))

    fishstat = format_fishstat(fishstat, code_to_scientific)
    fishstat["ISSCAAP Code"] = fishstat["Alpha3_Code"].map(code_to_isscaap)

    # Remove all rows without Alpha 3 species code
    fishstat = fishstat.dropna(subset="Alpha3_Code")

    # Set variables for the first and last year of data to be used
    first_year, last_year = 1950, 2023
    years = list(range(first_year, last_year + 1))

    # Define list of ISSCAAP codes to be removed from capture data
    # This include mammals and algae
    isscaap_to_remove = [61, 62, 63, 64, 71, 72, 73, 74, 81, 82, 83, 91, 92, 93, 94]

    isscaap_mask = ~fishstat["ISSCAAP Code"].isin(isscaap_to_remove)
    fishstat = fishstat[isscaap_mask]

    # Define list of FAO major fishing areas to keep in capture data
    areas_to_keep = [
        21,
        27,
        31,
        34,
        37,
        41,
        47,
        51,
        57,
        61,
        67,
        71,
        77,
        81,
        87,
        "48, 58, 88",
    ]
    mask_485888 = fishstat["Area"].isin([48, 58, 88])
    fishstat["Area"] = fishstat["Area"].astype(object)
    fishstat.loc[mask_485888, "Area"] = "48, 58, 88"

    area_mask = fishstat["Area"].isin(areas_to_keep)
    fishstat = fishstat[area_mask]

    # Retrieve top 10 species per area
    # This function also saves this data to aggregate_tables folder
    top_species_fp = os.path.join(table_dir, "top10species_by_area.xlsx")
    top_species_by_area = compute_top_species_by_area(
        fishstat, top_species_fp, year=last_year, num_species=10
    )

    # Data from figures to be stored here
    capture_peaks = {}
    capture_production = {}
    decade_averages = {}

    # Group the Fishstat data by area
    capture_by_area = fishstat.groupby(["Area"]).sum().reset_index()[["Area"] + years]

    # Generate and save figures
    for area in tqdm(areas_to_keep, desc="Capture Production Figures"):
        # Retrieve top species for area
        top_species = top_species_by_area[top_species_by_area["Area"] == area][
            "top_species"
        ].values[0]

        cp_area, dec_ave, cps = create_capture_production_figure(
            area, capture_by_area, top_species, fishstat, output_dir
        )

        capture_production[area] = cp_area
        decade_averages[area] = dec_ave
        capture_peaks[area] = cps

    print(f"Capture production figures saved to {output_dir}")

    # Save figure data
    with pd.ExcelWriter(
        os.path.join(table_dir, "capture_production_figures_data.xlsx")
    ) as writer:
        for area, production in capture_production.items():
            comb = pd.DataFrame(
                columns=["Aggregation"] + list(range(first_year, last_year + 1))
            )
            for key, production_ts in production.items():
                comb.loc[len(comb)] = [key] + list(production_ts)

            for decade, value in decade_averages[area].items():
                comb.loc[comb["Aggregation"] == "total", decade] = value

            comb.to_excel(writer, sheet_name=f"Area {area}", index=False)


if __name__ == "__main__":
    main()
