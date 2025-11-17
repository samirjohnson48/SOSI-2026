"""

"""

import os
import pandas as pd
import numpy as np
import json

from utils.aggregate_tables import *
from utils.stock_assessments import get_asfis_mappings, read_stock_data
from utils.species_landings import format_fishstat, explode_stocks


def main():
    # Define directories for input and output files
    parent_dir = os.getcwd()
    input_dir = os.path.join(parent_dir, "input")
    clean_data_dir = os.path.join(parent_dir, os.path.join("output", "clean_data"))
    output_dir = os.path.join(parent_dir, os.path.join("output", "aggregate_tables"))

    os.makedirs(output_dir, exist_ok=True)

    # Retrieve ASFIS mappings
    asfis_mapping = get_asfis_mappings(input_dir, "ASFIS_sp_2024.csv")
    asfis = asfis_mapping["ASFIS"]
    scientific_to_isscaap = asfis_mapping["ASFIS Scientific Name to ISSCAAP Code"]
    scientific_names = asfis_mapping["ASFIS Scientific Names"]

    # -- Tables based on number --
    print("Computing tables based on number...")
    # Retrieve stock lists (assessed/unassessed and only assessed stocks)
    stock_reference = pd.read_excel(
        os.path.join(input_dir, "stock_reference_list.xlsx")
    )
    stock_assessments = pd.read_excel(
        os.path.join(clean_data_dir, "stock_assessments.xlsx")
    )

    # Compute Status by Number grouped by Area and Tier
    sbn_area = compute_status_by_number(stock_assessments, "Analysis Group")
    sbn_tier = compute_status_by_number(stock_assessments, "Tier")
    
    # Compute Status by Number grouped by Area weighted by Tier
    sbn_area_v2 = compute_weighted_status_by_number(stock_assessments)
    
    vc_tier_area = compute_count_for_group(stock_assessments, group_col="Analysis Group", count_col="Tier")
    
    # Retrieve SOFIA Status by Number
    sofia_indices = {
        "Area 21": (46, 0, 0),
        "Area27": (40, 0, 0),
        "Area 31": (51, 0, 0),
        "Area34": (71, 0, 0),
        "Area37": (60, 0, 0),
        "Area41": (62, 0, 0),
        "Area47": (44, 0, 0),
        "Area51": (52, 0, 0),
        "Area57": (64, 0, 0),
        "Area 61": (46, 0, 0),
        "Area67": (41, 0, 0),
        "Area71": (63, 0, 0),
        "Area77": (33, 0, 0),
        "area81v2": (38, 0, 0),
        "Area87": (31, 0, 0),
        "Tunas_HilarioISSF": (19, 0, 0),
    }

    sofia_sheets = sofia_indices.keys()
    sofia_sheet_to_area = {
        sheet: "".join([char for char in sheet if char.isdigit()])
        for sheet in sofia_sheets
    }
    sofia_sheet_to_area = {
        sheet: f"Area {area}" if area.isdigit() else area
        for sheet, area in sofia_sheet_to_area.items()
    }
    sofia_sheet_to_area["area81v2"] = "Area 81"
    sofia_sheet_to_area["Tunas_HilarioISSF"] = "Tuna"

    sofia_file_path = os.path.join(
        input_dir, "sofia2024.xlsx"
    )
    sbn_sofia_dict = read_stock_data(
        sofia_file_path, sofia_indices, desc="SOFIA Sheets"
    )

    # Reformat SOFIA status by number
    for sheet, df in sbn_sofia_dict.items():
        sbn_sofia_dict[sheet]["Analysis Group"] = sofia_sheet_to_area[sheet]
        sbn_sofia_dict[sheet] = df[
            ["Analysis Group", "Overfished", "Fully Fished ", "Under fished"]
        ]
        sbn_sofia_dict[sheet] = sbn_sofia_dict[sheet].rename(
            columns={
                "Overfished": "No. of O",
                "Fully Fished ": "No. of MSF",
                "Under fished": "No. of U",
            }
        )
        sbn_sofia_dict[sheet]["No. of Sustainable"] = (
            sbn_sofia_dict[sheet]["No. of U"] + sbn_sofia_dict[sheet]["No. of MSF"]
        )
        sbn_sofia_dict[sheet]["No. of Unsustainable"] = sbn_sofia_dict[sheet][
            "No. of O"
        ]
        sbn_sofia_dict[sheet]["No. of stocks"] = (
            sbn_sofia_dict[sheet]["No. of Sustainable"]
            + sbn_sofia_dict[sheet]["No. of Unsustainable"]
        )

    sbn_sofia = pd.DataFrame()

    for sheet, df in sbn_sofia_dict.items():
        if sbn_sofia.empty:
            sbn_sofia = df.copy()
        else:
            sbn_sofia = pd.concat([sbn_sofia, df])

    sbn_sofia = pd.concat(
        [sbn_sofia, pd.DataFrame({"Analysis Group": "Global"}, index=[len(sbn_sofia)])]
    )

    cols_to_sum = [
        "No. of stocks",
        "No. of U",
        "No. of MSF",
        "No. of O",
        "No. of Sustainable",
        "No. of Unsustainable",
    ]
    sbn_sofia.loc[sbn_sofia["Analysis Group"] == "Global", cols_to_sum] = (
        sbn_sofia[cols_to_sum].sum().values
    )

    pct_cols = []
    for col in cols_to_sum:
        sbn_sofia[col] = sbn_sofia[col].astype(int)
        if col != "No. of stocks":
            pct_col = col.replace("No. of ", "") + " (%)"
            pct_cols.append(pct_col)
            sbn_sofia[pct_col] = (sbn_sofia[col] / sbn_sofia["No. of stocks"]) * 100

    sbn_col_order = ["Analysis Group"] + cols_to_sum + pct_cols
    sbn_sofia = sbn_sofia[sbn_col_order]

    # Compare status by number for the two methods
    sbn_comp = compare_status_by_number(sbn_area, sbn_sofia)

    # Save status by number files
    sbn_area_fp = os.path.join(output_dir, "status_by_number_area.xlsx")
    sbn_area.to_excel(sbn_area_fp, index=False)
    round_excel_file(sbn_area_fp)

    sbn_tier_fp = os.path.join(output_dir, "status_by_number_tier.xlsx")
    sbn_tier.to_excel(sbn_tier_fp, index=False)
    round_excel_file(sbn_tier_fp)
    
    sbn_area_v2_fp = os.path.join(output_dir, "status_by_number_weighted_by_tier.xlsx")
    sbn_area_v2.to_excel(sbn_area_v2_fp)
    round_excel_file(sbn_area_v2_fp)

    sbn_comp_fp = os.path.join(output_dir, "compare_status_by_number.xlsx")
    sbn_comp.to_excel(sbn_comp_fp)
    round_excel_file(sbn_comp_fp)
    
    # Save value counts for tier across areas
    vc_tier_area_fp = os.path.join(output_dir, "value_counts_tier_by_area.xlsx")
    vc_tier_area.to_excel(vc_tier_area_fp)

    # Create summary of stocks and save file
    sos = compute_summary_of_stocks(stock_reference)
    sos_fp = os.path.join(output_dir, "summary_of_stocks.xlsx")
    sos.to_excel(sos_fp)
    
    # Create summary of areas by tier and save file
    summary_area_tier = compute_summary_area_by_tier(stock_assessments)
    summary_area_tier_fp = os.path.join(output_dir, "summary_of_areas_by_tier.xlsx")
    
    with pd.ExcelWriter(summary_area_tier_fp) as writer:
        for tier, summary in summary_area_tier.items():
            summary.to_excel(writer, sheet_name=tier)
            
    round_excel_file(summary_area_tier_fp)

    # Save same aggregations for individual areas
    area_summary_dir = os.path.join(output_dir, "Analysis Group Statistics")
    os.makedirs(area_summary_dir, exist_ok=True)

    for ag in stock_assessments["Analysis Group"].unique():
        sbna = sbn_area[sbn_area["Analysis Group"] == ag]
        sbnt = compute_status_by_number(
            stock_assessments[stock_assessments["Analysis Group"] == ag], "Tier"
        )
        sosa = compute_summary_of_stocks(
            stock_reference[stock_reference["Analysis Group"] == ag]
        )
        cbn = sbn_comp[sbn_comp[("", "Analysis Group")] == ag]
        
        sat = pd.DataFrame()
        for tier, summary in summary_area_tier.items():
            if ag in summary.index:
                tier_sat = summary.loc[ag].to_frame().T
                tier_sat.index=[tier]
                sat = pd.concat([sat, tier_sat])

        area_summary_fp = os.path.join(area_summary_dir, f"{ag}_summary.xlsx")
        
        with pd.ExcelWriter(area_summary_fp) as writer:
            sbna.to_excel(writer, sheet_name="Status by Number")
            sbnt.to_excel(writer, sheet_name="Status by Tier")
            sosa.to_excel(writer, sheet_name="Summary of Stocks")
            sat.to_excel(writer, sheet_name="Summary of Area by Tier")

            if ag in sbn_comp[("", "Analysis Group")].unique():
                cbn.to_excel(writer, sheet_name="Comparison by Number")

        round_excel_file(area_summary_fp)

    # -- Tables based on fishstat landings --
    print("Computing tables based on Fishstat data...")
    # Retrieve fishstat data from input folder
    fishstat = pd.read_csv(os.path.join(input_dir, "global_capture_production.csv"))

    # Format fishstat data
    mappings = get_asfis_mappings(input_dir, "ASFIS_sp_2024.csv")
    asfis = mappings["ASFIS"]
    code_to_scientific = dict(zip(asfis["Alpha3_Code"], asfis["Scientific_Name"]))

    fishstat = format_fishstat(fishstat, code_to_scientific)

    # Only keep data from FAO major fishing areas in analysis
    numerical_areas = set(areas for ag in stock_assessments["Analysis Group"].unique() for areas in get_numbers_from_string(ag))

    fishstat_area_mask = fishstat["Area"].isin(numerical_areas)
    fishstat = fishstat[fishstat_area_mask]

    # Add ISSCAAP Code to capture data
    fishstat["ISSCAAP Code"] = fishstat["ASFIS Scientific Name"].map(
        scientific_to_isscaap
    )
    
    # Drop 2022 data from fishstat
    fishstat = fishstat.drop(columns=2022)

    # Compute status by number for top ten species globally and save file
    top_ten_species = [
        "Engraulis ringens",
        "Gadus chalcogrammus",
        "Katsuwonus pelamis",
        "Clupea harengus",
        "Thunnus albacares",
        "Micromesistius poutassou",
        "Sardina pilchardus",
        "Scomber japonicus",
        "Gadus morhua",
        "Sardinops sagax",
    ]
    sbn_top10_species = compute_species_status_by_number(
        stock_assessments, top_ten_species, fishstat
    )
    sbn_top10_fp = os.path.join(output_dir, "top10_species_status_by_number.xlsx")
    sbn_top10_species.to_excel(sbn_top10_fp)
    round_excel_file(sbn_top10_fp)

    # -- Tables based on species landings --
    print("Computing tables based on species landings...")
    # Build appendix landings tables
    # Retrieve species landings
    species_landings = pd.read_excel(
        os.path.join(clean_data_dir, "species_landings.xlsx")
    )

    # Add ISSCAAP Code, ASFIS Name, Status, and Uncertainty to species landings
    stock_assessments_ext = explode_stocks(stock_assessments)
    species_landings = pd.merge(species_landings, stock_assessments_ext, on=["FAO Area", "ASFIS Scientific Name"])

    # Retrieve aquaculture landings and reformat
    aquaculture = pd.read_csv(
        os.path.join(input_dir, "global_aquaculture_production.csv")
    )
    aquaculture = format_fishstat(aquaculture)
    aquaculture = aquaculture.rename(
        columns={
            "ASFIS species (Name)": "ASFIS Name",
            "ASFIS species (Code)": "ISSCAAP Code",
            "ASFIS species (Scientific name)": "ASFIS Scientific Name"
        }
    )
    aquaculture = aquaculture.drop(columns=2022)

    # Only keep data for relevant areas
    ac_area_mask = aquaculture["Area"].isin(numerical_areas)

    aquaculture = aquaculture[ac_area_mask]

    # Create ISSCAAP Group code to name map
    isscaap_code_to_name = dict(
        zip(asfis["ISSCAAP_Group_Code"], asfis["ISSCAAP_Group_Name_EN"])
    )

    # Retrieve country data and create ISO3 to name map
    country_codes = pd.read_excel(
        os.path.join(input_dir, "NOCS.xlsx"), sheet_name="Codes"
    )
    iso3_to_name = dict(zip(country_codes["ISO3"], country_codes["LIST NAME"]))
    iso3_to_name["EAZ"] = "Zanzibar"

    # Define ISSCAAP Codes to remove from appendix landings,
    # and later the percent coverage calculations as well.
    # (unless they appear in assessment, then they are added back in to total area landings)
    isscaap_to_remove = [61, 62, 63, 64, 71, 72, 73, 74, 81, 82, 83, 91, 92, 93, 94]
    
    # Define the special groups with the list of species
    special_groups_df = pd.read_excel(os.path.join(input_dir, "special_groups_species.xlsx"))
    special_groups = create_sg_dict(special_groups_df)
    
    # Compute appendix landings
    appendix_decs, appendix_years = compute_appendix_landings(
        species_landings,
        fishstat,
        aquaculture,
        isscaap_to_remove,
        isscaap_code_to_name,
        scientific_names,
        iso3_to_name,
        special_groups
    )
    
    # Save appendix landings files
    # By decade
    appendix_decs_fp = os.path.join(output_dir, "appendix_landings_decades.xlsx")

    with pd.ExcelWriter(appendix_decs_fp) as writer:
        for area, summary in appendix_decs.items():
            summary.to_excel(writer, sheet_name=str(area))

    round_excel_file(
        appendix_decs_fp,
        decimal_places=0,
        lt_one=True,
        skiprows=1
    )

    # By year
    appendix_years_fp = os.path.join(output_dir, "appendix_landings_years.xlsx")

    with pd.ExcelWriter(appendix_years_fp) as writer:
        for area, summary in appendix_years.items():
            summary.to_excel(writer, sheet_name=str(area))

    round_excel_file(
        appendix_years_fp,
        decimal_places=0,
        lt_one=True,
        skiprows=1
    )

    # -- Tables based on stock landings --
    print("Computing tables based on stock landings...")
    # Retrieve stock landings
    stock_landings_fao_areas = pd.read_excel(os.path.join(clean_data_dir, "stock_landings.xlsx"))

    # Merge with stock assessments for Status, Tier, etc.
    primary_key = ["ASFIS Scientific Name", "Location"]
    stock_landings_fao_areas = pd.merge(stock_landings_fao_areas, stock_assessments, on=primary_key)
    
    # Group across analysis group for
    stock_landings = (
        stock_landings_fao_areas.groupby(["Analysis Group", "ASFIS Scientific Name", "Location"])
        .agg({
                "ASFIS Name": "first",
                "ISSCAAP Code": "first",
                "Status": "first",
                "Tier": "first",
                "Stock Landings 2021": "sum"
             }
            )
        .reset_index()
    )
    
    # Group 48,58,88 together for percent coverage
    southern_mask = stock_landings_fao_areas["FAO Area"].isin([48,58,88])
    stock_landings_fao_areas["FAO Area"] = stock_landings_fao_areas["FAO Area"].astype(object)
    stock_landings_fao_areas.loc[southern_mask, "FAO Area"] = "48,58,88"
    
    # Compute percent coverage
    pc = compute_percent_coverage(
        stock_landings_fao_areas,
        species_landings,
        fishstat,
        isscaap_to_remove,
    )

    # Compute percent coverage across tiers
    pc_tiers = compute_percent_coverage_tiers(
        stock_landings_fao_areas,
        species_landings,
        fishstat,
        isscaap_to_remove,
    )
    
    # Compute percent coverage for SOFIA data
    # Retrieve SOFIA landings
    sofia_landings_fao_areas = pd.read_excel(os.path.join(clean_data_dir, "sofia_landings.xlsx"))
    
    pc_sofia = compute_percent_coverage(
        sofia_landings_fao_areas,
        species_landings,
        fishstat,
        isscaap_to_remove,
        landings_key=2021,
    )

    # Compute and save the comparison of percent coverage
    pc_comp = pd.merge(pc_sofia, pc, on="FAO Area", how="right", suffixes=(" Previous", " Updated"))

    pc_comp_fp = os.path.join(output_dir, "percent_coverage_comparison.xlsx")
    pc_comp.to_excel(pc_comp_fp, index=False)
    round_excel_file(pc_comp_fp)

    # Save percent coverage across tiers
    pc_tiers_fp = os.path.join(output_dir, "percent_coverage_tiers.xlsx")
    pc_tiers.to_excel(pc_tiers_fp)
    round_excel_file(pc_tiers_fp)

    # Compute weighted percentages with and w/o Tunas category
    wp_area = compute_weighted_percentages(stock_landings)
    
    # Compute weighted percentages for SOFIA
    # Aggregate landings based on Analysis Group
    agg_dict = {
        "ASFIS Name": "first",
        2021: "sum"
    }
    
    sofia_landings = (
        sofia_landings_fao_areas.groupby(["Analysis Group", "ASFIS Scientific Name", "Status"])
        .agg(agg_dict)
        .reset_index()
    )
    
    sofia_assessed_mask = sofia_landings["Status"].isin(["U", "M", "O"])
    sofia_landings_assessed = sofia_landings[sofia_assessed_mask].copy()

    sofia_landings_assessed = sofia_landings_assessed.rename(
        columns={2021: "Stock Landings 2021"}
    )
    sofia_landings_assessed = sofia_landings_assessed[
        ["Analysis Group", "ASFIS Scientific Name", "Status", "Stock Landings 2021"]
    ]

    wp_sofia = compute_weighted_percentages(sofia_landings_assessed)

    # Compare the weighted percentages for the two assessments
    wp_comp = compare_weighted_percentages(wp_sofia, wp_area)

    # Save the updated weighted percentages w/Tuna area separate
    wp_area_fp = os.path.join(output_dir, "status_by_landings_area.xlsx")
    wp_area.to_excel(wp_area_fp)
    round_excel_file(wp_area_fp)

    # Save the comparison by landings of updated and previous method
    wp_comp_fp = os.path.join(output_dir, "comparison_by_landings.xlsx")
    # Add percent coverage footnote
    wp_comp.to_excel(wp_comp_fp)
    round_excel_file(wp_comp_fp)

    # Compute weighted percentages across tiers
    wp_tiers = compute_weighted_percentages(stock_landings, key="Tier")

    # Save weighted percentages across tiers
    wp_tiers_fp = os.path.join(output_dir, "status_by_landings_tier.xlsx")
    wp_tiers.to_excel(wp_tiers_fp)
    round_excel_file(wp_tiers_fp)

    # Get weighted percentages and total landings
    wp_totl = get_weighted_percentages_and_total_landings(
        wp_area, 
        fishstat, 
        species_landings, 
        special_groups=special_groups,
        isscaap_to_remove=isscaap_to_remove
    )

    # Save wp w/totals w/footnote explaining why assessed landings / total landings * 100
    # per area does not correspond to the percent coverage (doesn't account for addition of Tuna landings, etc.)
    sbl_footnote = (
        f"Note: Area landings exclude landings from ISSCAAP codes {", ".join([str(i) for i in isscaap_to_remove])}, \n"
        + "except for stocks which have been incorporated in assessment."
    )
    wp_totl_fp = os.path.join(output_dir, "status_by_landings_w_totals_area.xlsx")
    add_footnote(wp_totl, sbl_footnote, multi_index=True).to_excel(wp_totl_fp)
    round_excel_file(wp_totl_fp)

    # Compute weighted percentages across tiers per area
    wp_tiers_area = get_weighted_percentages_by_tier_and_area(stock_landings, wp_totl)

    # Save
    wp_tiers_area_fp = os.path.join(output_dir, "status_by_landings_tier_and_area.xlsx")
    wp_tiers_area.to_excel(wp_tiers_area_fp)
    round_excel_file(wp_tiers_area_fp)

    # Save aggregations for areas individually
    for ag in stock_landings["Analysis Group"].unique():
        wp = wp_area.loc[ag].to_frame().T
        
        wpt = compute_weighted_percentages(
            stock_landings[stock_landings["Analysis Group"] == ag], key="Tier"
        )
        area_lbl = f"{ag} Total"
        wpt.index = list(wpt.index[:-1]) + [area_lbl]
        
        cbl = wp_comp.loc[ag].to_frame().T if ag in wp_comp.index else None

        area_summary_fp = os.path.join(
            output_dir, os.path.join("Analysis Group Statistics", f"{ag}_summary.xlsx")
        )
        with pd.ExcelWriter(
            area_summary_fp,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace",
        ) as writer:
            wp.to_excel(writer, sheet_name="Status by Landings (Area)")
            wpt.to_excel(writer, sheet_name="Status by Landings (Tier)")

            if area in wp_comp.index:
                cbl.to_excel(writer, sheet_name="Comparison by Landings")

        round_excel_file(area_summary_fp)

    # Compute weighted percentages for top ten species globally
    wp_top10 = compute_species_weighted_percentages(stock_landings, top_ten_species)

    # Save
    wp_top10_fp = os.path.join(output_dir, "weight_percentages_top10species.xlsx")
    wp_top10.to_excel(wp_top10_fp)
    round_excel_file(wp_top10_fp)
    
    numerical_ags = [ag for ag in stock_assessments["Analysis Group"].unique() if ag not in ["Area 67 - Salmon", "Sharks", "Tuna"]]
    top_species_areas = compute_top_species_by_area(numerical_ags, stock_assessments, stock_landings, fishstat)
    top_species_areas_fp = os.path.join(output_dir, "top10_assessed_species_by_area.xlsx")
    
    with pd.ExcelWriter(top_species_areas_fp) as writer:
        for ag, top_species in top_species_areas.items():
            top_species.to_excel(writer, sheet_name=ag)
        

    print(f"All files saved to {output_dir}")


if __name__ == "__main__":
    main()

