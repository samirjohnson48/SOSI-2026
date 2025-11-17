"""
This file includes all functions used for assigned of landings weights
for all assessed stocks

These functions are implemented in ./main/stock_weights.py
"""

import pandas as pd
import numpy as np
import re


def retrieve_31_37_81_weights(file_path, sheet_names=["Area31Jeremy", "37", "81"]):
    """Retrieves weights for areas 31, 37, 81 from the original sheets.
       The weights are taken from the reported landings by consultants.

    Args:
        file_path (str): Path to original assessment file
        sheet_names (list, optional): Names of sheets for areas 31, 37, 81, respectively.
                                      Defaults to ["Area31Jeremy", "37", "81"].
    """
    combined_data = pd.DataFrame()

    area_dict = dict(zip(sheet_names, [31, 37, 81]))

    for sheet in sheet_names:
        data = pd.read_excel(file_path, sheet_name=sheet)

        data["Area"] = area_dict[sheet]

        rename_dict = {
            "Scientific name": "ASFIS Scientific Name",
            "Location (stock)": "Location",
            "Value for SANKEY": "Weight 1",
            "Landings": "Weight 1",
            "Landings (tonnes) Author's Observation / Estimate (2021) - if not in FishStat": "Weight 1",
            "More appropriate ASFIS Scientific Name": "MAASN",
        }

        data = data.rename(columns=rename_dict)

        # Remove stocks where weight is not available
        data = data[data["Weight 1"] != "not available"]

        if "Catches" in data.columns:
            data["Weight 1"] = data["Weight 1"].fillna(data["Catches"])

        # Use More appropriate ASFIS Scientific Name to match cleaned data
        data["ASFIS Scientific Name"] = data[["ASFIS Scientific Name", "MAASN"]].apply(
            lambda row: (
                row["MAASN"]
                if row["MAASN"] != "to ignore"
                else row["ASFIS Scientific Name"]
            ),
            axis=1,
        )

        # Update locations for Area 37
        if sheet == "37":
            data = data.reset_index(names="Original Line No.")
            data["Original Line No."] += 2

            loc_changes = [
                (14, "Algeria 1"),
                (17, "Algeria 2"),
                (19, "Algeria 3"),
                (27, "Levant Sea 1"),
                (28, "Levant Sea 2"),
            ]

            for change in loc_changes:
                mask = data["Original Line No."] == change[0]
                data.loc[mask, "Location"] = change[1]

            data = data.drop(columns="Original Line No.")

        # Need to do specific extraction for area 81
        if sheet == "81":
            data["Weight 1"] = data["Weight 1"].apply(lambda x: 1 if x == "<1" else x)

            df = data[
                data["Weight 1"].apply(
                    lambda x: isinstance(x, str) and "notes" in x.lower()
                )
            ][["ASFIS Scientific Name", "Location", "Weight 1", "Notes"]]

            def find_landings_81(notes):
                if "=" not in notes:
                    return np.nan

                landings = notes.split(" = ")[-1]
                if " &" in landings:
                    landings = landings.split(" & ")
                    if " and " in landings[-1]:
                        landings[-1] = "".join(
                            [char if char.isdigit() else "" for char in landings[-1]]
                        )
                    landings = sum([float(l) if l.isdigit() else 0 for l in landings])
                    return landings

                landings_str = ""
                for char in landings:
                    if char.isdigit():
                        landings_str += char
                    else:
                        break
                landings = float(landings_str)

                if "SPO 3" in notes or "SNA" in notes:
                    factor = 0.5
                else:
                    factor = 1

                return factor * landings

            df["Weight 1"] = df["Notes"].apply(find_landings_81)

            # Convert some landings from number to weight
            # Weight of Chilean flat oyster is about 55 g
            df.loc[
                df["ASFIS Scientific Name"] == "Ostrea chilensis",
                "Weight 1",
            ] = (
                7.6 * 1e6 * (55 / 1e3) / 1e3
            )

            # Notes: The species name is Chrysophrys auratus, but previously
            # referred to as Pagrus auratus. We use the landings from the latter as the weight
            df.loc[df["Location"] == "SNA 1 (East Northland)", "Weight 1"] = 2289.5

            # Update Area 81 weights with the weights taken from the notes
            df = df[["ASFIS Scientific Name", "Location", "Weight 1"]]
            data = pd.merge(
                data,
                df,
                on=["ASFIS Scientific Name", "Location"],
                how="left",
                suffixes=("", " Update"),
            )
            data["Weight 1"] = data["Weight 1 Update"].fillna(data["Weight 1"])
            data = data.drop(columns="Weight 1 Update")

        data = data[["Area", "ASFIS Scientific Name", "Location", "Weight 1"]].dropna(
            subset="ASFIS Scientific Name"
        )

        data["Weight 1"] = data["Weight 1"].fillna(1)

        combined_data = pd.concat([combined_data, data])

    combined_data = combined_data.reset_index(drop=True)

    return combined_data


def extract_alphanumeric(string):
    if not isinstance(string, str):
        return string
    return re.sub(r"[^a-zA-Z0-9]", "", string)


def merge_weights(
    weights, new_weights, primary_key, weight1_na=False, clean_location=False
):
    if clean_location:
        weights["Location Match"] = weights["Location"].apply(extract_alphanumeric)
        new_weights["Location Match"] = new_weights["Location"].apply(
            extract_alphanumeric
        )
        primary_key.remove("Location")
        primary_key.append("Location Match")

    merge = pd.merge(
        weights,
        new_weights,
        on=primary_key,
        how="left",
        suffixes=("", "_new"),
        indicator=True,
    )

    new_cols = [col for col in merge.columns if "_new" in col and col != "Location_new"]
    for new_col in new_cols:
        og_col = new_col.replace("_new", "")
        merge[og_col] = merge[new_col].fillna(merge[og_col])

    if weight1_na:
        merge.loc[merge["_merge"] == "both", "Weight 1"] = np.nan

    cols_to_drop = new_cols + ["_merge"]

    if clean_location:
        cols_to_drop.append("Location Match")
        primary_key.remove("Location Match")
        primary_key.append("Location")
        cols_to_drop.append("Location_new")

    merge = merge.drop(columns=cols_to_drop)

    return merge


def compute_weights(group):
    # See if weight 2 should be used
    if all(
        group["Weight 1"].apply(lambda x: isinstance(x, str) or pd.isna(x) or x == 0)
    ):
        # If no Weight 1 or Weight 2, use uniform weighting for group
        if all(group["Weight 2"].isna()) or all(group["Weight 2"] == 0):
            return pd.Series(1 / len(group), index=group.index)
        else:
            group["Weight 2"] = group["Weight 2"].fillna(1)
            return group["Weight 2"] / group["Weight 2"].sum()

    # Set base value of Weight 1 to 0.001
    # All stocks with species landings should get non-zero catch
    zero_mask = group["Weight 1"] == 0
    group.loc[zero_mask, "Weight 1"] = 1e-3

    # Get all rows with valid entries for Weight 1
    val = group[
        group["Weight 1"].apply(
            lambda x: isinstance(x, (int, float)) and not pd.isna(x)
        )
    ]

    # Rows with no assigned secondary weight get lowest value of 1
    group["Weight 2"] = group["Weight 2"].fillna(1)

    for idx, row in group.iterrows():
        if isinstance(row["Weight 1"], str) or pd.isna(row["Weight 1"]):
            # If priority weight is missing
            if sum(val["Weight 2"] == row["Weight 2"]) > 0:
                # Check if there is a stock which has priority weight with equal secondary weight
                # Use the mean priority weight from all stocks with equal secondary weight
                group.loc[idx, "weight"] = val[val["Weight 2"] == row["Weight 2"]][
                    "Weight 1"
                ].mean()
            else:
                # Find the row(s) which secondary weight closest to this row's secondary weight
                w = min(
                    val["Weight 2"].unique(), key=lambda x: abs(x - row["Weight 2"])
                )
                group.loc[idx, "weight"] = val[val["Weight 2"] == w]["Weight 1"].mean()
        else:
            group.loc[idx, "weight"] = row["Weight 1"]

    return group["weight"] / group["weight"].sum()


def validate_normalization(
    weights,
    group_key=["FAO Area", "ASFIS Scientific Name"],
    weight_key="Normalized Weight",
):
    for group_name, group in weights.groupby(group_key):
        assert np.isclose(
            group[weight_key].sum(), 1
        ), f"Weights for {group_name} are not normalized"

    print(f"All weights are normalized grouped by {", ".join(group_key)}")
