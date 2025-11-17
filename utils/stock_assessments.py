"""
This file includes all functions used for the data cleanup and validation processes
for the stock assessments

These functions are implemented in ./main/stock_assessments.py
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm


def get_asfis_mappings(
    input_dir,
    asfis_file,
    name_changes_file="",
    mapping_file_delimiter=";",
    name_changes_sheet_name="Updated",
):
    """Retrieves ASFIS mappings.

    This function reads ASFIS data and name change data from specified files,
    creates mappings between scientific names, common names, and ISSCAAP codes,
    and returns a dictionary containing these mappings.

    Args:
        input_dir (str): The directory containing the input files.
        asfis_file (str): The name of the ASFIS data file (CSV).
        name_changes_file (str): The name of the name changes file (Excel).
            Defaults to "".
        mapping_file_delimiter (str, optional): The delimiter used in the ASFIS
            file. Defaults to ";".
        name_changes_sheet_name (str, optional): The name of the sheet in the
            name changes file. Defaults to "Updated".

    Returns:
        dict: A dictionary containing the following mappings:
            "ASFIS": ASFIS DataFrame.
            "ASFIS Scientific Name to ASFIS Name": Mapping from scientific to common names.
            "ASFIS Scientific Name to ISSCAAP Code": Mapping from scientific names to ISSCAAP codes.
            "ASFIS Scientific Names": Unique scientific names from ASFIS data.

            And the following mappings if name_changes_file is specified:
                "ASFIS Scientific Name Update": Dictionary of scientific name updates.
                "ASFIS Name Update": Dictionary of common name updates.
    """
    asfis = pd.read_csv(
        os.path.join(input_dir, asfis_file), delimiter=mapping_file_delimiter
    )

    scientific_to_name = dict(zip(asfis["Scientific_Name"], asfis["English_name"]))
    scientific_to_isscaap = dict(
        zip(asfis["Scientific_Name"], asfis["ISSCAAP_Group_Code"])
    )
    scientific_names = asfis["Scientific_Name"].unique()

    mappings = {
        "ASFIS": asfis,
        "ASFIS Scientific Name to ASFIS Name": scientific_to_name,
        "ASFIS Scientific Name to ISSCAAP Code": scientific_to_isscaap,
        "ASFIS Scientific Names": scientific_names,
    }

    if name_changes_file:
        name_changes = pd.read_excel(
            os.path.join(input_dir, name_changes_file),
            sheet_name=name_changes_sheet_name,
        )

        current_df = name_changes[name_changes["Updates"] == "current"]
        old_df = name_changes[name_changes["Updates"] == "old"]

        merged_df = pd.merge(
            old_df, current_df, on="Alpha3_Code", suffixes=("_old", "_current")
        )

        scientific_update = dict(
            zip(merged_df["Scientific_Name_old"], merged_df["Scientific_Name_current"])
        )

        mappings["ASFIS Scientific Name Update"] = scientific_update

    return mappings


def read_stock_data(file_path, sheet_indices, desc="Stock assessment sheets"):
    """Reads stock assessment data from an Excel file into a dictionary of DataFrames.

    This function reads data from specified sheets of an Excel file, selecting
    rows based on provided indices and skipping a specified number of rows.
    It adds "Original Line No." and "Sheet" columns to each DataFrame to create
    a unique identifier for each row.  It also includes a progress bar using `tqdm`.

    Args:
        file_path (str): Path to the Excel file.
        sheet_indices (dict): A dictionary where keys are sheet names and values
            are tuples `(skiprows, start_row, end_row)`. `skiprows` is the number
            of rows to skip before reading data, and `start_row` and `end_row`
            (inclusive) specify the rows to read from that sheet.
        desc (str, optional): Description printed in console when loading sheets.

    Returns:
        dict: A dictionary where keys are sheet names and values are Pandas
            DataFrames containing the read data. Each DataFrame will have added
            "Original Line No." (the original row index) and "Sheet" (the sheet
            name) columns.
    """
    overview = {}

    for sheet, indices in tqdm(sheet_indices.items(), desc=desc):
        skiprows, start, end = indices

        overview[sheet] = pd.read_excel(
            file_path, sheet_name=sheet, skiprows=skiprows
        ).loc[start:end]

        # Combination of Original Line No., Sheet will be primary key of each Dataframe for each area
        overview[sheet] = (
            overview[sheet].reset_index().rename(columns={"index": "Original Line No."})
        )
        overview[sheet]["Original Line No."] += skiprows + 2
        overview[sheet]["Sheet"] = sheet

    return overview


def add_area_column(overview, sheet_to_area):
    """Adds an "Area" column to DataFrames.

    This function takes a dictionary of DataFrames and a mapping from sheet names
    to area values. It adds a new "Area" column to each DataFrame, assigning
    the corresponding area value based on the sheet name.

    Args:
        overview (dict): A dictionary where keys are sheet names and values are
            Pandas DataFrames.
        sheet_to_area (dict): A dictionary mapping sheet names to area values.
            Area values can be strings or integers. If a sheet name is not
            present as a key in this dictionary, the sheet name itself is used
            as the area value.

    Returns:
        dict: A *new* dictionary (not a copy of the input) with the DataFrames
            modified to include the "Area" column.
    """
    new_overview = {}

    for sheet, df in overview.items():
        area = sheet_to_area.get(sheet, sheet)

        new_overview[sheet] = df.copy()
        new_overview[sheet]["Area"] = area

    return new_overview


def fill_col_na(df, col1, col2):
    """Fills missing values in one column with values from another column.

    This function takes a Pandas DataFrame and two column names. It fills any
    missing values (NaN) in the first column (`col1`) with the corresponding
    values from the second column (`col2`).

    Args:
        df (pd.DataFrame): The input Pandas DataFrame.
        col1 (str): The name of the column to fill missing values in.
        col2 (str): The name of the column whose values will be used to fill
            the missing values in `col1`.

    Returns:
        pd.Series: A Pandas Series representing the updated `col1` with filled
            missing values. The original DataFrame `df` is *not* modified in place.
    """
    return df[col1].fillna(df[col2])


def standardize_columns(overview, columns_map):
    """Renames columns in DataFrames within a dictionary based on a mapping.

    This function takes a dictionary of DataFrames and a column name mapping.
    It renames the columns of each DataFrame according to the provided mapping.

    Args:
        overview (dict): A dictionary where keys are sheet names and values are
            Pandas DataFrames.
        col_dict (dict): A dictionary where keys are the *old* column names
            and values are the *new* column names.

    Returns:
        dict: A new dictionary (a copy of the input) with the DataFrames modified
            in place to have renamed columns.
    """
    df_dict = overview.copy()

    for sheet, df in df_dict.items():
        if sheet in columns_map:
            df_dict[sheet] = df.rename(columns=columns_map[sheet])

    return df_dict


def add_column_from_merge(df1, df2, primary_key, col_to_add):
    """Adds a column from one DataFrame to another based on a primary key.

    This function merges two DataFrames based on a shared primary key and adds
    a specified column from the second DataFrame to the first.  A left merge
    is performed, so all rows from the first DataFrame are kept.

    Args:
        df1 (pd.DataFrame): The first DataFrame, to which the new column will be added.
        df2 (pd.DataFrame): The second DataFrame, containing the column to add.
        primary_key (str): The name of the column that serves as the primary key
            for the merge.
        col_to_add (str): The name of the column to add from `df2` to `df1`.

    Returns:
        pd.DataFrame: A new DataFrame that is the result of the merge.
    """
    return df1.merge(
        df2[[primary_key, col_to_add]], on=primary_key, how="left", suffixes=("", "_x")
    )


def ffill_columns(df, columns):
    """Fills missing values in specified columns using forward fill.

    This function fills missing values (NaN) in one or more columns of a DataFrame
    using the forward fill method.  Forward fill propagates the last observed
    non-null value forward to fill subsequent NaNs.

    Args:
        df (pd.DataFrame): The input Pandas DataFrame.
        columns (str or list): A string or a list of strings representing the
            name(s) of the column(s) to fill missing values in.

    Returns:
        pd.DataFrame: The DataFrame with the specified columns modified in place.
        It is also returned for convenience.
    """
    if not isinstance(columns, list):
        columns = [columns]

    for col in columns:
        df[col] = df[col].ffill()

    return df


def use_more_appropriate_scientific_name(
    overview,
    og_col="ASFIS Scientific Name",
    new_col="More appropriate ASFIS Scientific Name",
    flag_col="Status",
    flag="to ignore",
):
    """Updates scientific names in DataFrames based on a more appropriate name column.

    This function iterates through a dictionary of DataFrames and updates the
    scientific names based on a "More Appropriate" column. It also adds a flag
    to the "Status" column if a specific string is found in the "More Appropriate"
    scientific name.

    Args:
        overview (dict): A dictionary where keys are sheet names and values are
            Pandas DataFrames.
        og_col (str, optional): The name of the column containing the original
            scientific names. Defaults to "ASFIS Scientific Name".
        new_col (str, optional): The name of the column containing the more
            appropriate scientific names. Defaults to "More Appropriate ASFIS Scientific Name".
        flag_col (str, optional): The name of the column to store status information.
            Defaults to "Status".
        flag (str, optional): The string to search for in the "More Appropriate"
            scientific name to add to the status. Defaults to "to ignore".

    Returns:
        dict: A *copy* of the input dictionary with the DataFrames modified
            in place.
    """
    df_dict = overview.copy()

    for sheet, df in df_dict.items():
        if new_col in df.columns:
            df_dict[sheet][flag_col] = df.apply(
                lambda row: (
                    row[flag_col] + f" ({flag})"
                    if isinstance(row[new_col], str)
                    and flag in row[new_col].lower()
                    and isinstance(row[flag_col], str)
                    else row[flag_col]
                ),
                axis=1,
            )

            df_dict[sheet][og_col] = df.apply(
                lambda row: (
                    row[new_col]
                    if isinstance(row[new_col], str)
                    and flag not in row[new_col].lower()
                    and not pd.isna(row[new_col])
                    else row[og_col]
                ),
                axis=1,
            )

    return df_dict


def drop_cols(overview, col_names):
    """Drops specified columns from DataFrames within a dictionary.

    This function iterates through a dictionary of DataFrames and drops the
    specified columns from each DataFrame.  It only attempts to drop columns
    that are actually present in the DataFrame.

    Args:
        overview (dict): A dictionary where keys are sheet names and values are
            Pandas DataFrames.
        col_names (list): A list of column names to drop.

    Returns:
        dict: A *copy* of the input dictionary with the specified columns
            dropped from the DataFrames.
    """
    df_dict = overview.copy()

    for sheet, df in df_dict.items():
        cols_to_drop = [col for col in col_names if col in df.columns]

        df_dict[sheet] = df.drop(columns=cols_to_drop)

    return df_dict


def get_common_name(sn, scientific_to_name):
    """Retrieves the common name(s) for a scientific name.

    This function retrieves the common name(s) associated with a given scientific
    name from a provided mapping. If the scientific name contains a comma, it
    is split into multiple scientific names, and the corresponding common names
    are concatenated.

    Args:
        sn (str): The scientific name to look up.
        scientific_to_name (dict): A dictionary mapping scientific names to
            common names.

    Returns:
        str: The common name(s) associated with the scientific name, or an
            empty string if the scientific name is not found.
    """
    if isinstance(sn, float):
        return np.nan

    if ", " not in sn:
        return scientific_to_name.get(sn, np.nan)
    elif sn in scientific_to_name:
        return scientific_to_name[sn]

    sns = sn.split(", ")
    cn = []

    for s in sns:
        if s in scientific_to_name:
            name = scientific_to_name[s]
            if isinstance(name, float) and np.isnan(name):
                cn.append(s)
            else:
                cn.append(name)

    return ", ".join(cn)


def get_isscaap_code(sn, scientific_to_isscaap):
    """Retrieves the ISSCAAP code for a scientific name.

    This function retrieves the ISSCAAP code associated with a given scientific
    name from a provided mapping. If the scientific name contains a comma, it
    is split into multiple scientific names, and the corresponding ISSCAAP code
    are collected. If multiple ISSCAAP codes are found, a warning is printed.

    Args:
        sn (str): The scientific name to look up.
        scientific_to_isscaap (dict): A dictionary mapping scientific names to
            ISSCAAP codes.

    Returns:
        str or None: The ISSCAAP code associated with the scientific name,
            or None if the scientific name is not found. If multiple ISSCAAP
            codes are found for a comma-separated scientific name, the first
            code is returned.
    """
    if isinstance(sn, float):
        return np.nan

    if ", " in sn:
        isscaaps = []
        for s in sn.split(","):
            s = s.strip()
            isscaap = scientific_to_isscaap.get(s)
            if isscaap:
                return isscaaps.append(isscaap)
        if isscaaps and len(set(isscaaps)) > 1:
            print(f"Differing ISSCAAP Codes for species {sn}")
            return np.nan
        elif isscaaps:
            return isscaaps[0]

    return scientific_to_isscaap.get(sn)


def standardize_column_values(overview, cols_w_map={}, key="ASFIS Scientific Name"):
    """
    Standardizes column values in DataFrames within a dictionary based on a mapping.

    Args:
        overview (dict): A dictionary where keys are sheet names and values are DataFrames.
        cols_w_map (dict): A dictionary where keys are column names and values are mapping dictionaries.
        key (str): The column name used as the key for mapping.
        split_comma (bool): If True, splits the key on commas before mapping.

    Returns:
        dict: A dictionary with standardized column values in DataFrames.
    """
    df_dict = overview.copy()

    for sheet, df in df_dict.items():
        for col, standard_map in cols_w_map.items():
            if col == "ASFIS Name":
                df_dict[sheet][col] = (
                    df[key].apply(get_common_name, args=(standard_map,)).fillna(df[col])
                )
            elif col == "ISSCAAP Code":
                df_dict[sheet][col] = (
                    df[key]
                    .apply(get_isscaap_code, args=(standard_map,))
                    .fillna(df[col])
                )
            else:
                df_dict[sheet][col] = df[key].map(lambda x: standard_map.get(x, x))

    return df_dict


def correct_values(overview, correct_values_dict, matching_col="ASFIS Scientific Name"):
    """Corrects specific values in DataFrames based on a nested mapping.

    This function iterates through a dictionary of DataFrames and applies corrections
    to specific values in specified columns. The corrections are defined in a
    nested dictionary.

    Args:
        overview (dict): A dictionary where keys are sheet names and values are
            Pandas DataFrames.
        correct_values_dict (dict): A nested dictionary where the *outer* keys
            are sheet names, the *inner* keys are column names, and the *innermost*
            keys are the values in the `matching_col` column that need to be
            corrected. The *innermost values* are the corrected values.
        matching_col (str, optional): The name of the column used to match the
            values to be corrected. Defaults to "ASFIS Scientific Name".

    Returns:
        dict: A *copy* of the input dictionary with the DataFrames modified
            in place.
    """
    df_dict = overview.copy()

    for sheet, correct_map in correct_values_dict.items():
        for col, updates in correct_map.items():
            for matching_val, new_val in updates.items():
                df_dict[sheet].loc[
                    df_dict[sheet][matching_col] == matching_val, col
                ] = new_val

    return df_dict


def update_values(overview, mapping, update_col="ASFIS Scientific Name", sheets=[]):
    """Updates values in a specified column based on a mapping.

    This function iterates through a dictionary of DataFrames and updates the values
    in a specified column based on a provided mapping.  Values not found in the
    mapping are left unchanged.

    Args:
        overview (dict): A dictionary where keys are sheet names and values are
            Pandas DataFrames.
        mapping (dict): A dictionary where keys are the *old* values and values
            are the *new* values for the value update.
        update_col (str, optional): The name of the column to update.
            Defaults to "ASFIS Scientific Name".

    Returns:
        dict: A *copy* of the input dictionary with the specified column in the
            DataFrames modified in place.
    """
    df_dict = overview.copy()

    for sheet, df in df_dict.items():
        if not sheets or sheet in sheets:
            df_dict[sheet][update_col] = df[update_col].map(lambda x: mapping.get(x, x))

    return df_dict


def remove_values(overview, remove_dict):
    """Removes rows containing specified values in specified columns.

    This function iterates through a dictionary of DataFrames and removes rows
    where the values in specified columns match values provided in a nested
    dictionary.

    Args:
        overview (dict): A dictionary where keys are sheet names and values are
            Pandas DataFrames.
        remove_dict (dict): A nested dictionary where the *outer* keys are sheet
            names, the *inner* keys are column names, and the *innermost* values
            are lists of values to remove from the corresponding column.

    Returns:
        dict: A *copy* of the input dictionary with the DataFrames modified
            in place (rows removed).
    """
    df_dict = overview.copy()

    for sheet, removals in remove_dict.items():
        for col, vals_to_remove in removals.items():
            mask = ~df_dict[sheet][col].isin(vals_to_remove)
            df_dict[sheet] = df_dict[sheet][mask]

    return df_dict


def remove_stocks(overview, stocks_to_remove, output_dir):
    """Removes specific stock entries from DataFrames based on line numbers.

    This function iterates through a dictionary of DataFrames and removes rows
    corresponding to specific stock entries, identified by their "Original Line No.".

    Args:
        overview (dict): A dictionary where keys are sheet names and values are
            Pandas DataFrames.
        stocks_to_remove (dict): A dictionary where keys are sheet names and
            values are lists of "Original Line No." values to remove from the
            corresponding sheet.

    Returns:
        dict: A *copy* of the input dictionary with the specified stock entries
            removed from the DataFrames.
    """
    df_dict = overview.copy()

    stocks_removed = pd.DataFrame()

    for sheet, remove_info in stocks_to_remove.items():
        line_nos = [info[0] for info in remove_info]
        reasons = [info[1] for info in remove_info]

        mask = df_dict[sheet]["Original Line No."].isin(line_nos)

        cols = ["Area", "ASFIS Scientific Name", "Location", "Tier", "Status"]
        sr = df_dict[sheet][mask][cols].copy()
        sr["Sheet in Base File"] = sheet
        sr["Reason Removed"] = reasons
        stocks_removed = pd.concat([stocks_removed, sr])

        df_dict[sheet] = df_dict[sheet][~mask]

    stocks_removed.to_excel(
        os.path.join(output_dir, "stocks_removed.xlsx"), index=False
    )

    return df_dict


def change_locations(overview, location_changes):
    """Changes locations in DataFrames based on specified line numbers.

    This function iterates through a dictionary of DataFrames and updates the
    "Location" column for specific rows, identified by their "Original Line No.".

    Args:
        overview (dict): A dictionary where keys are sheet names and values are
            Pandas DataFrames.
        location_changes (list): A list of tuples, where each tuple contains:
            - The sheet name (str).
            - A list of location change tuples. Each location change tuple contains:
                - The "Original Line No." (int) of the row to modify.
                - The new "Location" value (str).

    Returns:
        dict: A *copy* of the input dictionary with the "Location" column
            updated in the specified rows of the DataFrames.
    """
    df_dict = overview.copy()

    for sheet, locs in location_changes.items():
        for loc_tup in locs:
            idx, loc = loc_tup
            mask = df_dict[sheet]["Original Line No."] == idx
            df_dict[sheet].loc[mask, "Location"] = loc

    return df_dict


def filter_dfs(overview, col_values):
    """Filters DataFrames based on specified column values.

    This function iterates through a dictionary of DataFrames and filters each
    DataFrame to keep only rows where the values in specified columns match
    the provided values.

    Args:
        overview (dict): A dictionary where keys are sheet names and values are
            Pandas DataFrames.
        col_values (dict): A dictionary where keys are column names and values
            are lists of values to keep in the corresponding column.

    Returns:
        dict: A *copy* of the input dictionary with the DataFrames filtered
            according to the specified column values.
    """
    df_dict = overview.copy()

    for sheet, df in df_dict.items():
        for col, vals in col_values.items():
            mask = df[col].isin(vals)
            df_dict[sheet] = df[mask]

    return df_dict


def fix_nan_location(df):
    data = df.copy()

    nan_mask = data["Location"].isna()
    area = df["Area"].values[0]

    def make_location_from_area(area):
        if isinstance(area, str) and area.isdigit() or area == "67 Other Stocks":
            area_loc = f"Area {area}"
        elif area == "48,58,88":
            area_loc = f"Areas {area}"
        else:
            area_loc = area

        return area_loc

    data.loc[nan_mask, "Location"] = data[nan_mask]["Area"].apply(
        make_location_from_area
    )

    return data


def add_back_to_fao_area(loc, special_group, loc_to_area, area_to_name={}):
    areas = loc_to_area[special_group].get(loc)

    if isinstance(areas, list) and len(areas) == 1:
        area = areas[0]

        if isinstance(area, (int, float)):
            return area_to_name.get(area, str(area))
        else:
            print(
                f"Location {loc} in category {special_group} maps to Area {area} of wrong type ({type(area)})"
            )

            return np.nan
    elif isinstance(areas, list) and len(areas) > 1:
        msg = (
            f"Location {loc} in category {special_group} corresponds to multiple FAO Areas. "
            + "Cannot add back to FAO Area unless it maps to a single FAO Area."
        )
        print(msg)

        return np.nan
    else:
        return np.nan


def use_standard_columns(overview, standard_columns):
    """Selects and standardizes columns in DataFrames based on a dictionary of columns.

    This function iterates through a dictionary of DataFrames and selects only
    the columns specified as keys in the `standard_columns` dictionary. If any
    of the specified columns are missing from a DataFrame, a KeyError is raised,
    and a message is printed indicating the missing columns.

    Args:
        overview (dict): A dictionary where keys are sheet names and values are
            Pandas DataFrames.
        standard_columns (dict): A dictionary whose keys are the column names to
            select from each DataFrame. The values of the dictionary are ignored.

    Returns:
        dict: A *copy* of the input dictionary with the DataFrames modified to
            contain only the specified standard columns.

    Raises:
        KeyError: If any of the standard columns are missing from a DataFrame.
    """
    df_dict = overview.copy()

    for sheet, df in df_dict.items():
        try:
            df_dict[sheet] = df[standard_columns]
        except KeyError:
            missing_cols = set(standard_columns) - set(df.columns)
            print(f"Sheet {sheet} is missing column(s) {", ".join(missing_cols)}")
            raise KeyError

    return df_dict


def standardize_dtypes(overview, col_dtypes):
    """Sets the data type of specified columns element-wise in DataFrames.

    This function iterates through a dictionary of DataFrames and attempts to
    cast the values in specified columns to the provided data types. It handles
    cases where multiple data types are provided (as a tuple) and gracefully
    handles casting errors by leaving the original value unchanged.

    Args:
        overview (dict): A dictionary where keys are sheet names and values are
            Pandas DataFrames.
        col_dtypes (dict): A dictionary where keys are column names and values
            are either a single data type (e.g., `int`, `float`, `str`) or a
            tuple of data types to try casting in order.

    Returns:
        dict: A *copy* of the input dictionary with the specified columns
            modified in place (element-wise data type conversion).
    """
    df_dict = overview.copy()

    for sheet, df in df_dict.items():
        for col_name, dtype_or_tuple in col_dtypes.items():
            if col_name in df_dict[sheet].columns:
                if isinstance(dtype_or_tuple, type):
                    try:
                        df_dict[sheet][col_name] = df[col_name].astype(dtype_or_tuple)
                    except ValueError:
                        message = f"Cannot cast column {col_name} in sheet {sheet} to type {dtype_or_tuple.__name__}"
                        raise ValueError(message)
                elif isinstance(dtype_or_tuple, tuple):

                    def try_cast(val, tuple):
                        if isinstance(val, float) and np.isnan(val):
                            return val

                        for dtype in tuple:
                            try:
                                return dtype(val)
                            except (ValueError, TypeError):
                                pass
                        return val

                    df_dict[sheet][col_name] = df[col_name].apply(
                        try_cast, args=(dtype_or_tuple,)
                    )

    return df_dict


def concatenate_data(overview, cols_to_sort=[]):
    """Concatenates DataFrames from a dictionary into a single DataFrame.

    This function takes a dictionary of DataFrames and concatenates them into
    a single DataFrame. Optionally, it sorts the resulting DataFrame by
    specified columns.

    Args:
        overview (dict): A dictionary where keys are sheet names and values are
            Pandas DataFrames.
        cols_to_sort (list, optional): A list of column names to sort the
            concatenated DataFrame by. Defaults to an empty list (no sorting).

    Returns:
        pd.DataFrame: A single DataFrame resulting from the concatenation of
            all DataFrames in the input dictionary.
    """
    overview_df = pd.DataFrame()

    for sheet, df in overview.items():
        overview_df = pd.concat([overview_df, df])

    if cols_to_sort:
        overview_df = overview_df.sort_values(cols_to_sort).reset_index(drop=True)
    else:
        overview_df = overview_df.reset_index(drop=True)

    return overview_df


def assign_fao_area(row, location_to_area):
    area = row["Area"]

    if isinstance(area, str) and area.isdigit():
        return area

    if isinstance(area, str) and "67" in area:
        return "67"

    loc = row["Location"]

    if area == "48,58,88":
        south_area = loc.split(".")[0]
        if south_area.isdigit():
            return south_area

    areas = location_to_area[area].get(loc, "")

    if not isinstance(areas, list):
        return str(areas)
    elif isinstance(areas, list) and len(areas) == 0:
        return ""

    return ", ".join([str(a) for a in areas])


def validate_primary_key(df, primary_key=["ASFIS Scientific Name", "Location"]):
    """Validates the uniqueness and non-null values of a primary key in a DataFrame.

    This function checks if the specified primary key columns in a DataFrame
    contain any null (NaN) values and if the combination of primary key values
    is unique. If any null values or duplicate combinations are found, a
    ValueError is raised.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        primary_key (list or str, optional): A list or string representing the
            column(s) that form the primary key. Defaults to
            ["Area", "ASFIS Scientific Name", "Location"].

    Raises:
        ValueError: If any null values or duplicate primary key combinations
            are found.
    """
    if not isinstance(primary_key, list):
        primary_key = [primary_key]

    for key in primary_key:
        na_mask = df[key].isna()

        if sum(na_mask) > 0:
            nas = list(df[na_mask].index.astype(str))  # Convert indices to strings
            message = f"Column {key} has NaN value(s) at indices {', '.join(nas)}"
            raise ValueError(message)

    duplicate_mask = df[primary_key].duplicated()

    if sum(duplicate_mask) > 0:
        dups = df[duplicate_mask][primary_key].values
        message = f"Non-unique primary key for value(s) {dups}"
        raise ValueError(message)


def validate_consistent_values(
    df, group_key="ASFIS Scientific Name", cols_to_check=["ASFIS Name", "ISSCAAP Code"]
):
    """Validates that specified columns have consistent values within groups.

    This function checks if specified columns have consistent values within groups
    defined by a group key column in a DataFrame. If inconsistent values are
    found, a ValueError is raised.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        group_key (str, optional): The name of the column to group by.
            Defaults to "Alpha3_Code".
        cols_to_check (list, optional): A list of column names to check for
            consistent values within each group. Defaults to
            ["ASFIS Name", "ISSCAAP Code"].

    Raises:
        ValueError: If inconsistent values are found in any of the specified
            columns within any group.
    """
    groups = df.groupby(group_key)

    for col in cols_to_check:
        for sn, group in groups:
            check_vals = set(group[col].values)

            if len(check_vals) > 1 and not all(pd.isna(v) for v in check_vals):
                print(f"{sn} has differing values for {col}: {check_vals}")
