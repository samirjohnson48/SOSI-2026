"""

"""

import os
import pandas as pd
from tqdm import tqdm

from utils.stock_weights import *
from utils.species_landings import explode_stocks


def main():
    # Define directories for input and output files
    parent_dir = os.getcwd()
    input_dir = os.path.join(parent_dir, "input")
    output_dir = os.path.join(parent_dir, os.path.join("output", "clean_data"))

    # Retrieve list of assessed stocks
    weights = pd.read_excel(os.path.join(output_dir, "stock_assessments.xlsx"))

    # Retrieve list of assign weights for stocks
    weights_input = pd.read_excel(os.path.join(input_dir, "weights_input.xlsx"))

    # Merge
    primary_key = ["ASFIS Scientific Name", "Location"]
    weights = pd.merge(weights, weights_input, on=primary_key, how="left")

    # Explode the weights across the FAO Major Fishing Area
    weights = explode_stocks(weights)

    # Progress bar
    tqdm.pandas()

    weights["Normalized Weight"] = (
        weights.groupby(["FAO Area", "ASFIS Scientific Name"])[["Weight 1", "Weight 2"]]
        .progress_apply(compute_weights)
        .reset_index(level=[0, 1], drop=True)
    )

    # Validate weight normalization
    validate_normalization(weights, group_key=["FAO Area", "ASFIS Scientific Name"])

    cols_to_save = ["FAO Area"] + primary_key + ["Normalized Weight"]
    weights = weights[cols_to_save]

    # Save assigned weights to output file
    file_path = os.path.join(output_dir, "stock_weights.xlsx")
    print(f"Saving stocks with weights to {file_path}")
    weights.to_excel(file_path, index=False)


if __name__ == "__main__":
    main()
