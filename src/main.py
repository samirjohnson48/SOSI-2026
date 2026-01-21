"""
Main file for running the SOSI 2026 ETL Pipeline
"""

import yaml
import logging
import argparse
import os
from pathlib import Path
from tqdm import tqdm


# Define directories
src_dir = Path(__file__).resolve().parent
project_dir = src_dir.parent
config_dir = project_dir / "config"
logs_dir = project_dir / "logs"

logger = logging.getLogger(__name__)
logging.basicConfig(filename=logs_dir / "SOSI.log", level=logging.DEBUG)

from .authenticate import SOSIAuthenticator
from .extract import SOSIExtractor
from .transform import SOSITransformer
from .plot import SOSIPlotter
from .load import SOSILoader
from .utils import find_key, remove_key


def main():
    """ """
    # Configure command line parameters
    with open(config_dir / "args.yaml", "r") as file:
        args_dict = yaml.safe_load(file)
    parser = argparse.ArgumentParser(description="SOSI 2026")

    for arg, arg_info in args_dict.items():
        required = arg_info.pop("required", False)
        abbr = arg_info.pop("abbreviation", None)
        arg_name = arg if required else f"--{arg}"
        if abbr:
            parser.add_argument(arg_name, f"-{abbr}", **arg_info)
        else:
            parser.add_argument(arg_name, **arg_info)
    args = parser.parse_args()

    # Retrieve configuration settings
    with open(config_dir / "pipeline.yaml", "r") as file:
        config = yaml.safe_load(file)

    # ---------- EXTRACTION ----------
    logger.info("---------- Beginning Extraction ----------")
    authenticator = SOSIAuthenticator(
        config["authentication"]["google_api"]["cred_path_env_var"],
        config["authentication"]["google_api"]["scopes"],
        config["authentication"]["dropbox_api"]["token_env_var"],
    )
    drive_service = authenticator.get_google_service(service_name="drive")
    sheets_service = authenticator.get_google_service(service_name="sheets")
    extractor = SOSIExtractor(
        drive_service=drive_service,
        sheets_service=sheets_service,
        save_files=args.save_files,
        remove_cache=args.remove_cache,
    )
    tables = {"source": {}, "cleaned": {}, "computed": {}, "output": {}}

    table_pbar = tqdm(
        config["extraction"].items(),
        leave=False,
        colour="green",
        ascii=True,
        unit="source",
    )
    for source_name, source_info in table_pbar:
        table_pbar.set_description(f"Extracting tables from {source_name}")
        logger.info(f"--> Extracting source tables")
        extracted_tables = extractor.extract_tables(
            source_info, source_name, args.use_cache
        )
        tables["source"] |= extracted_tables

    logger.info("---------- Extraction Complete ----------")
    # ---------- TRANSFORMATION ----------
    logger.info("---------- Beginning Transformation ----------")
    transformer = SOSITransformer(
        assessment_year=config["assessment_year"], error_log_dir=logs_dir / "error"
    )

    # Combine stock assessments into one table to create stock reference table
    tables["cleaned"]["stock_reference"] = transformer.create_stock_reference(
        tables["source"], config["extraction"]
    )

    logger.debug(
        f"Stock reference columns: {tables['cleaned']['stock_reference'].columns}"
    )

    # Apply schema verification and transformations from config files
    logger.info("Applying schema verification and transformations...")
    for file_path in tqdm(
        Path(config_dir / "schema").glob("*.yaml"),
        desc="Schema verification and transformations",
        leave=False,
    ):
        table_name = file_path.name.split(".")[0]
        table = find_key(tables, table_name)
        if table is None:
            raise KeyError(
                f"Table {table_name} has schema file but has not been extracted."
            )

        logger.info(f"-> {table_name} schema process")

        with file_path.open("r") as file:
            schema = yaml.safe_load(file)

        table_transformed = transformer.apply_schema_and_transform(
            table, schema, table_name
        )
        primary_key = schema["primary_key"]
        transformer.check_primary_key(table_transformed, primary_key, table_name)

        tables["cleaned"][table_name] = table_transformed

    # Create stock assessment table from stock reference table based on status values
    tables["computed"]["stock_assessments"] = transformer.create_stock_assessments(
        tables["cleaned"]["stock_reference"]
    )

    logger.info("-> Computing species landings")
    tables["computed"]["species_landings"] = transformer.compute_species_landings(
        tables["computed"]["stock_assessments"],
        tables["cleaned"]["capture"],
    )

    logger.info("-> Computing stock landings")
    tables["computed"]["stock_landings"] = transformer.compute_stock_landings(
        tables["computed"]["stock_assessments"], tables["computed"]["species_landings"]
    )

    # ---------- SUMMARY TABLES AND PLOTS ----------
    analysis_config_dir = config_dir / "analysis"
    with open(analysis_config_dir / "tables.yaml", "r") as file:
        output_tables_config = yaml.safe_load(file)
    with open(analysis_config_dir / "figures.yaml", "r") as file:
        output_figures_config = yaml.safe_load(file)

    # Create output tables
    for table_name, params in output_tables_config.items():
        input_table = find_key(remove_key(tables, "source"), params.pop("input_table"))
        join_table = (
            find_key(remove_key(tables, "source"), params.pop("join_table"))
            if "join_table" in params
            else None
        )
        tables["output"][table_name] = transformer.compute_aggregate_table(
            input_table=input_table, join_table=join_table, **params
        )

    # Create output plots
    plotter = SOSIPlotter(
        tables=remove_key(tables, "source"),
        assessment_year=config["assessment_year"],
        isscaap_to_exclude=config["plotting"]["isscaap_to_exclude"],
        species_to_exclude=config["plotting"]["species_to_exclude"],
        show_figure=config["plotting"]["show_figure"],
    )
    figures = {}
    for figure_name, params in output_figures_config.items():
        # Skip the variable definition block in the configuration file
        if figure_name == "variables":
            continue
        fig = plotter.create_figure(params)
        figures[figure_name] = fig

    # ---------- LOADING ----------
    print(os.getenv)
    dbx = authenticator.get_dropbox_client()
    loader = SOSILoader(
        dbx,
        table_extension=config["loading"]["table_extension"],
        figure_extension=config["loading"]["figure_extension"],
    )

    # Load tables
    for table_type, tbls in tables.items():
        match table_type:
            case "computed" | "output":
                loader.upload_tables(tbls, table_type, config["version"])
            case _:
                continue

    # Load figures
    loader.upload_figures(
        figures,
        config["version"],
        config["loading"]["dpi"],
        config["loading"]["save_index"],
    )


if __name__ == "__main__":
    main()
