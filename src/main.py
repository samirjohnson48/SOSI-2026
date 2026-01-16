"""
Main file for running the SOSI 2026 ETL Pipeline
"""

import yaml
from pathlib import Path
import logging
from tqdm import tqdm
import argparse


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
        config["authentication"]["google_api"]["cred_env_var"],
        config["authentication"]["google_api"]["scopes"],
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
        extracted_tables = extractor.extract_tables(source_info, args.use_cache)
        tables["source"] |= extracted_tables

    logger.info("---------- Extraction Complete ----------")
    # ---------- TRANSFORMATION ----------
    logger.info("---------- Beginning Transformation ----------")
    transformer = SOSITransformer(config=config, error_log_dir=logs_dir / "error")
    years = range(
        config["transformation"]["years"]["first"],
        config["transformation"]["years"]["last"] + 1,
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
        if table_name in tables["source"]:
            table = tables["source"][table_name]
        elif table_name in tables["cleaned"]:
            table = tables["cleaned"][table_name]
        else:
            raise KeyError(
                "Table {table_name} has schema file but has not been extracted."
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
        tables["computed"]["stock_assessments"], tables["cleaned"]["capture"], years
    )

    logger.info("-> Computing stock landings")
    tables["computed"]["stock_landings"] = transformer.compute_stock_landings(
        tables["computed"]["stock_assessments"], tables["computed"]["species_landings"]
    )

    # ---------- SUMMARY TABLES ----------
    with open(config_dir / "analysis.yaml", "r") as file:
        analysis_config = yaml.safe_load(file)

    for table_name, params in analysis_config.items():
        input_table = params.pop("input_table")
        tables["output"][table_name] = transformer.compute_aggregate_table(
            input_table=tables["computed"][input_table], **params
        )

    breakpoint()


if __name__ == "__main__":
    main()
