"""
Main file for running the SOSI 2026 ETL Pipeline
"""

import yaml
import logging
import os
from pathlib import Path
from tqdm import tqdm


# Define directories
src_dir = Path(__file__).resolve().parent
project_dir = src_dir.parent
config_dir = project_dir / "config"
logs_dir = project_dir / "logs"

logger = logging.getLogger(__name__)
log_levels = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARN": logging.WARN}
logging.basicConfig(
    filename=logs_dir / "SOSI.log",
    level=log_levels[os.getenv("LOG_LEVEL", "INFO")],
)

breakpoint()

from .authenticate import SOSIAuthenticator
from .extract import SOSIExtractor
from .transform import SOSITransformer
from .plot import SOSIPlotter
from .load import SOSILoader
from .utils import find_key, remove_key, find_table, parse_args_config


def main():
    """ """
    # Configure command line parameters
    with open(config_dir / "args.yaml", "r") as file:
        args_config = yaml.safe_load(file)
    args, all_flag = parse_args_config(args_config)

    # Retrieve configuration settings
    with open(config_dir / "pipeline.yaml", "r") as file:
        config = yaml.safe_load(file)

    # ---------- EXTRACTION ----------
    logger.info("---------- Beginning Extraction ----------")
    authenticator = SOSIAuthenticator(
        config["authentication"]["google"]["service_account"]["cred_path_env_var"],
        config["authentication"]["google"]["oauth"]["cred_path_env_var"],
    )
    extract_drive_service = authenticator.get_google_service(
        service_name="drive",
        scopes=config["authentication"]["google"]["service_account"]["scopes"]["drive"],
    )
    sheets_service = authenticator.get_google_service(
        service_name="sheets",
        scopes=config["authentication"]["google"]["service_account"]["scopes"][
            "sheets"
        ],
    )
    extractor = SOSIExtractor(
        drive_service=extract_drive_service,
        sheets_service=sheets_service,
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
            source_info=source_info,
            source_name=source_name,
            extract_args=args.extract,
            extract_all_flag=all_flag,
        )
        tables["source"] |= extracted_tables

    logger.info("---------- Extraction Complete ----------")
    # ---------- TRANSFORMATION ----------
    logger.info("---------- Beginning Transformation ----------")
    transformer = SOSITransformer(
        assessment_year=config["assessment_year"],
        isscaap_to_exclude=config["transformation"]["isscaap_to_exclude"],
        error_log_dir=logs_dir / "error",
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
        output_args = params["args"]
        function_name = params["function"]
        # TODO: Generalize input_table / join_table for lists/dicts resp.
        input_table = find_table(
            tables=tables,
            args=output_args,
            table_key="input_table",
            keys_to_remove="source",
            pop_from_args=True,
        )
        join_table = (
            find_table(
                tables=tables,
                args=output_args,
                table_key="join_table",
                keys_to_remove="source",
                pop_from_args=True,
            )
            if "join_table" in output_args
            else None
        )
        tables["output"][table_name] = transformer.compute_table(
            input_table=input_table,
            join_table=join_table,
            function_name=function_name,
            args=output_args,
        )

    # Create output plots
    plotter = SOSIPlotter(
        tables=remove_key(tables, "source"),
        assessment_year=config["assessment_year"],
        isscaap_to_exclude=config["transformation"]["isscaap_to_exclude"],
        species_to_exclude=config["plotting"]["species_to_exclude"],
    )
    figures = {}
    for figure_name, params in output_figures_config.items():
        # Skip the variable definition block in the configuration file
        if figure_name == "variables":
            continue
        fig = plotter.create_figure(
            figure_name=figure_name,
            params=params,
            figures_to_show=args.plot,
            show_all_flag=all_flag,
        )
        figures[figure_name] = fig

    # ---------- LOADING ----------
    load_drive_service = authenticator.get_google_service(
        service_name="drive", creds_type="oauth"
    )
    loader = SOSILoader(
        drive_service=load_drive_service,
        folder_id=config["loading"]["folder_id"],
        branch_env_var=config["loading"]["branch_env_var"],
    )

    # Load tables
    for table_type, tbls in tables.items():
        match table_type:
            case "computed" | "output":
                loader.upload_tables(
                    tbls,
                    config["loading"]["tables"]["extension"],
                    table_type,
                    config["version"],
                    config["loading"]["tables"]
                    .get("save_index", True)
                    .get(table_type, True),
                    config["loading"]["tables"].get("replace_on_exists", True),
                    items_to_load=args.load,
                    load_all_flag=all_flag,
                )
                if args.output is not None:
                    loader.save_tables(
                        tables=tbls,
                        extension=config["loading"]["tables"]["extension"],
                        table_type=table_type,
                        output_dir=args.output,
                        save_index=config["loading"]["tables"]
                        .get("save_index", True)
                        .get(table_type, True),
                        replace_on_exists=config["loading"]["tables"].get(
                            "replace_on_exists", True
                        ),
                    )
            case _:
                continue

    # Load figures
    loader.upload_figures(
        figures,
        config["loading"]["figures"]["extension"],
        config["loading"]["figures"]["dpi"],
        config["version"],
        items_to_load=args.load,
        load_all_flag=all_flag,
    )
    if args.output is not None:
        loader.save_figures(
            figures=figures,
            extension=config["loading"]["figures"]["extension"],
            dpi=config["loading"]["figures"]["dpi"],
            output_dir=args.output,
            replace_on_exists=config["loading"]["figures"].get(
                "replace_on_exists", True
            ),
        )


if __name__ == "__main__":
    main()
