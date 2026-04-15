"""
Main file for running the SOSI 2026 ETL Pipeline
"""

import yaml
import logging
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from .authenticate import SOSIAuthenticator
from .extract import SOSIExtractor
from .transform import SOSITransformer
from .plot import SOSIPlotter
from .load import SOSILoader
from .schema import SchemaRules
from .utils import (
    find_key,
    sort_dict,
    is_step_enabled,
    remove_key,
    find_table,
    parse_args_config,
    is_step_enabled,
    find_by_attribute,
)

# Define directories
src = Path(__file__).resolve().parent
root = src.parent
config_dir = root / "config"
logs_dir = root / "logs"

# Import the models for Neon DB upload
from web_app import models

# Load the environment variables
load_dotenv(dotenv_path=".env")

# Configure command line parameters
with open(config_dir / "args.yaml", "r") as file:
    args_config = yaml.safe_load(file)
ARGS, ALL_FLAG = parse_args_config(args_config)

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=logs_dir / "SOSI.log",
    level=getattr(logging, ARGS.verbosity),
)


def main():
    """ """
    # Retrieve configuration settings
    with open(config_dir / "pipeline.yaml", "r") as file:
        config = yaml.safe_load(file)

    # ---------- AUTHENTICATION ----------
    logger.info("---------- Beginning Authentication ----------")
    authenticator = SOSIAuthenticator(
        config["authentication"]["google"]["service_account"]["cred_path_env_var"],
        config["authentication"]["google"]["oauth"]["cred_path_env_var"],
    )
    drive_service_account = authenticator.get_google_service(
        service_name="drive",
        scopes=config["authentication"]["google"]["service_account"]["scopes"]["drive"],
    )
    sheets_service = authenticator.get_google_service(
        service_name="sheets",
        scopes=config["authentication"]["google"]["service_account"]["scopes"][
            "sheets"
        ],
    )
    logger.info("---------- Authentication Complete ----------")

    # ---------- EXTRACTION ----------
    logger.info("---------- Beginning Extraction ----------")
    extractor = SOSIExtractor(
        drive_service=drive_service_account,
        sheets_service=sheets_service,
    )

    # Define keys for data vault, which will hold all tables
    # These describe the types of tables which will be stored
    source = "source"
    info = "info"
    sosi = "sosi"
    output = "output"
    tables = {
        source: {},
        info: {},
        sosi: {},
        output: {},
    }
    # Define standard table names
    stock_r = "stock_reference"
    sta = "stock_assessments"
    spl = "species_landings"
    spl_mod = "species_landings_mod"
    stl = "stock_landings"
    prev = "_prev"

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
            extract_args=ARGS.extract,
            extract_all_flag=ALL_FLAG,
        )
        tables[source] |= extracted_tables

    logger.info("---------- Extraction Complete ----------")
    # ---------- TRANSFORMATION ----------
    logger.info("---------- Beginning Transformation ----------")
    transformer = SOSITransformer(
        editions=config["transformation"]["editions"],
        isscaap_to_exclude=config["transformation"]["isscaap_to_exclude"],
        error_log_dir=logs_dir / "error",
    )

    # Apply schema verification and transformations from config files
    logger.info("Applying schema verification and transformations...")
    schema_configs = config["transformation"]["schema"]
    schema_rules = SchemaRules()
    for table_name, schema_config in schema_configs.items():
        schema_fp = (
            config_dir / "schema" / schema_config.get("file_name", f"{table_name}.yaml")
        )
        with schema_fp.open("r") as file:
            schema = yaml.safe_load(file)

        table = find_key(tables, table_name)
        if table is None:
            raise KeyError(
                f"Table {table_name} has schema file but has not been extracted."
            )

        table_transformed = transformer.apply_schema_and_transform(
            table, schema, schema_rules, table_name
        )
        key = schema_config["key"]

        tn = schema_config.get("rename", table_name)
        tables[key][tn] = table_transformed

    tables[sosi][stock_r] = transformer.create_stock_reference(
        tables[sosi], schema_configs
    )
    tables[sosi][sta] = transformer.clean_stock_assessments(
        tables[sosi][stock_r], tables[info]["asfis"]
    )

    # Set the transformer class variables using the stock assessments table
    transformer.set_class_variables(tables[sosi][sta], tables[info]["asfis"])

    # Add the sosi grouping column to capture & aquaculture tables
    tables[info]["capture"] = transformer.add_sosi_grouping(
        tables[info]["capture"], merge_southern=True
    )
    tables[info]["aquaculture"] = transformer.add_sosi_grouping(
        tables[info]["aquaculture"], merge_southern=True
    )

    logger.info("-> Computing species landings")
    tables[sosi][spl] = transformer.compute_species_landings(
        tables[sosi][sta],
        tables[info]["capture"],
    )
    with open(config_dir / "substitutions.yaml", "r") as file:
        landings_substitutions = yaml.safe_load(file)
    tables[sosi][spl_mod] = transformer.compute_species_landings_mod(
        tables[sosi][spl],
        tables[sosi][sta],
        tables[info]["capture"],
        tables[info]["asfis"],
        landings_substitutions,
    )

    logger.info("-> Computing stock landings")
    tables[sosi][stl] = transformer.compute_stock_landings(
        tables[sosi][sta], tables[sosi][spl_mod]
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
        input_table = find_table(
            tables=tables,
            args=output_args,
            table_key="input_table",
            keys_to_remove=source,
            pop_from_args=True,
        )
        join_table = (
            find_table(
                tables=tables,
                args=output_args,
                table_key="join_table",
                keys_to_remove=source,
                pop_from_args=True,
            )
            if "join_table" in output_args
            else None
        )
        tables[output][table_name] = transformer.compute_table(
            input_table=input_table,
            join_table=join_table,
            function_name=function_name,
            args=output_args,
        )

    # Create output plots
    plotter = SOSIPlotter(
        tables=remove_key(tables, source),
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
            figures_to_show=ARGS.plot,
            show_all_flag=ALL_FLAG,
        )
        figures[figure_name] = fig

    # ---------- LOADING ----------
    drive_service_oauth = authenticator.get_google_service(
        service_name="drive", creds_type="oauth"
    )
    db_engine = (
        authenticator.get_db_engine(config["loading"]["db_env_var"])
        if ARGS.load_db is not None
        else None
    )
    loader = SOSILoader(
        drive_service_oauth=drive_service_oauth,
        drive_service_account=drive_service_account,
        sheets_service=sheets_service,
        drive_folder_id=config["loading"]["folder_id"],
        db_engine=db_engine,
        branch_env_var=config["loading"]["branch_env_var"],
    )

    # Ensure database schema is synced if we are loading tables to the db
    if ARGS.load_db is not None:
        loader.sync_database_schema()

    # Define database loading order for foreign key restraints
    all_tables_to_load_db = find_by_attribute(
        models, "__tablename__", get_attr_name=True
    )

    # Load tables
    for table_type, tbls in tables.items():
        if table_type == source:  # Skip the source upload
            continue

        tables_to_load_db = {
            k: v
            for k, v in tbls.items()
            if is_step_enabled(k, ARGS.load_db, ALL_FLAG) and k in all_tables_to_load_db
        }
        tables_to_load_drive = {
            k: v
            for k, v in tbls.items()
            if is_step_enabled(k, ARGS.load_drive, ALL_FLAG)
        }
        if table_type in [info, sosi]:  # These tables get uploaded to DB
            if tables_to_load_db:
                sorted_tables = sort_dict(tables_to_load_db, all_tables_to_load_db)
                loader.upload_tables_db(sorted_tables)

        if table_type in [sosi, output]:  # These get uploaded to drive
            if tables_to_load_drive:
                loader.upload_tables_drive(
                    tables_to_load_drive,
                    config["loading"]["tables"]["extension"],
                    table_type,
                    config["version"],
                    config["loading"]["tables"]
                    .get("save_index", {})
                    .get(table_type, True),
                    config["loading"]["tables"].get("replace_on_exists", True),
                )

        if ARGS.output is not None:
            loader.save_tables(
                tables=tbls,
                extension=config["loading"]["tables"].get("local_extension", "csv"),
                table_type=table_type,
                output_dir=ARGS.output,
                save_index=config["loading"]["tables"]
                .get("save_index", True)
                .get(table_type, True),
                replace_on_exists=config["loading"]["tables"].get(
                    "replace_on_exists", True
                ),
            )

    # Update stock catch in master sheet
    if ARGS.update_catch:
        tables_info = config["extraction"]["SOSI_2026_workspace"]["tables"]
        loader.update_catch_col(
            tables_info["sosi2026"]["file_name"],
            tables_info["sosi2026"]["sheet_name"],
            tables[sosi][stl],
            config["assessment_year"],
        )
        loader.update_catch_col(
            tables_info["sosi2025"]["file_name"],
            tables_info["sosi2025"]["sheet_name"],
            tables[sosi][stl + prev],
            config["previous_assessment_year"],
        )

    # Load figures
    figures_to_load = {
        k: v
        for k, v in figures.items()
        if is_step_enabled(k, ARGS.load_drive, ALL_FLAG)
    }
    if figures_to_load:
        loader.upload_figures(
            figures_to_load,
            config["loading"]["figures"]["extension"],
            config["loading"]["figures"]["dpi"],
            config["version"],
        )
    if ARGS.output is not None:
        loader.save_figures(
            figures=figures,
            extension=config["loading"]["figures"]["extension"],
            dpi=config["loading"]["figures"]["dpi"],
            output_dir=ARGS.output,
            replace_on_exists=config["loading"]["figures"].get(
                "replace_on_exists", True
            ),
        )


if __name__ == "__main__":
    main()
