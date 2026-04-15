"""
Transform class used transform SOSI 2026 base data
for analysis
"""

import pandas as pd
import numpy as np
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Literal
from pandas.api.typing import FrozenList

logger = logging.getLogger(__file__)

from .utils import join_tables, broadcast_df, make_series_unique, sort_df, order_columns
from .schema import SchemaRules


class SOSITransformer:
    # Define standard column names to be used in transformations
    AREA_LABEL_COL = "fao_area_label"
    AREA_COL = "fao_area"
    AREAS_COL = "fao_areas"
    COMMON_NAME_COL = "common_name"
    COUNTRY_CODE_COL = "country_un_code"
    COUNTRY_NAME_COL = "name"
    EDITION_COL = "sosi_edition"
    GROUP_COL = "sosi_grouping"
    ID_COL = "uid"
    ISSCAAP_COL = "isscaap_code"
    LANDINGS_COL = "landings"
    LOCATION_COL = "location"
    PRODUCTION_COL = "production"
    RECORD_COL = "sosi_record_type"
    SPECIES_CODE_COL = "asfis_code"
    SPECIES_CODES_COL = "asfis_codes"
    SPECIES_NAME_COL = "species_name"
    SPECIES_NAMES_COL = "species_names"
    STATUS_COL = "status"
    STOCK_ID_COL = "stock_id"
    TAXCODE_COL = "taxonomic_code"
    TAXCODES_COL = "taxonomic_codes"
    TIER_COL = "tier"
    WEIGHT_COL = "weight"
    YEAR_COL = "year"

    # Length of taxonomic code for entries which may represent classes
    CLASS_TAXCODE_LEN = 15
    # Default arguments for expanding stock assessments
    DEFAULT_SPECIES_EXPLODE = {
        "cols_to_extend": [SPECIES_CODES_COL, SPECIES_NAMES_COL, TAXCODES_COL],
        "new_col_names": {
            SPECIES_CODES_COL: SPECIES_CODE_COL,
            SPECIES_NAMES_COL: SPECIES_NAME_COL,
            TAXCODES_COL: TAXCODE_COL,
        },
        "new_col_dtypes": {
            c: "string" for c in [SPECIES_CODES_COL, SPECIES_NAMES_COL, TAXCODES_COL]
        },
    }
    DEFAULT_AREA_EXPLODE = {
        "cols_to_extend": AREAS_COL,
        "new_col_names": {AREAS_COL: AREA_COL},
        "new_col_dtypes": {AREAS_COL: "Int64"},
    }
    # Indicator of whether table has global stocks
    FAO_GLOBAL_INDICATOR = "fao:global"
    # Number of 'X' in taxonomic_code for each hierachy level
    HIERARCHY_LEVELS = {"genus": 2, "family": 5, "order": 8, "class": 10}
    # Map the ISSCAAP code divisions to their respective class codes
    ISSCAAP_FALLBACKS = {
        1: "FRF",  # Freshwater fishes NEI
        2: "MZZ",  # Marine fishes NEI
        3: "MZZ",  # Marine fishes NEI
        4: "CRU",  # Marine crustaceans NEI
        5: "MOL",  # Marine molluscs NEI
        8: "MSH",  # Marine shells NEI
        9: "SWX",  # Seaweeds NEI
    }
    # Create a SOSI Namespace for uid creation
    SOSI_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "sosi.fao.org")
    # Southern areas which are grouped together for analysis
    SOUTHERN_AREAS = [48, 58, 88]
    # Delimiter for species codes in stock assessment tables
    SPECIES_CODE_DELIMITER = ";"
    # Valid status for creating stock assessment table
    STATUS_VALS = ["U", "M", "O"]
    # Source name in pipeline.yaml for the stock assessment tables
    STOCK_TABLE_SOURCE = "SOSI_2026_workspace"
    # First year of recorded landings
    FIRST_YEAR = 1950
    # Maps scaling values to unit representation
    SCALING_UNIT_MAP = {1: "", 1e-3: "K", 1e-6: "M"}
    # The label of the global total row in aggregate tables
    TOTALS_LABEL = "Global"
    # Value for sosi_record_type which indicates valid stock assessment
    VALID_RECORD = "SoSIndex"

    def __init__(
        self,
        editions: dict[int, int],
        isscaap_to_exclude: list[int],
        error_log_dir: Path | None = None,
    ):
        self.editions = editions
        self.current_edition = max(editions.keys())
        self.ass_year = max(editions.values())
        self.isscaap_to_exclude = isscaap_to_exclude
        self.error_log_dir = error_log_dir

        # Class variables to be set later
        self.global_species = None
        self.fao_areas = None
        self.non_asfis_species = None

        self.stats = {"rows_dropped": 0}

        logger.info("SOSITransformer initialized.")

    def check_primary_key(
        self, table: pd.DataFrame, primary_key: list[str] | str, table_name: str = ""
    ):
        if isinstance(primary_key, str):
            primary_key = [primary_key]
        checks = {
            "duplicate": lambda t, pk: t[pk].duplicated(keep=False),
            "NA": lambda t, pk: t[pk].isna().any(axis=1),
        }
        for check_name, check_f in checks.items():
            mask = check_f(table, primary_key)
            fails = mask.sum() > 0

            if fails:
                fail_rows = table[mask].sort_values(primary_key)
                logger.error(
                    f"Table {table_name} does not have valid primary key {primary_key} due to check {check_name}"
                )
                logger.error("Failing rows:")
                logger.error(fail_rows[primary_key])
                if self.error_log_dir is not None:
                    if not os.path.exists(self.error_log_dir):
                        os.makedirs(self.error_log_dir)
                    fail_rows.to_csv(self.error_log_dir / f"{table_name}_pk_fail.csv")
                # TODO: Change this back to assertion error once the duplicate has been removed
                # raise AssertionError(
                #     f"Invalid primary key {primary_key} for table {table_name} based on {check_name}"
                # )
                print(
                    f"Invalid primary key {primary_key} for table {table_name} based on {check_name}"
                )
                if check_name == "duplicate":
                    print(
                        f"Keeping first duplicate value for primary key {primary_key}"
                    )
                    table.drop_duplicates(
                        subset=primary_key, inplace=True, ignore_index=True
                    )
            else:
                logger.debug(
                    f"Table {table_name} primary key {primary_key} passes check {check_name}"
                )

    def _apply_transformations(
        self,
        table: pd.DataFrame,
        transformations: dict[str, str],
        transform_from_map: dict[str, dict[str, str]],
        sr: SchemaRules,
        table_name: str = "",
    ) -> pd.DataFrame:
        """
        Applies a set of transformations on the columns of a table
        """
        result = table.copy()

        # First apply transformations from certain columns
        # i.e. if a new column needs to be added based on a transformation of another column
        for col, transformation_info in transform_from_map.items():
            if isinstance(transformation_info, str):
                # transformation_info is really just the name of the new column
                old_col = transformation_info
                result[col] = table[old_col]
                continue
            old_col = transformation_info["column"]
            transformation_name = transformation_info["transformation"]
            try:
                transformation = getattr(sr, transformation_name)
                if col in table.columns:
                    is_na = table[col].isna()
                    result.loc[is_na, col] = transformation(table.loc[is_na, old_col])
                else:
                    result[col] = transformation(table[old_col])
            except SyntaxError:
                raise SyntaxError(
                    f"Incorrect syntax for transformation {transformation} applying to table {table_name} in column {old_col}"
                )

            except Exception as e:
                raise Exception(
                    f"An unexpected error occurred when applying transformation {transformation_name} to table {table_name} on column {col}: {e}"
                )
        # Now apply the transformation on the whole column
        # whether or not it was created from another column
        for col, trs in transformations.items():
            if not isinstance(trs, list):
                trs = [trs]
            for tn in trs:
                try:
                    t = getattr(sr, tn)
                    result[col] = t(result[col])
                except SyntaxError:
                    raise SyntaxError(
                        f"Incorrect syntax for transformation '{t}' applying to table '{table_name}' in column '{col}'"
                    )
                except AttributeError:
                    raise AttributeError(
                        f"Unknown function '{tn}' passed to transform column '{col}' in table '{table_name}'."
                    )
                except Exception as e:
                    raise Exception(
                        f"An unexpected error occurred when applying transformation {tn} to table {table_name} on column {col}: {e}"
                    )

        return result

    def _reduce_bool_mask(self, mask: pd.Series, rule: str) -> pd.Series:
        if rule == "any":
            return mask.groupby(level=0).any()
        elif rule == "all":
            return mask.groupby(level=0).all()
        raise ValueError(f"Invalid rule {rule} given. Must be 'any' or 'all'")

    def _apply_restrictions(
        self,
        table: pd.DataFrame,
        restrictions: dict[str, list[str]],
        sr: SchemaRules,
        table_name: str = "",
    ) -> pd.DataFrame:
        """
        Applies the set of restrictions placed on a column in the schema config file.
        Restrictions are provided as a list of tuples (rule, col) for each column.
        rule: a string representation of a Python lambda function acting on a pd.Series objects
        code: "error" means error on fails, "filter" means filter out rows on fail
        """
        result_table = table.copy()
        for col, rests in restrictions.items():
            for rest in rests:
                rule_name, code = rest
                rule = getattr(sr, rule_name)
                passes_mask = rule(result_table[col])
                if len(passes_mask) > len(result_table):
                    passes_mask = self._reduce_bool_mask(passes_mask, "all")
                if code == "error":
                    assert (~passes_mask).sum() == 0, (
                        f"Table {table_name}, column {col} does not satisfy restriction: {rest}."
                        + "\n"
                        + "Failing rows: \n"
                        + f"{result_table.loc[~passes_mask]}"
                    )
                elif code == "filter":
                    rows_dropped = (~passes_mask).sum()
                    logger.info(
                        f"Dropped {rows_dropped} from table: {table_name} based on restriction: {rule} for column: {col}"
                    )
                    logger.debug("Failing rows: \n", result_table.loc[~passes_mask])
                    if rows_dropped > 0:
                        logger.debug(result_table.loc[~passes_mask])
                        self.stats["rows_dropped"] += rows_dropped
                        result_table = result_table.loc[passes_mask]
                else:
                    raise ValueError(
                        f"Incorrect code: {code} passed to restriction on column: {col} in table: {table_name}"
                    )

        return result_table.reset_index(drop=True)

    def _parse_schema(self, schema: dict[str, Any]) -> tuple:
        """
        Parses schema dictionary and returns tuple of objects
        representing the various schema checks and transformations
        """
        cols = list(schema["columns"].keys())

        transformation_map = {}
        transform_from_map = {}
        dtype_map = {}
        restrictions_map = {}
        rename_map = {}

        for col, info in schema["columns"].items():
            if "transformation" in info:
                transformation_map[col] = info["transformation"]
            if "from" in info:
                transform_from_map[col] = info["from"]
            if "dtype" in info:
                dtype_map[col] = info["dtype"]
            if "restrictions" in info:
                restrictions_map[col] = info["restrictions"]
            if "rename" in info:
                rename_map[col] = info["rename"]

        return (
            cols,
            transformation_map,
            transform_from_map,
            dtype_map,
            restrictions_map,
            rename_map,
        )

    def _create_id_col(
        self,
        df: pd.DataFrame,
        pk: list[str] | str | None = None,
        method: Literal["concat", "uid"] = "uid",
        base_col: str | None = None,
        mask_col: str | None = None,
        mask_transformation: str | None = None,
        sr: SchemaRules | None = None,
    ) -> pd.Series:
        if pk is None:  # We use the index to create the UID's
            assert method == "uid"  # Cannot use concat method without primary key
            id_col = pd.Series(
                [str(uuid.uuid5(self.SOSI_NAMESPACE, f"index_{i}")) for i in df.index]
            )
        else:  # Use concat, or use primary key to create the UID's
            if not isinstance(pk, list):
                pk = [pk]
            rows = zip(*[df[col].fillna("NA").astype(str) for col in pk])
            pk_concat = ["_".join(row) for row in rows]
            id_col: pd.Series
            match method:
                case "concat":
                    id_col = pd.Series(pk_concat, index=df.index)
                case "uid":
                    id_col = pd.Series(
                        [
                            str(uuid.uuid5(self.SOSI_NAMESPACE, val))
                            for val in pk_concat
                        ],
                        index=df.index,
                    )
        if base_col is not None:
            mask = pd.Series(True, index=df.index)
            if mask_col is not None:
                if mask_transformation is not None:
                    assert sr is not None
                    transformation = getattr(sr, mask_transformation)
                    mask = transformation(df[mask_col])
                else:
                    mask = df[mask_col].astype(bool)
            mask |= df[base_col].isna() | df[base_col].duplicated()
            id_col = df[base_col].where(~mask, id_col)

        return id_col

    def apply_schema_and_transform(
        self, table: pd.DataFrame, schema: dict, sr: SchemaRules, table_name: str = ""
    ) -> pd.DataFrame:
        """
        Applies schema definition (selection, transformations, data type casting, and renaming) to a table
        """
        table_cleaned = table.fillna(pd.NA)
        value_set_columns = schema.get("value_set_columns")
        if value_set_columns:
            match value_set_columns:
                case list() | str():
                    table_cleaned = table_cleaned.dropna(subset=value_set_columns)
                case dict():
                    for col, info in value_set_columns.items():
                        value = info.get("value")
                        if value is not None:
                            mask = table_cleaned[col].eq(value)
                            table_cleaned = table_cleaned[mask]
                        else:
                            table_cleaned = table_cleaned.dropna(subset=col)

        (
            cols,
            transformation_map,
            transform_from_map,
            dtype_map,
            restrictions_map,
            rename_map,
        ) = self._parse_schema(schema)

        table_transformed = self._apply_transformations(
            table_cleaned, transformation_map, transform_from_map, sr, table_name
        )
        table_typed = table_transformed.astype(dtype_map)
        table_restricted = self._apply_restrictions(
            table_typed, restrictions_map, sr, table_name
        )
        table_reduced = table_restricted[cols]
        table_renamed = table_reduced.rename(columns=rename_map)

        if "create_id" in schema:
            id_args = schema["create_id"]
            id_name = id_args.get("id_name", self.ID_COL)
            mask_args = id_args.get("mask", {})
            table_renamed[id_name] = self._create_id_col(
                table_renamed,
                pk=schema.get("primary_key"),
                method=id_args.get("method", "uid"),
                base_col=id_args.get("base"),
                mask_col=mask_args.get("column"),
                mask_transformation=mask_args.get("transformation"),
                sr=sr,
            )

        return table_renamed

    def create_stock_reference(
        self,
        sosi_tables: dict[str, pd.DataFrame],
        schema_configs: dict[str, dict[str, str]],
    ) -> pd.DataFrame:
        stock_sheets = [
            sosi_tables[tn]
            for tn, schema_config in schema_configs.items()
            if schema_config.get("stock_sheet", False)
        ]
        stock_reference = pd.concat(stock_sheets)
        # Make the uid unique across all editions
        stock_reference[self.ID_COL] = (
            stock_reference[self.STOCK_ID_COL]
            + "_"
            + stock_reference[self.EDITION_COL].astype(str)
        )
        return stock_reference

    def _explode_columns(
        self,
        table: pd.DataFrame,
        cols_to_extend: str | list[str],
        split_delimiter: str | None = None,
        new_col_names: dict[str, str] | None = None,
        new_col_dtypes: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        table_ext = table.copy()
        cols = [cols_to_extend] if isinstance(cols_to_extend, str) else cols_to_extend

        if split_delimiter is not None:
            for col in cols:
                table_ext[col] = table_ext[col].astype(str).str.split(split_delimiter)

        table_ext = table_ext.explode(cols)

        if new_col_dtypes is not None:
            table_ext = table_ext.astype(new_col_dtypes)

        if new_col_names is not None:
            table_ext = table_ext.rename(columns=new_col_names)

        return table_ext

    def _set_global_species(
        self,
        stock_assessments: pd.DataFrame,
        group_col: str = GROUP_COL,
        special_groups: list[str] | None = None,
    ) -> None:
        if special_groups is None:
            area_mask = stock_assessments[group_col].str.lower().str.contains("area")
            special_groups = list(stock_assessments[~area_mask][group_col].unique())

        sg_mask = stock_assessments[group_col].isin(special_groups)
        global_species = self._explode_columns(
            stock_assessments[sg_mask],
            cols_to_extend=self.AREAS_COL,
            new_col_names={self.AREAS_COL: self.AREA_COL},
            new_col_dtypes={self.AREAS_COL: "Int64"},
        )
        global_species = self._explode_columns(
            global_species,
            cols_to_extend=self.SPECIES_CODES_COL,
            new_col_names={self.SPECIES_CODES_COL: self.SPECIES_CODE_COL},
            new_col_dtypes={self.SPECIES_CODES_COL: "string"},
        )

        self.global_species = global_species[
            [group_col, self.SPECIES_CODE_COL, self.AREA_COL]
        ].drop_duplicates()

    def _set_fao_areas(
        self,
        stock_assessments: pd.DataFrame,
    ) -> None:
        self.fao_areas = [
            int(a)
            for a in stock_assessments[self.AREAS_COL].dropna().explode().unique()
        ]

    def _set_non_asfis_species(
        self, stock_assessments: pd.DataFrame, asfis: pd.DataFrame
    ) -> None:
        sta_ext = self._expand_stock_assessments(stock_assessments)
        non_asfis_mask = ~sta_ext[self.SPECIES_CODE_COL].isin(
            asfis[self.SPECIES_CODE_COL]
        )

        non_asfis_species = sta_ext.loc[
            non_asfis_mask, [self.SPECIES_CODE_COL, self.TAXCODE_COL, self.ISSCAAP_COL]
        ]
        if self.non_asfis_species is None:
            self.non_asfis_species = non_asfis_species
        elif isinstance(self.non_asfis_species, pd.DataFrame):
            self.non_asfis_species = pd.concat(
                [self.non_asfis_species, non_asfis_species]
            ).drop_duplicates()

    def set_class_variables(
        self,
        stock_assessments: pd.DataFrame,
        asfis: pd.DataFrame,
    ) -> None:
        self._set_global_species(stock_assessments)
        self._set_fao_areas(stock_assessments)
        self._set_non_asfis_species(stock_assessments, asfis)

    def _filter_stock_assessments(
        self,
        stock_assessments: pd.DataFrame,
        status_col: str = STATUS_COL,
        status_vals: list[str] = STATUS_VALS,
    ) -> pd.DataFrame:
        status_mask = stock_assessments[status_col].isin(status_vals)
        record_mask = stock_assessments[self.RECORD_COL].eq(self.VALID_RECORD)
        return stock_assessments[status_mask & record_mask].drop(
            columns=self.RECORD_COL
        )

    def _expand_stock_assessments(
        self,
        stock_assessments: pd.DataFrame,
        species_kwargs: dict[str, Any] | None = None,
        area_kwargs: dict[str, Any] | None = None,
        expand_species: bool = True,
        expand_area: bool = True,
    ) -> pd.DataFrame:
        """
        Expands stock assessments by exploding species and area columns.
        """
        result = stock_assessments.copy()
        sp_kwargs = species_kwargs or self.DEFAULT_SPECIES_EXPLODE
        a_kwargs = area_kwargs or self.DEFAULT_AREA_EXPLODE

        if expand_species:
            result = self._explode_columns(result, **sp_kwargs)
        if expand_area:
            result = self._explode_columns(result, **a_kwargs)

        return result

    def create_stock_assessments(
        self,
        source_tables: dict[str, pd.DataFrame],
        extraction_config: dict[str, Any],
        stock_table_source: str = STOCK_TABLE_SOURCE,
        reset_index: bool = True,
    ) -> pd.DataFrame:
        stock_assessments = pd.concat(
            [
                source_tables[table_name]
                for table_name in extraction_config[stock_table_source]["tables"].keys()
            ]
        )

        if reset_index:
            stock_assessments = stock_assessments.reset_index(drop=True)

        return stock_assessments

    def _create_sudo_taxonomic_codes(
        self,
        no_taxcodes: pd.DataFrame,
        asfis: pd.DataFrame,
        species_col: str = SPECIES_NAME_COL,
    ) -> pd.Series:
        no_taxcodes["genus"] = no_taxcodes[species_col].fillna("").str.split(" ").str[0]

        asfis_lookup = asfis.assign(
            genus_match=asfis[self.SPECIES_NAME_COL].str.split().str[0],
            tax_prefix=asfis[self.TAXCODE_COL].str.slice(stop=-2) + "__",
        )
        genus_modes = asfis_lookup.groupby("genus_match")["tax_prefix"].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else None
        )

        sudo_taxcodes = no_taxcodes["genus"].map(genus_modes)
        return sudo_taxcodes.fillna("_")

    def clean_stock_assessments(
        self,
        stock_assessments: pd.DataFrame,
        asfis: pd.DataFrame,
    ) -> pd.DataFrame:
        suf = "_sta"
        sta_filtered = self._filter_stock_assessments(stock_assessments)

        explode_species_kwargs = {
            "cols_to_extend": [
                self.SPECIES_CODES_COL,
                self.SPECIES_NAMES_COL,
            ],
            "new_col_names": {
                self.SPECIES_CODES_COL: self.SPECIES_CODE_COL,
                self.SPECIES_NAMES_COL: self.SPECIES_NAME_COL,
            },
            "new_col_dtypes": {
                self.SPECIES_CODES_COL: "string",
                self.SPECIES_NAMES_COL: "string",
            },
        }
        sta_ext = self._expand_stock_assessments(
            sta_filtered,
            species_kwargs=explode_species_kwargs,
        )

        asfis_cols = [
            self.SPECIES_CODE_COL,
            self.SPECIES_NAME_COL,
            self.COMMON_NAME_COL,
            self.ISSCAAP_COL,
            self.TAXCODE_COL,
        ]
        asfis_mod = asfis.copy()
        asfis_mod[self.SPECIES_NAME_COL] = make_series_unique(
            asfis[self.SPECIES_NAME_COL]
        )
        sta = pd.merge(
            sta_ext,
            asfis_mod[asfis_cols],
            on=self.SPECIES_CODE_COL,
            how="left",
            suffixes=(suf, ""),
        )

        no_taxcode_mask = sta[self.TAXCODE_COL].isna()
        if no_taxcode_mask.any():
            sudo_taxcodes = self._create_sudo_taxonomic_codes(
                sta.loc[
                    no_taxcode_mask,
                    [self.ISSCAAP_COL + suf, self.SPECIES_NAME_COL + suf],
                ],
                asfis,
                species_col=self.SPECIES_NAME_COL + suf,
            )
            sta.loc[no_taxcode_mask, self.TAXCODE_COL] = sudo_taxcodes

        suf_cols = [c for c in sta.columns if c.endswith(suf)]
        for col_suf in suf_cols:
            target_col = col_suf.replace(suf, "")
            if target_col in sta.columns:
                sta[target_col] = sta[target_col].fillna(sta[col_suf])
        sta.drop(columns=suf_cols, inplace=True)

        join_str = lambda x: ";".join(x.dropna().astype(str).unique())
        unique_list = lambda x: list(x.unique())

        aggregations = {
            self.SPECIES_CODE_COL: unique_list,
            self.AREA_COL: unique_list,
            self.SPECIES_NAME_COL: unique_list,
            self.COMMON_NAME_COL: join_str,
            self.TAXCODE_COL: unique_list,
        }

        other_cols = [
            c
            for c in sta.columns
            if c
            not in set(aggregations)
            | {self.ID_COL, self.AREAS_COL, self.SPECIES_CODES_COL}
        ]
        for col in other_cols:
            aggregations[col] = "first"

        cleaned_sta = (
            sta.groupby(self.ID_COL)
            .agg(aggregations)
            .rename(
                columns={
                    self.SPECIES_CODE_COL: self.SPECIES_CODES_COL,
                    self.AREA_COL: self.AREAS_COL,
                    self.SPECIES_NAME_COL: self.SPECIES_NAMES_COL,
                    self.TAXCODE_COL: self.TAXCODES_COL,
                }
            )
            .reset_index()
        )

        return cleaned_sta

    def _pivot_production(
        self,
        table: pd.DataFrame,
        area_col: str = AREA_COL,
        species_col: str = SPECIES_CODE_COL,
        year_col: str = YEAR_COL,
        production_col: str = PRODUCTION_COL,
        years: list[int] | range | None = None,
    ) -> pd.DataFrame:
        if years is None:
            years = range(self.FIRST_YEAR, self.ass_year + 1)
        year_mask = table[year_col].isin(years)
        cap_masked = table[year_mask]
        cap_grouped = cap_masked.groupby(by=[area_col, species_col, year_col])[
            production_col
        ].sum()
        cap_reset = cap_grouped.reset_index()
        cap_pivot = cap_reset.pivot(columns=year_col, index=[area_col, species_col])
        cap_pivot.columns = [col[1] for col in cap_pivot.columns]
        cap = cap_pivot.reset_index()
        return cap

    def compute_species_landings(
        self,
        stock_assessments: pd.DataFrame,
        capture: pd.DataFrame,
        production_col: str = PRODUCTION_COL,
        species_col: str = SPECIES_CODE_COL,
        area_col: str = AREA_COL,
        year_col: str = YEAR_COL,
    ) -> pd.DataFrame:
        """
        Computes the landings for all species in the assessment
        """
        sta_ext = self._expand_stock_assessments(stock_assessments)
        sta_ext = sta_ext.drop_duplicates(subset=[area_col, species_col])[
            [area_col, species_col]
        ]

        cap = (
            capture.groupby([species_col, area_col, year_col])[production_col]
            .sum()
            .reset_index()
        )

        years = list(range(self.FIRST_YEAR, self.ass_year + 1))
        unique_pairs = sta_ext[[species_col, area_col]].drop_duplicates()
        skeleton = (
            unique_pairs.assign(key=1)
            .merge(pd.DataFrame({year_col: years, "key": 1}), on="key")
            .drop("key", axis=1)
        )

        species_landings = pd.merge(
            skeleton,
            cap[[species_col, area_col, year_col, production_col]],
            how="left",
            on=[species_col, area_col, year_col],
        )

        species_landings[self.ID_COL] = self._create_id_col(
            species_landings,
            [species_col, area_col, year_col],
            method="uid",
        )

        return species_landings

    def _get_level_info(
        self,
        asfis: pd.DataFrame,
        level_name: str,
        level_n: int,
        add_count: bool = False,
    ) -> pd.DataFrame:
        asf = asfis.copy()

        if (
            level_name == "class"
        ):  # Use ISSCAAP code instead of taxonomic code for finding class
            code_col = f"{self.SPECIES_CODE_COL}_{level_name}"
            group_col = code_col
            asf[code_col] = (
                (asf[self.ISSCAAP_COL].fillna(0) / 10)
                .astype(int)
                .map(self.ISSCAAP_FALLBACKS)
            )
        else:
            tax_col = f"{level_name}_{self.TAXCODE_COL}"
            group_col = tax_col
            asf[tax_col] = asf[self.TAXCODE_COL].str[:(-level_n)] + "X" * level_n

            # If the species is already at a higher taxonomy, don't assign the tax code
            is_higher = asf[self.TAXCODE_COL].str.contains("X" * level_n)
            asf.loc[is_higher, tax_col] = pd.NA

            merge_cols = [self.TAXCODE_COL, self.SPECIES_CODE_COL]
            asf = pd.merge(
                asf,
                asfis[merge_cols],
                left_on=tax_col,
                right_on=self.TAXCODE_COL,
                how="left",
                suffixes=("", f"_{level_name}"),
            ).drop(columns=f"{self.TAXCODE_COL}_{level_name}")

        if add_count:
            asf[f"{level_name}_count"] = asf.groupby(group_col)[group_col].transform(
                "count"
            )

        return asf

    def _supplement_species_landings(
        self,
        species_landings: pd.DataFrame,
        capture: pd.DataFrame,
        substitutions: dict[str, dict[int, list[str]]],
    ) -> tuple[pd.DataFrame, list[str]]:
        spl = species_landings.copy()
        cap = capture.query(f"{self.YEAR_COL} == {self.ass_year}").drop(
            columns=self.YEAR_COL
        )

        sub_keys = []

        for sub_sp, info in substitutions.items():
            for area, species in info.items():
                assert ~(
                    spl[self.SPECIES_CODE_COL].eq(sub_sp) & spl[self.AREA_COL].eq(area)
                ).any(), (
                    f"Species {sub_sp} in area {area} is already present in the assessment. Cannot supplment these landings for other species."
                )
                added = False
                sub_l = cap.query(
                    f"{self.SPECIES_CODE_COL} == '{sub_sp}' and {self.AREA_COL} == {area}"
                )[self.PRODUCTION_COL].sum()
                n = len(species)
                for sp in species:
                    mask = (
                        spl[self.SPECIES_CODE_COL].eq(sp)
                        & spl[self.AREA_COL].eq(area)
                        & spl[self.YEAR_COL].eq(self.ass_year)
                    )
                    if sum(mask) > 0:
                        spl.loc[mask, self.PRODUCTION_COL] += sub_l / n
                        added = True
                if added:
                    sub_keys.append(sub_sp + "_" + str(area))

        return spl, sub_keys

    def _compute_species_landings_remainder(
        self,
        species_landings: pd.DataFrame,
        asfis: pd.DataFrame,
        capture: pd.DataFrame,
        assessment_years: list[int],
        sub_keys: list[str] = [],
    ) -> pd.DataFrame:
        year_mask = species_landings[self.YEAR_COL].isin(assessment_years)
        na_mask = species_landings[self.PRODUCTION_COL].isna()
        zero_mask = species_landings[self.PRODUCTION_COL].eq(0)
        group_cols = [self.SPECIES_CODE_COL, self.AREA_COL, self.YEAR_COL]

        no_l_mask = year_mask & (na_mask | zero_mask)
        no_l = species_landings[no_l_mask][group_cols]
        has_l = species_landings[~no_l_mask]

        species_info = asfis.copy()
        # Add species which do not show up in ASFIS with their sudo taxonomic code
        if self.non_asfis_species is not None and not self.non_asfis_species.empty:
            species_info = pd.concat(
                [species_info, self.non_asfis_species]
            ).drop_duplicates(subset=[self.SPECIES_CODE_COL])

        for name, n in self.HIERARCHY_LEVELS.items():
            species_info = self._get_level_info(species_info, name, n, add_count=True)

        no_l = pd.merge(no_l, species_info, on=self.SPECIES_CODE_COL, how="left")
        no_l[self.PRODUCTION_COL] = 0.0

        cap = (
            capture[capture[self.YEAR_COL].isin(assessment_years)]
            .groupby(group_cols)[self.PRODUCTION_COL]
            .sum()
            .reset_index()
        )
        aux_landings = no_l.copy()
        reported_keys = list(
            (
                has_l[self.SPECIES_CODE_COL].astype(str)
                + "_"
                + has_l[self.AREA_COL].astype(str)
                + "_"
                + has_l[self.YEAR_COL].astype(str)
            ).unique()
        )
        if sub_keys:
            reported_keys += sub_keys
        for name, n in self.HIERARCHY_LEVELS.items():
            level_code_col = f"{self.SPECIES_CODE_COL}_{name}"
            aux_landings = (
                pd.merge(
                    aux_landings,
                    cap,
                    left_on=[level_code_col, self.AREA_COL, self.YEAR_COL],
                    right_on=group_cols,
                    how="left",
                    suffixes=("", "_cap"),
                )
                .drop(columns=f"{self.SPECIES_CODE_COL}_cap")
                .rename(
                    columns={
                        f"{self.PRODUCTION_COL}_cap": f"{self.PRODUCTION_COL}_{name}"
                    }
                )
            )

            current_keys = (
                aux_landings[level_code_col].astype(str)
                + "_"
                + aux_landings[self.AREA_COL].astype(str)
                + "_"
                + aux_landings[self.YEAR_COL].astype(str)
            )
            already_reported = current_keys.isin(reported_keys)
            still_no_l = aux_landings[self.PRODUCTION_COL].eq(0)
            mask = ~already_reported & still_no_l

            level_l = aux_landings.loc[mask, f"{self.PRODUCTION_COL}_{name}"].fillna(0)
            level_c = aux_landings.loc[mask, f"{name}_count"].fillna(1)
            aux_landings.loc[mask, self.PRODUCTION_COL] += level_l / level_c

        aux_landings[self.ID_COL] = self._create_id_col(
            aux_landings,
            [self.SPECIES_CODE_COL, self.AREA_COL, self.YEAR_COL],
            method="uid",
        )
        cols_to_keep = group_cols + [self.PRODUCTION_COL, self.ID_COL]

        return pd.concat([has_l, aux_landings[cols_to_keep]])

    def compute_species_landings_mod(
        self,
        species_landings: pd.DataFrame,
        stock_assessments: pd.DataFrame,
        capture: pd.DataFrame,
        asfis: pd.DataFrame,
        substitutions: dict[str, dict[int, list[str]]],
        assessment_years: list[int] | None = None,
    ) -> pd.DataFrame:
        assessment_years = assessment_years or list(self.editions.values())
        year_mask = species_landings[self.YEAR_COL].isin(assessment_years)
        spl = species_landings[year_mask]
        spl_supp, sub_keys = self._supplement_species_landings(
            spl, capture, substitutions
        )
        spl_mod = self._compute_species_landings_remainder(
            spl_supp, asfis, capture, assessment_years, sub_keys=sub_keys
        )

        # Only keep the species which are reported for a given year
        sta_ext = self._expand_stock_assessments(stock_assessments)
        sta_ext[self.YEAR_COL] = sta_ext[self.EDITION_COL].map(self.editions)
        group_cols = [self.SPECIES_CODE_COL, self.AREA_COL, self.YEAR_COL]
        sta_ext = sta_ext[group_cols].drop_duplicates()
        spl_mod = pd.merge(spl_mod, sta_ext, on=group_cols)
        return spl_mod

    def _compute_stock_weights(
        self, sta_ext: pd.DataFrame, base_weights: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        group_cols = [self.AREA_COL, self.SPECIES_CODE_COL, self.EDITION_COL]
        uniform_weights = (
            sta_ext.groupby(by=group_cols)[self.ID_COL]
            .value_counts(normalize=True)
            .reset_index()
        ).rename(columns={"proportion": self.WEIGHT_COL})

        uniform_weights[self.YEAR_COL] = uniform_weights[self.EDITION_COL].map(
            self.editions
        )

        if base_weights is None:
            return uniform_weights

        assert self.ID_COL in base_weights.columns, (
            f"{self.ID_COL} must be present in the base weights."
        )
        assert self.WEIGHT_COL in base_weights.columns, (
            f"{self.WEIGHT_COL} must be present in the base weights."
        )

        stock_weights = pd.merge(
            uniform_weights,
            base_weights[[self.ID_COL, self.SPECIES_CODE_COL, self.WEIGHT_COL]],
            on=[self.ID_COL, self.SPECIES_CODE_COL],
            suffixes=("_uniform", ""),
            how="left",
        )
        stock_weights[self.WEIGHT_COL] = stock_weights[self.WEIGHT_COL].fillna(
            stock_weights[f"{self.WEIGHT_COL}_uniform"]
        )

        # assert all(
        #     abs(stock_weights.groupby(group_cols)[self.ID_COL].transform("sum") - 1)
        #     < 1e-3
        # ), "Stock weights are not normalized."

        return stock_weights

    def compute_stock_landings(
        self,
        stock_assessments: pd.DataFrame,
        species_landings_mod: pd.DataFrame,
        base_weights: pd.DataFrame | None = None,
        expanded: bool = False,
    ) -> pd.DataFrame:
        sta_ext = self._expand_stock_assessments(stock_assessments)
        stock_weights = self._compute_stock_weights(sta_ext, base_weights)

        stock_landings = pd.merge(
            stock_weights,
            species_landings_mod,
            on=[self.AREA_COL, self.SPECIES_CODE_COL, self.YEAR_COL],
            how="left",
            suffixes=("", "_spl"),
        )
        stock_landings[self.LANDINGS_COL] = (
            stock_landings[self.PRODUCTION_COL] * stock_landings[self.WEIGHT_COL]
        )

        if expanded:
            return stock_landings

        return (
            stock_landings.groupby(self.ID_COL)[self.LANDINGS_COL].sum().reset_index()
        )

    def _parse_area_label(self, area_label: str) -> int:
        area = area_label.split(":")[-1]

        if area == self.FAO_GLOBAL_INDICATOR:
            return -1
        try:
            return int(area)
        except ValueError:
            raise ValueError(f"Unknown area {area} in area label {area_label}")

    def add_sosi_grouping(
        self, capture: pd.DataFrame, merge_southern: bool = False
    ) -> pd.DataFrame:
        assert isinstance(self.global_species, pd.DataFrame)

        cap = pd.merge(
            capture,
            self.global_species,
            how="left",
            on=[self.SPECIES_CODE_COL, self.AREA_COL],
        )

        if merge_southern:
            southern_mask = cap[self.AREA_COL].isin(self.SOUTHERN_AREAS)
            cap.loc[southern_mask, self.GROUP_COL] = "Area" + "_".join(
                [str(a) for a in self.SOUTHERN_AREAS]
            )

        default_ag = "Area" + cap[self.AREA_COL].astype(str)
        cap[self.GROUP_COL] = cap[self.GROUP_COL].fillna(default_ag)

        return cap

    def _compute_total_landings(
        self,
        capture: pd.DataFrame,
        group_col: str,
        group_val: Any | None = None,
        isscaap_to_exclude: list[int] | None = None,
        years: int | list[int] | None = None,
        area_col: str = AREA_COL,
        isscaap_col: str = ISSCAAP_COL,
        production_col: str = PRODUCTION_COL,
        year_col: str = YEAR_COL,
        scale: float = 1,
    ) -> pd.DataFrame | pd.Series | float:
        assert self.fao_areas is not None
        mask = capture[area_col].isin(self.fao_areas)

        if years is not None:
            years_list = [years] if isinstance(years, int) else years
            mask &= capture[year_col].isin(years_list)

        if isscaap_to_exclude:
            mask &= ~capture[isscaap_col].isin(isscaap_to_exclude)

        if group_val is not None:
            mask &= capture[group_col] == group_val

        filtered_df = capture[mask]

        if isinstance(years, (list, range)) or (years is None and group_val is None):
            return (
                filtered_df.groupby([group_col, year_col])[production_col]
                .sum()
                .mul(scale)
                .unstack(level=year_col)
            )

        group_by_cols = [
            col
            for col, val in [(group_col, group_val), (year_col, years)]
            if val is None
        ]

        if group_by_cols:
            return filtered_df.groupby(group_by_cols)[production_col].sum() * scale

        return filtered_df[production_col].sum() * scale

    def _compute_percent_coverage(
        self,
        species_landings_mod,
        capture: pd.DataFrame,
        asfis: pd.DataFrame,
        sosi_edition: int | list[int] | None = None,
        group_col: str = AREA_COL,
        fao_areas: pd.DataFrame | None = None,
        isscaap_to_exclude: list[int] = [],
        scale: float | str = 1e-6,
        totals_row: bool = True,
        totals_label: str = TOTALS_LABEL,
    ) -> pd.DataFrame:
        if isinstance(scale, str):
            try:
                scale = eval(scale)
            except SyntaxError as e:
                raise SyntaxError(
                    f"Could not process scale as a float: {scale}. \n Message: {e}"
                )
        assert isinstance(scale, float)
        sosi_edition = sosi_edition or self.current_edition
        if not isinstance(sosi_edition, list):
            sosi_edition = [sosi_edition]

        assessment_years = [self.editions[e] for e in sosi_edition]
        year_mask = species_landings_mod[self.YEAR_COL].isin(assessment_years)
        spl = species_landings_mod[year_mask]

        cap = pd.merge(capture, asfis, on=self.SPECIES_CODE_COL)

        if fao_areas is not None:
            cap = pd.merge(cap, fao_areas, on=self.AREA_COL)
            spl = pd.merge(spl, fao_areas, on=self.AREA_COL)

        coverage = (
            spl.groupby([group_col, self.YEAR_COL])[self.PRODUCTION_COL]
            .sum()
            .mul(scale)
            .unstack(level=self.YEAR_COL)
        )

        total_landings = self._compute_total_landings(
            cap,
            group_col=group_col,
            isscaap_to_exclude=isscaap_to_exclude,
            scale=scale,
            years=assessment_years,
        )
        assert isinstance(total_landings, pd.DataFrame)

        unit_label = f"({self.SCALING_UNIT_MAP[scale]}T)"
        cov_label, tl_label = f"coverage {unit_label}", f"total landings {unit_label}"
        coverage.columns = pd.MultiIndex.from_product([coverage.columns, [cov_label]])
        total_landings.columns = pd.MultiIndex.from_product(
            [total_landings.columns, [tl_label]]
        )
        percent_coverage = pd.merge(
            coverage,
            total_landings,
            how="left",
            left_index=True,
            right_index=True,
        )

        if totals_row:
            percent_coverage.loc[totals_label] = percent_coverage.sum()

        for year in assessment_years:
            pc_col = (year, "coverage (%)")
            cov_col, tl_col = (year, cov_label), (year, tl_label)
            percent_coverage[pc_col] = (
                percent_coverage[cov_col] / percent_coverage[tl_col] * 100
            )

        return percent_coverage

    def _compute_percent_coverage_rfmo(
        self,
        capture: pd.DataFrame,
        asfis: pd.DataFrame,
        managed_isscaap: list[int],
        fao_area_managed_map: dict[int, float],
        isscaap_to_exclude: list[int],
        scale: float | str = 1e-6,
        assessment_year: int | None = None,
    ) -> pd.DataFrame:
        scale = float(scale) if isinstance(scale, str) else scale
        assessment_year = assessment_year or self.ass_year

        valid_asfis_mask = ~asfis[self.ISSCAAP_COL].isin(isscaap_to_exclude)
        valid_asfis = asfis[valid_asfis_mask][[self.SPECIES_CODE_COL, self.ISSCAAP_COL]]

        managed_areas = list(fao_area_managed_map.keys())
        managed_cap_mask = capture[self.AREA_COL].isin(managed_areas)
        year_mask = capture[self.YEAR_COL].eq(assessment_year)
        cap = capture[managed_cap_mask & year_mask]
        cap = pd.merge(cap, valid_asfis, on=self.SPECIES_CODE_COL)
        cap["is_managed"] = cap[self.ISSCAAP_COL].isin(managed_isscaap)

        stats = (
            cap.groupby([self.AREA_COL, "is_managed"])[self.PRODUCTION_COL]
            .sum()
            .unstack(fill_value=0)
        ) * scale

        landings_label = f"Landings ({self.SCALING_UNIT_MAP[scale]}T)"

        cols = pd.MultiIndex.from_tuples(
            [
                (landings_label, "total"),
                (landings_label, "coverage"),
                ("Percentage", "coverage"),
            ]
        )

        result = pd.DataFrame(
            data={
                cols[0]: stats[True] + stats[False],
                cols[1]: stats[True] * pd.Series(fao_area_managed_map),
                cols[2]: 0.0,
            },
            index=stats.index,
        )

        total_row = result.sum()
        result.loc["total"] = total_row

        result[cols[2]] = (result[cols[1]] / result[cols[0]]).fillna(0) * 100

        return result

    def _compute_counts_by_weight(
        self,
        input_table: pd.DataFrame,
        value: str,
        by: str | list[str],
        weight_col: str,
        weight_map: dict | None = None,
        top_level: str | None = None,
    ) -> pd.DataFrame:
        df = input_table.copy()
        if weight_map is not None:
            df[weight_col] = df[weight_col].map(weight_map)

        if isinstance(by, str):
            by = [by]

        if top_level is not None:
            by.append(top_level)

        counts = df.groupby(by)[[value, weight_col]].value_counts().reset_index()
        w = "weighted_counts"
        counts[w] = counts[weight_col] * counts["count"]

        group_cols = [*by, value]
        level = [value]
        if top_level is not None:
            level.insert(0, top_level)

        return counts.groupby(group_cols)[w].sum().unstack(level=level)

    def _compute_aggregate_table(
        self,
        input_table: pd.DataFrame,
        value: str,
        by: str | list[str],
        join_table: pd.DataFrame | dict[str, pd.DataFrame] | None = None,
        join_key: list[str] | str | dict[str, str | list[str]] | None = None,
        filter_query: str | None = None,
        top_level: str | None = None,
        show_counts: bool = True,
        show_percentages: bool = True,
        show_counts_total: bool = True,
        value_map: dict[str, str] | None = None,
        totals_row: bool = True,
        totals_label: str = "Total",
        weight_col: str | None = None,
        weight_map: dict | None = None,
        counts_scale: float | str | None = None,
        counts_name: str = "count",
        value_order: list[Any] | None = None,
        dropna_crosstab: bool = False,
    ) -> pd.DataFrame:
        if join_table is not None and join_key is not None:
            data = join_tables(
                input_table, join_table, join_key, suffixes=("", "_joined")
            )
        else:
            data = input_table.copy()

        if filter_query is not None:
            data = data.query(filter_query)

        col_vars = [top_level, value] if top_level else [value]
        counts = pd.crosstab(
            index=data[by] if isinstance(by, str) else [data[col] for col in by],
            columns=[data[v] for v in col_vars],
            values=data[weight_col] if weight_col else None,
            aggfunc="sum" if weight_col else None,
            dropna=dropna_crosstab,
        )

        if totals_row:
            counts.loc[totals_label] = counts.sum()

        metrics = {}
        if show_counts:
            count_df = counts.copy()
            if show_counts_total:
                if top_level:
                    group_totals = count_df.T.groupby(level=0).sum().T
                    group_totals.columns = pd.MultiIndex.from_product(
                        [group_totals.columns, ["Total"]]
                    )
                    count_df = pd.concat([count_df, group_totals], axis=1)
                else:
                    count_df.insert(0, "Total", count_df.sum(axis=1))

            if counts_scale:
                count_df *= float(counts_scale)
            metrics[counts_name] = count_df

        if show_percentages:
            if top_level:
                group_row_totals = counts.T.groupby(level=0).sum().T
                metrics["percentage"] = (
                    counts.div(group_row_totals, level=0, axis=1) * 100
                )
            else:
                row_totals = counts.sum(axis=1)
                metrics["percentage"] = counts.div(row_totals, axis=0) * 100

        result = pd.concat(metrics, axis=1).fillna(0)

        if value_map:
            group_vals = [
                result.columns.get_level_values(i)
                for i in range(result.columns.nlevels - 1)
            ]
            group_vals.append(result.columns.get_level_values(-1).map(value_map))
            mapped_result = result.T.groupby(group_vals).sum().T
            result = pd.concat([result, mapped_result], axis=1)

        if top_level:
            result = result.reorder_levels([1, 0, 2], axis=1)

        return order_columns(result, value_order=value_order)

    def _compute_total_production_wide(
        self,
        production: pd.DataFrame,
        asfis: pd.DataFrame,
        isscaap_to_exclude: list[int] | None,
        index_names: FrozenList,
        scale: float,
        name: str,
    ) -> pd.DataFrame:
        if isscaap_to_exclude is None:
            isscaap_to_exclude = self.isscaap_to_exclude

        prod = pd.merge(production, asfis, on=self.SPECIES_CODE_COL, how="left")
        total_prod = self._compute_total_landings(
            capture=prod,
            group_col=self.GROUP_COL,
            isscaap_to_exclude=isscaap_to_exclude,
            scale=scale,
        )
        assert isinstance(total_prod, pd.DataFrame)

        total_prod.index = pd.MultiIndex.from_product(
            [total_prod.index, [pd.NA], [pd.NA], [name], [pd.NA]],
            names=index_names,
        )
        total_prod.columns = pd.MultiIndex.from_product(
            [
                [self.YEAR_COL.title()],
                total_prod.columns,
            ]
        )

        return total_prod

    def _compute_top_production_countries(
        self,
        capture: pd.DataFrame,
        countries: pd.DataFrame | None = None,
        n: int = 5,
        assessment_year: int | None = None,
        hide_single_countries: bool = False,
    ) -> pd.Series:
        assessment_year = assessment_year or self.ass_year

        cap = capture.query(f"{self.YEAR_COL} == {assessment_year}")
        group_cols = [self.GROUP_COL, self.SPECIES_CODE_COL]
        country_cap = (
            cap.groupby(group_cols + [self.COUNTRY_CODE_COL])[self.PRODUCTION_COL]
            .sum()
            .reset_index()
            .sort_values(
                by=group_cols + [self.PRODUCTION_COL],
                ascending=[True, True, False],
            )
        )
        ordered_countries = country_cap.groupby(group_cols).cumcount()

        if countries is not None:
            country_col = self.COUNTRY_NAME_COL
            name_map = countries.set_index(self.COUNTRY_CODE_COL)[country_col]
            country_cap[country_col] = country_cap[self.COUNTRY_CODE_COL].map(name_map)
        else:
            country_col = self.COUNTRY_CODE_COL

        top_countries_mask = ordered_countries < n
        valid_prod_mask = country_cap[self.PRODUCTION_COL] > 0

        top_countries = (
            country_cap[top_countries_mask & valid_prod_mask]
            .groupby(group_cols)[country_col]
            .apply(lambda x: ", ".join(x))
        )

        if hide_single_countries:
            mask = top_countries.str.len() > 1
            top_countries = top_countries[mask]

        return top_countries

    def _compute_appendix_landings(
        self,
        stock_assessments: pd.DataFrame,
        species_landings: pd.DataFrame,
        capture: pd.DataFrame,
        aquaculture: pd.DataFrame,
        asfis: pd.DataFrame,
        countries: pd.DataFrame,
        sosi_edition: int | None = None,
        isscaap_to_exclude: list[int] | None = None,
        scale: str | float = 1e-3,
        n_top_countries: int = 5,
    ) -> dict[str, pd.DataFrame]:
        sosi_edition = sosi_edition or self.current_edition
        if isinstance(scale, str):
            scale = float(scale)

        sta = stock_assessments[stock_assessments[self.EDITION_COL].eq(sosi_edition)]
        sta_ext = self._expand_stock_assessments(sta)

        cols_to_keep = [
            self.GROUP_COL,
            self.AREA_COL,
            self.SPECIES_CODE_COL,
            self.STATUS_COL,
            self.TIER_COL,
        ]
        stl = join_tables(
            sta_ext[cols_to_keep],
            join_table={
                "species_landings": species_landings,
                "asfis": asfis,
            },
            join_key={
                "species_landings": [self.SPECIES_CODE_COL, self.AREA_COL],
                "asfis": self.SPECIES_CODE_COL,
            },
            how="left",
        )

        index = [
            self.GROUP_COL,
            self.ISSCAAP_COL,
            self.SPECIES_CODE_COL,
            self.COMMON_NAME_COL,
            self.SPECIES_NAME_COL,
        ]
        year_pivot = (
            stl.drop_duplicates(
                subset=[self.GROUP_COL, self.SPECIES_CODE_COL, self.YEAR_COL]
            ).pivot_table(
                index=index,
                columns=self.YEAR_COL,
                values=self.PRODUCTION_COL,
                aggfunc="sum",
            )
            * scale
        )
        year_pivot.columns = pd.MultiIndex.from_product([["Year"], year_pivot.columns])
        totals = year_pivot.groupby(self.GROUP_COL).sum()
        totals.index = pd.MultiIndex.from_product(
            [
                totals.index,
                [np.nan],
                [np.nan],
                ["Total selected species groups"],
                [np.nan],
            ],
            names=year_pivot.index.names,
        )
        year_pivot_w_totals = pd.concat(
            [year_pivot.reset_index(), totals.reset_index()], ignore_index=True
        )
        year_pivot_w_totals = year_pivot_w_totals.set_index(index).sort_index()

        status_pivot = stl.query(f"{self.YEAR_COL} == {self.ass_year}").pivot_table(
            index=index,
            columns=[self.TIER_COL, self.STATUS_COL],
            aggfunc="size",
            fill_value=0,
        )
        status_pivot.columns.names = [self.TIER_COL.title(), self.STATUS_COL.title()]
        assert isinstance(status_pivot.columns, pd.MultiIndex)
        status_pivot.columns = status_pivot.columns.set_levels(
            [
                self.TIER_COL.title() + " " + str(l)
                for l in status_pivot.columns.levels[0]
            ],
            level=0,
        )

        spl_app = pd.concat([year_pivot_w_totals, status_pivot], axis=1)

        top_countries = self._compute_top_production_countries(
            capture, countries=countries, n=n_top_countries, hide_single_countries=True
        )
        top_countries_name = ("", f"Most Active Countries in {self.ass_year}")
        top_countries.name = top_countries_name
        cols = [top_countries_name] + list(spl_app.columns)
        spl_app = spl_app.join(
            top_countries, on=[self.GROUP_COL, self.SPECIES_CODE_COL]
        )[cols]

        total_cap = self._compute_total_production_wide(
            capture,
            asfis,
            isscaap_to_exclude=isscaap_to_exclude,
            scale=scale,
            index_names=spl_app.index.names,
            name="Total marine capture",
        )

        diff_cap = broadcast_df(
            total_cap,
            totals,
            level=self.GROUP_COL,
            operation="subtract",
            set_level_vals={self.COMMON_NAME_COL: "Total other species groups"},
        )

        total_aqua = self._compute_total_production_wide(
            aquaculture,
            asfis,
            isscaap_to_exclude=isscaap_to_exclude,
            scale=scale,
            index_names=spl_app.index.names,
            name="Total aquaculture",
        )

        total_prod = broadcast_df(
            total_cap,
            total_aqua,
            level=self.GROUP_COL,
            operation="add",
            set_level_vals={self.COMMON_NAME_COL: "Total production"},
        )

        result = pd.concat(
            [
                spl_app.reset_index(),
                diff_cap.reset_index(),
                total_cap.reset_index(),
                total_aqua.reset_index(),
                total_prod.reset_index(),
            ],
            ignore_index=True,
        )
        result = result.set_index(index)

        totals_order = {
            "Total selected species groups": 1,
            "Total other species groups": 2,
            "Total marine capture": 3,
            "Total aquaculture": 4,
            "Total production": 5,
        }
        sorted_result = sort_df(
            result,
            order=totals_order,
            level=self.COMMON_NAME_COL,
            sort_by=[self.GROUP_COL, self.ISSCAAP_COL, self.COMMON_NAME_COL],
        )

        appendix_landings = {
            str(group): df.droplevel(self.GROUP_COL)
            for group, df in sorted_result.groupby(level=self.GROUP_COL)
        }
        return appendix_landings

    def _compute_status_top_species(
        self,
        stock_assessments: pd.DataFrame,
        stock_landings: pd.DataFrame,
        by: str | None = None,
        filter_query: str | None = None,
        n: int = 10,
        function_name: str | None = None,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        sta_ext = self._expand_stock_assessments(stock_assessments, expand_area=False)
        stl = pd.merge(sta_ext, stock_landings, on=self.ID_COL)

        if filter_query is not None:
            stl = stl.query(filter_query)

        group_cols = [by, self.SPECIES_NAME_COL] if by else [self.SPECIES_NAME_COL]
        top_species_df = (
            stl.groupby(group_cols)[self.LANDINGS_COL]
            .sum()
            .reset_index()
            .sort_values(self.LANDINGS_COL, ascending=False)
            if not by
            else stl.groupby(group_cols)[self.LANDINGS_COL]
            .sum()
            .reset_index()
            .sort_values([by, self.LANDINGS_COL], ascending=[True, False])
        )

        top_species_index = (
            (top_species_df.groupby(by).head(n) if by else top_species_df.head(n))
            .set_index(group_cols)
            .index
        )

        agg_kwargs = {
            "by": group_cols,
            "join_table": sta_ext,
            "join_key": self.ID_COL,
            "show_counts": True,
            "show_percentages": True,
            "value_map": {"M": "Sustainable", "U": "Sustainable", "O": "Unsustainable"},
            "totals_row": False,
            "dropna_crosstab": True,
        }

        sbn = self._compute_aggregate_table(
            input_table=stock_landings,
            value=self.STATUS_COL,
            counts_name="count",
            **agg_kwargs,
        ).loc[top_species_index]

        sbl = self._compute_aggregate_table(
            input_table=stock_landings,
            value=self.STATUS_COL,
            weight_col=self.LANDINGS_COL,
            counts_scale=1e-3,
            counts_name="Landings (KT)",
            **agg_kwargs,
        ).loc[top_species_index]

        result = pd.merge(
            sbn,
            sbl,
            left_index=True,
            right_index=True,
            suffixes=("_number", "_landings"),
        )

        if by is not None:
            return {str(group): df.droplevel(by) for group, df in result.groupby(by)}

        return result

    def _compute_status_by_area(
        self,
        stock_assessments: pd.DataFrame,
        species_landings_mod: pd.DataFrame,
        compute_aggregate_table_args: dict | None = None,
        merge_southern: bool = True,
    ) -> pd.DataFrame:
        stl_ext = self.compute_stock_landings(
            stock_assessments, species_landings_mod, expanded=True
        )
        args = compute_aggregate_table_args or {}

        stl_ext[self.AREA_COL] = "Area " + stl_ext[self.AREA_COL].astype(str)
        if merge_southern:
            is_southern = stl_ext[self.AREA_COL].isin(
                [f"Area {a}" for a in self.SOUTHERN_AREAS]
            )
            stl_ext.loc[is_southern, self.AREA_COL] = "Area 48_58_88"

        return self._compute_aggregate_table(
            input_table=stl_ext,
            join_table=stock_assessments,
            join_key=self.ID_COL,
            **args,
        )

    def compute_table(
        self,
        input_table: pd.DataFrame | dict[str, pd.DataFrame],
        join_table: pd.DataFrame | dict[str, pd.DataFrame] | None,
        function_name: str,
        args: dict,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        match function_name.lower():
            case "compute_aggregate_table":
                assert isinstance(input_table, pd.DataFrame)
                return self._compute_aggregate_table(
                    input_table=input_table, join_table=join_table, **args
                )
            case "compute_percent_coverage":
                assert isinstance(input_table, dict)
                return self._compute_percent_coverage(
                    species_landings_mod=input_table["species_landings_mod"],
                    capture=input_table["capture"],
                    asfis=input_table["asfis"],
                    fao_areas=input_table.get("fao_areas"),
                    isscaap_to_exclude=self.isscaap_to_exclude,
                    **args,
                )
            case "compute_percent_coverage_rfmo":
                assert isinstance(input_table, pd.DataFrame)
                assert isinstance(join_table, pd.DataFrame)
                return self._compute_percent_coverage_rfmo(
                    capture=input_table,
                    asfis=join_table,
                    isscaap_to_exclude=self.isscaap_to_exclude,
                    **args,
                )
            case "compute_appendix_landings":
                assert isinstance(input_table, dict)
                assert isinstance(join_table, dict)
                return self._compute_appendix_landings(
                    stock_assessments=input_table["stock_assessments"],
                    species_landings=input_table["species_landings"],
                    capture=input_table["capture"],
                    aquaculture=input_table["aquaculture"],
                    asfis=join_table["asfis"],
                    countries=join_table["countries"],
                    **args,
                )
            case "compute_status_top_species":
                assert isinstance(input_table, pd.DataFrame)
                assert isinstance(join_table, pd.DataFrame)
                return self._compute_status_top_species(
                    stock_assessments=input_table,
                    stock_landings=join_table,
                    function_name=function_name,
                    **args,
                )
            case "compute_status_by_area":
                assert isinstance(input_table, pd.DataFrame)
                assert isinstance(join_table, pd.DataFrame)
                return self._compute_status_by_area(
                    stock_assessments=input_table,
                    species_landings_mod=join_table,
                    compute_aggregate_table_args=args,
                )
            case _:
                raise ValueError(
                    f"Unknown function name {function_name} passed to compute_table"
                )
