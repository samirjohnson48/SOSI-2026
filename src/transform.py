"""
Transform class used transform SOSI 2026 base data
for analysis
"""

import pandas as pd
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__file__)

from .utils import join_tables


class SOSITransformer:
    # Define standard column names to be used in transformations
    AREA_LABEL_COL = "fao_area_label"
    AREA_COL = "fao_area"
    AREAS_COL = "fao_areas"
    ID_COL = "grsf_stock_id"
    ISSCAAP_COL = "isscaap_code"
    LANDINGS_COL = "landings_2023"
    LOCATION_COL = "location"
    PRODUCTION_COL = "production"
    SPECIES_CODE_COL = "asfis_code"
    SPECIES_NAME_COL = "species_name"
    STATUS_COL = "status"
    WEIGHT_COL = "weight"
    YEAR_COL = "year"

    # Indicator of whether table has global stocks
    FAO_GLOBAL_INDICATOR = "fao:global"
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

    def __init__(
        self,
        assessment_year: int,
        isscaap_to_exclude: list[int],
        error_log_dir: Path | None = None,
        global_species: list[str] = [],
        fao_areas: list[int] = [],
    ):
        self.ass_year = assessment_year
        self.isscaap_to_exclude = isscaap_to_exclude
        self.error_log_dir = error_log_dir
        self.global_species = global_species
        self.fao_areas = fao_areas

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
                    table = table.drop_duplicates(subset=primary_key, ignore_index=True)
            else:
                logger.debug(
                    f"Table {table_name} primary key {primary_key} passes check {check_name}"
                )

    def _apply_transformations(
        self,
        table: pd.DataFrame,
        transformations: dict[str, str],
        transform_from_map: dict[str, dict[str, str]],
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
            transformation = transformation_info["transformation"]
            try:
                if col in table.columns:
                    is_na = table[col].isna()
                    result.loc[is_na, col] = eval(transformation)(
                        table.loc[is_na, old_col]
                    )
                else:
                    result[col] = eval(transformation)(table[old_col])
            except SyntaxError:
                raise SyntaxError(
                    f"Incorrect syntax for transformation {transformation} applying to table {table_name} in column {old_col}"
                )

            except Exception as e:
                raise Exception(
                    f"An unexpected error occurred when applying transformation {transformation} to table {table_name} on column {col}: {e}"
                )
        for col, transformation in transformations.items():
            try:
                result[col] = eval(transformation)(result[col])
            except SyntaxError:
                raise SyntaxError(
                    f"Incorrect syntax for transformation {transformation} applying to table {table_name} in column {col}"
                )
            except Exception as e:
                raise Exception(
                    f"An unexpected error occurred when applying transformation {transformation} to table {table_name} on column {col}: {e}"
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
                rule, code = rest
                passes_mask = eval(rule)(result_table[col])
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

    def apply_schema_and_transform(
        self, table: pd.DataFrame, schema: dict, table_name: str = ""
    ) -> pd.DataFrame:
        """
        Applies schema definition (selection, transformations, data type casting, and renaming) to a table
        """
        table_cleaned = table.fillna(pd.NA)
        value_set_columns = schema.get("value_set_columns")
        if value_set_columns:
            table_cleaned = table_cleaned.dropna(subset=value_set_columns)

        (
            cols,
            transformation_map,
            transform_from_map,
            dtype_map,
            restrictions_map,
            rename_map,
        ) = self._parse_schema(schema)

        cols_to_keep = list(set(cols) & set(table_cleaned.columns))

        table_reduced = table_cleaned[cols_to_keep].copy()
        table_transformed = self._apply_transformations(
            table_reduced, transformation_map, transform_from_map, table_name
        )
        table_typed = table_transformed.astype(dtype_map)
        table_restricted = self._apply_restrictions(
            table_typed, restrictions_map, table_name
        )
        table_renamed = table_restricted.rename(columns=rename_map)

        return table_renamed

    def create_stock_reference(
        self,
        source_tables: dict[str, pd.DataFrame],
        extraction_config: dict[str, Any],
        stock_table_source: str = STOCK_TABLE_SOURCE,
        reset_index: bool = True,
    ) -> pd.DataFrame:
        stock_reference = pd.concat(
            [
                source_tables[table_name]
                for table_name in extraction_config[stock_table_source]["tables"].keys()
            ]
        )

        if reset_index:
            stock_reference = stock_reference.reset_index(drop=True)

        return stock_reference

    def create_stock_assessments(
        self,
        stock_reference: pd.DataFrame,
        status_col: str = STATUS_COL,
        status_vals: list[str] = STATUS_VALS,
        set_global_species: bool = True,
        set_fao_areas: bool = True,
        area_label_col: str = AREA_LABEL_COL,
        species_col: str = SPECIES_CODE_COL,
        global_indicator: str = FAO_GLOBAL_INDICATOR,
    ) -> pd.DataFrame:
        if set_global_species:
            mask = stock_reference[area_label_col] == global_indicator
            self.global_species = list(stock_reference[mask][species_col].unique())

        if set_fao_areas:
            self.fao_areas = list(
                stock_reference[self.AREAS_COL].dropna().explode().unique()
            )

        status_mask = stock_reference[status_col].isin(status_vals)
        return stock_reference[status_mask].reset_index(drop=True)

    def _explode_column(
        self,
        table: pd.DataFrame,
        col_to_extend: str,
        split_delimiter: str | None = None,
        new_col_name: str | None = None,
        new_col_dtype: str | None = None,
    ) -> pd.DataFrame:
        table_ext = table.copy()
        if split_delimiter is not None:
            table_ext[col_to_extend] = table_ext[col_to_extend].str.split(
                split_delimiter
            )
        table_ext = table_ext.explode(col_to_extend)
        if new_col_dtype is not None:
            table_ext[col_to_extend] = table_ext[col_to_extend].astype(
                pd.api.types.pandas_dtype(new_col_dtype)
            )
        if new_col_name is not None:
            table_ext = table_ext.rename(columns={col_to_extend: new_col_name})
        return table_ext

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

    # TODO: modify for stocks which extend across multiple areas (tuna, sharks, etc.)
    def compute_species_landings(
        self,
        stock_assessments: pd.DataFrame,
        capture: pd.DataFrame,
        first_year: int = FIRST_YEAR,
        assessment_year: int | None = None,
        species_delimiter: str = SPECIES_CODE_DELIMITER,
        production_col: str = PRODUCTION_COL,
        species_col: str = SPECIES_CODE_COL,
        area_col: str = AREA_COL,
        areas_col: str = AREAS_COL,
        year_col: str = YEAR_COL,
    ) -> pd.DataFrame:
        """
        Computes the landings for all species in the assessment
        """
        if assessment_year is None:
            assessment_year = self.ass_year
        years = range(first_year, assessment_year + 1)
        cap = self._pivot_production(
            table=capture,
            area_col=area_col,
            species_col=species_col,
            year_col=year_col,
            production_col=production_col,
            years=years,
        )

        # Explode across species
        stock_assessments_ext = self._explode_column(
            table=stock_assessments[[areas_col, species_col]],
            col_to_extend=species_col,
            split_delimiter=species_delimiter,
        )

        # Explode across areas
        stock_assessments_ext = self._explode_column(
            table=stock_assessments_ext,
            col_to_extend=areas_col,
            new_col_name=area_col,
            new_col_dtype="Int64",
        )

        stock_assessments_ext = stock_assessments_ext.drop_duplicates(
            subset=[area_col, species_col]
        )[[area_col, species_col]]

        species_landings = pd.merge(
            stock_assessments_ext, cap, how="left", on=[area_col, species_col]
        )
        return species_landings

    def _compute_species_landings_remainder(
        self,
        no_landings_species: pd.DataFrame,
        species_hierachy_mapping: pd.DataFrame,
        capture: pd.DataFrame,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def compute_stock_landings(
        self,
        stock_assessments: pd.DataFrame,
        species_landings: pd.DataFrame,
        # species_hierachy_mapping: pd.DataFrame,
        # capture: pd.DataFrame,
    ) -> pd.DataFrame:
        # no_landings_mask = species_landings[self.ass_year].fillna(0).eq(0)
        # no_landings_species = species_landings[no_landings_mask][
        #     [self.AREA_COL, self.SPECIES_CODE_COL]
        # ]
        # species_landings_mod = self._compute_species_landings_remainder(
        #     no_landings_species, species_hierachy_mapping, capture
        # )

        stock_assessments_ext = self._explode_column(
            table=stock_assessments,
            col_to_extend=self.SPECIES_CODE_COL,
            split_delimiter=self.SPECIES_CODE_DELIMITER,
        )
        stock_assessments_ext = self._explode_column(
            table=stock_assessments_ext,
            col_to_extend=self.AREAS_COL,
            new_col_name=self.AREA_COL,
            new_col_dtype="Int64",
        )

        stock_weights = (
            stock_assessments_ext.groupby(by=[self.AREA_COL, self.SPECIES_CODE_COL])[
                self.ID_COL
            ]
            .value_counts(normalize=True)
            .reset_index()
        )
        stock_weights = stock_weights.rename(columns={"proportion": self.WEIGHT_COL})

        stock_landings = pd.merge(
            species_landings, stock_weights, on=[self.AREA_COL, self.SPECIES_CODE_COL]
        )
        stock_landings[self.LANDINGS_COL] = (
            stock_landings[self.ass_year] * stock_landings[self.WEIGHT_COL]
        )
        stock_landings = (
            stock_landings.groupby(self.ID_COL)[self.LANDINGS_COL].sum().reset_index()
        )

        return stock_landings

    def _compute_counts_by_weight(
        self,
        input_table: pd.DataFrame,
        value: str,
        by: str,
        weight_col: str,
        weight_map: dict | None = None,
    ) -> pd.DataFrame:
        df = input_table.copy()
        if weight_map is not None:
            df[weight_col] = df[weight_col].map(weight_map)

        counts = df.groupby(by)[[value, weight_col]].value_counts().reset_index()
        w = "weighted_counts"
        counts[w] = counts[weight_col] * counts["count"]

        return counts.groupby([by, value])[w].sum().unstack(level=value)

    def _parse_area_label(self, area_label: str) -> int:
        area = area_label.split(":")[-1]

        if area == self.FAO_GLOBAL_INDICATOR:
            return -1
        try:
            return int(area)
        except ValueError:
            raise ValueError(f"Unknown area {area} in area label {area_label}")

    def _compute_total_landings(
        self,
        cap: pd.DataFrame,
        group_col: str,
        group_val: Any | None = None,
        isscaap_to_exclude: list[int] = [],
        remove_global_species: bool = False,
        area_col: str = AREA_COL,
        isscaap_col: str = ISSCAAP_COL,
        production_col: str = PRODUCTION_COL,
        species_col: str = SPECIES_CODE_COL,
        year_col: str = YEAR_COL,
        scale: float = 1,
    ) -> float | pd.Series:
        query_str = f"{year_col} == {self.ass_year} and {area_col} in {self.fao_areas}"

        if len(isscaap_to_exclude) > 0:
            query_str += f" and {isscaap_col} not in {isscaap_to_exclude}"

        if group_col == self.AREA_COL and group_val is not None:
            if isinstance(group_val, str):
                group_val = self._parse_area_label(group_val)
            if group_val == -1:
                query_str += f" and {species_col} in {self.global_species}"
            elif group_val >= 0:
                query_str += f" and {group_col} == {group_val}"
        elif group_val is not None:
            query_str += f" and {group_col} == {group_val}"

        if remove_global_species:
            query_str += f" and {species_col} not in {self.global_species}"

        cap = cap.query(query_str)

        if group_val is None:
            return cap.groupby(group_col)[production_col].sum() * scale

        return cap[production_col].sum() * scale

    def _compute_percent_coverage(
        self,
        species_landings: pd.DataFrame,
        capture: pd.DataFrame,
        asfis: pd.DataFrame,
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

        cap = pd.merge(capture, asfis, on=self.SPECIES_CODE_COL)
        if fao_areas is not None:
            cap = pd.merge(cap, fao_areas, on=self.AREA_COL)
            spl = pd.merge(species_landings, fao_areas, on=self.AREA_COL)
        else:
            spl = species_landings
        coverage = spl.groupby(group_col)[self.ass_year].sum() * scale

        total_landings = self._compute_total_landings(
            cap,
            group_col=group_col,
            isscaap_to_exclude=isscaap_to_exclude,
            scale=scale,
        )
        assert isinstance(total_landings, pd.Series)

        percent_coverage = pd.merge(
            coverage,
            total_landings,
            how="left",
            left_index=True,
            right_index=True,
        )

        unit = self.SCALING_UNIT_MAP[scale]
        coverage_col = f"coverage ({unit}T)"
        total_landings_col = f"total_landings ({unit}T)"
        percent_coverage = percent_coverage.rename(
            columns={
                self.ass_year: coverage_col,
                self.PRODUCTION_COL: total_landings_col,
            }
        )

        if totals_row:
            percent_coverage.loc[totals_label] = percent_coverage.sum()

        pc_col = "coverage (%)"
        percent_coverage[pc_col] = (
            percent_coverage[coverage_col] / percent_coverage[total_landings_col] * 100
        )

        return percent_coverage

    def _compute_aggregate_table(
        self,
        input_table: pd.DataFrame,
        value: str,
        by: str,
        join_table: pd.DataFrame | dict[str, pd.DataFrame] | None = None,
        join_key: list[str] | str | dict[str, str] | dict[str, list[str]] | None = None,
        show_counts: bool = True,
        show_percentages: bool = True,
        value_map: dict[str, str] | None = None,
        totals_row: bool = True,
        totals_label: str = TOTALS_LABEL,
        weight_col: str | None = None,
        weight_map: dict | None = None,
    ) -> pd.DataFrame:
        if not (show_counts or show_percentages):
            raise ValueError(
                "compute_aggregate_table usage: one of show_counts or show_percentages must be True"
            )

        data: pd.DataFrame
        if join_table is not None and join_key is not None:
            data = join_tables(input_table, join_table, join_key)
        else:
            data = input_table.copy()

        if weight_col is not None:
            counts = self._compute_counts_by_weight(
                input_table=data,
                value=value,
                by=by,
                weight_col=weight_col,
                weight_map=weight_map,
            )
        else:
            counts = data.groupby(by)[value].value_counts().unstack(level=value)

        if totals_row:
            counts.loc[totals_label] = counts.sum()

        metrics: dict[str, pd.DataFrame] = {}
        if show_counts:
            metrics["count"] = counts
        if show_percentages:
            metrics["percentage"] = counts.div(counts.sum(axis=1), axis=0) * 100

        if value_map is not None:
            mapped_metrics: dict[str, pd.DataFrame] = {}
            for label, df in metrics.items():
                mapped_metrics[label] = df.T.groupby(df.columns.map(value_map)).sum().T

            for label in metrics:
                metrics[label] = pd.concat(
                    [metrics[label], mapped_metrics[label]], axis=1
                )

        return pd.concat(metrics, axis=1)

    def compute_table(
        self,
        input_table: pd.DataFrame | dict[str, pd.DataFrame],
        join_table: pd.DataFrame | dict[str, pd.DataFrame] | None,
        function_name: str,
        args: dict,
    ) -> pd.DataFrame:
        match function_name.lower():
            case "compute_aggregate_table":
                assert isinstance(input_table, pd.DataFrame)
                return self._compute_aggregate_table(
                    input_table=input_table, join_table=join_table, **args
                )
            case "compute_percent_coverage":
                assert isinstance(input_table, dict)
                return self._compute_percent_coverage(
                    species_landings=input_table["species_landings"],
                    capture=input_table["capture"],
                    asfis=input_table["asfis"],
                    fao_areas=input_table.get("fao_areas"),
                    isscaap_to_exclude=self.isscaap_to_exclude,
                    **args,
                )
            case _:
                raise ValueError()
