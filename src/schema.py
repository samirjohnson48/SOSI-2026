"""
Define restrictions and transformations to be used on data ingestion.
These functions are then specified for columns
in the config/schema YAML files
"""

import numpy as np
import pandas as pd


class SchemaRules:
    GLOBAL_FLAG = "fao:global"
    SOUTHERN_AREAS = [48, 58, 88]
    SOUTHERN_AREAS_ANALYSIS_GROUP = "Area48_58_88"
    TLW_UNIT = "Q_tlw"

    # ----- TRANSFORMATIONS -----
    def clean_isscaap_group(self, col: pd.Series) -> pd.Series:
        return col.str.strip('"')

    def clean_species_name(self, col: pd.Series) -> pd.Series:
        return col.fillna("_").str.split(";")

    def clean_asfis_code(self, col: pd.Series) -> pd.Series:
        return (
            col.fillna("_")
            .str.replace("asfis:", "")
            .str.replace(",", " ")
            .str.replace(";", " ")
            .str.split()
        )

    def strip_list(self, col: pd.Series) -> pd.Series:
        return col.apply(lambda x: [s.strip() for s in x])

    def create_fao_areas(
        self,
        fao_area_df: pd.DataFrame,
        fao_area_col: str = "fao_area",
        subarea_col: str = "subarea_description",
    ) -> pd.Series:
        fao_area = fao_area_df[fao_area_col]
        subarea = fao_area_df[subarea_col]
        global_mask = fao_area.eq(self.GLOBAL_FLAG)

        assert sum(subarea.loc[global_mask].isna()) == 0, (
            f"Must specify '{subarea_col}' for all stocks with '{fao_area_col}' == '{self.GLOBAL_FLAG}'"
        )

        fao_area_clean = fao_area.str.split(":").str[-1]
        fao_areas_str = pd.concat(
            [fao_area_clean.loc[~global_mask], subarea.loc[global_mask]]
        ).sort_index()
        fao_areas = fao_areas_str.str.split(", ").apply(
            lambda col: [int(a) for a in col]
        )
        return fao_areas

    def fill_na_true(self, col: pd.Series) -> pd.Series:
        return col.fillna(True)

    def create_fao_area_label(self, col: pd.Series) -> pd.Series:
        return "fao:" + col.astype(str)

    def create_sosi_grouping(self, col: pd.Series) -> pd.Series:
        return pd.Series(
            np.where(
                col.isin(self.SOUTHERN_AREAS),
                self.SOUTHERN_AREAS_ANALYSIS_GROUP,
                "Area" + col.astype(str),
            ),
            index=col.index,
            name=col.name,
        )

    # ----- RESTRICTIONS -----
    def not_na(self, col: pd.Series) -> pd.Series:
        return col.notna()

    def check_length_3(self, col: pd.Series) -> pd.Series:
        return col.str.len().eq(3)

    def check_tlw(self, col: pd.Series) -> pd.Series:
        return col.eq(self.TLW_UNIT)
