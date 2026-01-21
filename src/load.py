"""
Class used to load data into dropbox folder
"""

import io
import logging
import pandas as pd
import dropbox
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm
from typing import Any

logger = logging.getLogger(__name__)


class SOSILoader:
    def __init__(
        self, dbx: dropbox.Dropbox, table_extension: str, figure_extension: str
    ):
        self.dbx = dbx
        self.table_extension = table_extension
        self.figure_extension = figure_extension

    def _upload_table(
        self,
        table: pd.DataFrame,
        target_path: str,
        extension: str,
        save_index: bool,
        table_name: str = "",
    ) -> None:
        """
        Serializes a DataFrame and uploads it to Dropbox at a versioned path.
        Path: /version/table_type/table_name.extension
        """
        buffer = io.BytesIO()

        try:
            if extension.lower() == "csv":
                table.to_csv(buffer, index=save_index)
            elif extension.lower() in ["xlsx", "xls"]:
                table.to_excel(buffer, index=save_index, engine="openpyxl")
            elif extension.lower() == "parquet":
                table.to_parquet(buffer, index=save_index)
            else:
                raise ValueError(f"Unsupported upload extension: {extension}")

            buffer.seek(0)
            self.dbx.files_upload(
                buffer.getvalue(), target_path, mode=dropbox.files.WriteMode.overwrite
            )

            logger.info(f"Successfully uploaded {table_name} to {target_path}")

        except Exception as e:
            logger.error(f"Failed to upload {table_name} to Dropbox: {e}")
            raise

    def upload_tables(
        self,
        tables: dict[str, pd.DataFrame],
        table_type: str,
        pipeline_version: str,
        save_index=True,
    ):
        pbar = tqdm(
            tables.items(), leave=False, ascii=True, colour="green", unit="table"
        )
        for table_name, table in pbar:
            pbar.set_description(f"Loading {table_name}")
            target_path = f"/{pipeline_version}/tables/{table_type}/{table_name}.{self.table_extension}"
            self._upload_table(
                table,
                target_path,
                self.table_extension,
                save_index,
                table_name=table_name,
            )

    def _upload_figure(
        self,
        figure: Figure,
        target_path: str,
        extension: str,
        dpi: int = 300,
        figure_name: str = "",
    ) -> None:
        """
        Serializes a Matplotlib figure and uploads it to Dropbox.
        Path: /version/figure_type/figure_name.extension
        """
        buffer = io.BytesIO()

        try:
            figure.savefig(
                buffer, format=extension.lower(), dpi=dpi, bbox_inches="tight"
            )

            buffer.seek(0)
            self.dbx.files_upload(
                buffer.getvalue(), target_path, mode=dropbox.files.WriteMode.overwrite
            )

            logger.info(f"Successfully uploaded figure {figure_name} to {target_path}")

        except Exception as e:
            logger.error(f"Failed to upload figure {figure_name} to Dropbox: {e}")
            raise

        finally:
            plt.close(figure)
            buffer.close()

    def upload_figures(
        self,
        figures: dict[str, Any],
        pipeline_version: str,
        dpi: int,
    ):
        content_pbar = tqdm(
            figures.items(), leave=False, ascii=True, colour="green", unit="figure"
        )
        for content_name, contents in content_pbar:
            content_pbar.set_description(f"Loading {content_name}")
            if isinstance(contents, dict):
                fig_pbar = tqdm(
                    contents.items(),
                    leave=False,
                    ascii=True,
                    colour="green",
                    unit="figure",
                )
                for fig_name, fig in fig_pbar:
                    name_to_save = fig_name.lower().replace(" ", "_")
                    fig_pbar.set_description(f"Loading {fig_name}")
                    target_path = f"/{pipeline_version}/figures/{content_name}/{name_to_save}.{self.figure_extension}"
                    self._upload_figure(
                        fig, target_path, self.figure_extension, dpi, fig_name
                    )
            else:
                target_path = (
                    f"{pipeline_version}/figures/{content_name}.{self.figure_extension}"
                )
                self._upload_figure(
                    contents, target_path, self.figure_extension, dpi, content_name
                )
