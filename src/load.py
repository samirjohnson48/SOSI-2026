"""
Class used to load data into dropbox folder
"""

import io
import logging
import pandas as pd
import os
from googleapiclient.discovery import Resource
from googleapiclient.http import MediaIoBaseUpload
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from typing import Any

from .utils import get_branch, is_step_enabled

logger = logging.getLogger(__file__)


class SOSILoader:
    MIME_TYPES = {
        "folder": "application/vnd.google-apps.folder",
        "csv": "text/csv",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "xls": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "pdf": "application/pdf",
        "png": "image/png",
        "parquet": "application/octet-stream",
    }

    LOAD_ALL_FLAG_DEFAULT = "ALL"

    DATE_FORMAT = "%Y-%m-%d"

    def __init__(
        self,
        drive_service: Resource,
        folder_id: str,
        branch_env_var: str | None = None,
    ):
        self.service: Any = drive_service
        self.folder_id = folder_id
        self.branch = get_branch(branch_env_var)

    def _get_or_create_folder(self, folder_name: str, parent_id: str) -> str:
        folder_mime_type = self.MIME_TYPES["folder"]
        query = f"name = '{folder_name}' and mimeType = '{folder_mime_type}' and trashed = false and '{parent_id}' in parents"

        results = (
            self.service.files()
            .list(
                q=query,
                fields="files(id, name)",
                spaces="drive",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        files = results.get("files", [])

        if files:
            return files[0]["id"]

        folder_metadata = {
            "name": folder_name,
            "mimeType": folder_mime_type,
            "parents": [parent_id],
        }

        new_folder = (
            self.service.files()
            .create(body=folder_metadata, fields="id", supportsAllDrives=True)
            .execute()
        )

        return new_folder.get("id")

    def _get_table_buffer(
        self,
        table: pd.DataFrame,
        extension: str,
        save_index: bool,
        table_name: str = "",
    ) -> io.BytesIO:
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
            return buffer
        except Exception as e:
            logger.error(f"Failed to upload {table_name} to Dropbox: {e}")
            raise

    def _get_figure_buffer(
        self,
        figure: Figure,
        extension: str,
        dpi: int = 300,
        figure_name: str = "",
    ) -> io.BytesIO:
        buffer = io.BytesIO()

        try:
            figure.savefig(
                buffer, format=extension.lower(), dpi=dpi, bbox_inches="tight"
            )

            buffer.seek(0)
        except Exception as e:
            logger.error(f"Failed to upload figure {figure_name} to Dropbox: {e}")
            raise
        finally:
            plt.close(figure)

        return buffer

    def _get_existing_file_id(
        self, file_name: str, current_parent_id: str
    ) -> str | None:
        query = (
            f"name = '{file_name}' and '{current_parent_id}' in parents "
            + "and trashed = false"
        )

        existing_files = (
            self.service.files()
            .list(
                q=query,
                spaces="drive",
                fields="files(id)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )

        files = existing_files.get("files")

        if files:
            return files[0]["id"]
        return None

    def _upload_media(
        self,
        media: MediaIoBaseUpload,
        file_name: str,
        current_parent_id: str | None,
        file_id: str | None = None,
    ) -> str | None:
        if file_id is not None:
            file = (
                self.service.files()
                .update(
                    fileId=file_id,
                    media_body=media,
                    fields="id",
                    supportsAllDrives=True,
                )
                .execute()
            )
        else:
            file_metadata = {
                "name": file_name,
                "parents": [current_parent_id],
            }
            file = (
                self.service.files()
                .create(
                    body=file_metadata,
                    media_body=media,
                    fields="id",
                    supportsAllDrives=True,
                )
                .execute()
            )

        return file.get("id")

    def _upload_buffer(
        self,
        buffer: io.BytesIO,
        target_path: str,
        mime_type: str,
        replace_on_exists=True,
    ):
        parts = target_path.strip("/").split("/")
        folder_parts = parts[:-1]
        file_name = parts[-1]

        current_parent_id = self.folder_id
        for folder_name in folder_parts:
            current_parent_id = self._get_or_create_folder(
                folder_name, current_parent_id
            )

        buffer.seek(0)
        media = MediaIoBaseUpload(
            buffer, chunksize=-1, mimetype=mime_type, resumable=True
        )

        try:
            file_id = self._get_existing_file_id(file_name, current_parent_id)

            if file_id is None:
                upload_id = self._upload_media(media, file_name, current_parent_id)
            elif replace_on_exists:
                upload_id = self._upload_media(
                    media, file_name, current_parent_id, file_id
                )
            else:
                current_date = datetime.now().strftime(self.DATE_FORMAT)
                name, extension = file_name.split(".")
                fn = f"{name}_{current_date}.{extension}"
                upload_id = self._upload_media(media, fn, current_parent_id)
        except Exception as e:
            print(f"Failed to upload {file_name}: {e}")
            raise

        return upload_id

    def upload_tables(
        self,
        tables: dict[str, pd.DataFrame],
        extension: str,
        table_type: str,
        pipeline_version: str,
        save_index: bool = True,
        replace_on_exists: bool = True,
        items_to_load: str | list[str] | None = None,
        load_all_flag: str = LOAD_ALL_FLAG_DEFAULT,
    ):
        pbar = tqdm(
            tables.items(), leave=False, ascii=True, colour="green", unit="table"
        )
        for table_name, table in pbar:
            if is_step_enabled(table_name, items_to_load, load_all_flag):
                pbar.set_description(f"Loading {table_name}")
                target_path = f"/{self.branch}/{pipeline_version}/tables/{table_type}/{table_name}.{extension}"
                buffer = self._get_table_buffer(
                    table, extension, save_index, table_name
                )
                self._upload_buffer(
                    buffer,
                    target_path,
                    self.MIME_TYPES.get(extension.lower(), "text/csv"),
                    replace_on_exists,
                )

    def save_tables(
        self,
        tables: dict[str, pd.DataFrame],
        extension: str,
        table_type: str,
        output_dir: Path | str,
        save_index: bool = True,
        replace_on_exists: bool = True,
    ):
        if isinstance(output_dir, str):
            output_dir = Path(output_dir).resolve()

        table_output_dir = output_dir / "tables" / table_type
        if not os.path.exists(table_output_dir):
            os.makedirs(table_output_dir)

        for table_name, table in tables.items():
            file_name = f"{table_name}.{extension}"
            if not replace_on_exists and (table_output_dir / file_name).is_file():
                current_date = datetime.now().strftime(self.DATE_FORMAT)
                file_name = f"{table_name}_{current_date}.{extension}"
            table.to_csv(table_output_dir / file_name, index=save_index)

    def upload_figures(
        self,
        figures: dict[str, Any],
        extension: str,
        dpi: int,
        pipeline_version: str,
        items_to_load: str | list[str] | None = None,
        load_all_flag: str = LOAD_ALL_FLAG_DEFAULT,
        default_mime_type: str = "image/png",
    ):
        content_pbar = tqdm(
            figures.items(), leave=False, ascii=True, colour="green", unit="figure"
        )
        for content_name, contents in content_pbar:
            if is_step_enabled(content_name, items_to_load, load_all_flag):
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
                        target_path = f"/{self.branch}/{pipeline_version}/figures/{content_name}/{name_to_save}.{extension}"
                        buffer = self._get_figure_buffer(fig, extension, dpi, fig_name)
                        self._upload_buffer(
                            buffer,
                            target_path,
                            self.MIME_TYPES.get(extension.lower(), default_mime_type),
                        )
                else:
                    target_path = f"/{self.branch}/{pipeline_version}/figures/{content_name}.{extension}"
                    buffer = self._get_figure_buffer(
                        contents, extension, dpi, content_name
                    )
                    self._upload_buffer(
                        buffer,
                        target_path,
                        self.MIME_TYPES.get(extension.lower(), default_mime_type),
                    )

    def _save_figure(
        self,
        figure: Figure,
        figure_name: str,
        extension: str,
        dpi: int,
        output_dir: Path,
        replace_on_exists: bool,
    ):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        name_to_save = figure_name.lower().replace(" ", "_")
        file_name = f"{name_to_save}.{extension}"
        if not replace_on_exists and (output_dir / file_name).is_file():
            current_date = datetime.now().strftime(self.DATE_FORMAT)
            file_name = f"{name_to_save}_{current_date}.{extension}"
        figure.savefig(
            output_dir / file_name,
            format=extension.lower(),
            dpi=dpi,
            bbox_inches="tight",
        )

    def save_figures(
        self,
        figures: dict[str, Any],
        extension: str,
        dpi: int,
        output_dir: Path | str,
        replace_on_exists: bool = True,
    ):
        if isinstance(output_dir, str):
            output_dir = Path(output_dir).resolve()

        figure_output_dir = output_dir / "figures"
        if not os.path.exists(figure_output_dir):
            os.makedirs(figure_output_dir)

        for content_name, content in figures.items():
            if isinstance(content, dict):
                for fig_name, fig in content.items():
                    self._save_figure(
                        figure=fig,
                        figure_name=fig_name,
                        extension=extension,
                        dpi=dpi,
                        output_dir=figure_output_dir / content_name,
                        replace_on_exists=replace_on_exists,
                    )
            elif isinstance(content, Figure):
                self._save_figure(
                    figure=content,
                    figure_name=content_name,
                    extension=extension,
                    dpi=dpi,
                    output_dir=figure_output_dir,
                    replace_on_exists=replace_on_exists,
                )
