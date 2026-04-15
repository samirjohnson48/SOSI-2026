"""
Class used to load data into Google Drive and PostgreSQL Server
"""

import io
import inspect
import logging
import pandas as pd
import os
import sys
from googleapiclient.discovery import Resource
from googleapiclient.http import MediaIoBaseUpload
from googleapiclient.errors import HttpError
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from flask_migrate import upgrade
from sqlalchemy import inspect as sqlinspect, text
from sqlalchemy.orm.mapper import Mapper
from sqlalchemy.orm.decl_api import DeclarativeAttributeIntercept

from .utils import get_branch

project_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(project_dir))

from web_app.app import create_app
from web_app import models

logger = logging.getLogger(__name__)


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
    DATE_FORMAT = "%Y-%m-%d"
    DB_URL_ENV_VAR = "NEON_DATABASE_URL"

    def __init__(
        self,
        drive_service_oauth: Resource,
        drive_service_account: Resource,
        sheets_service: Resource,
        drive_folder_id: str,
        db_engine: Any,
        branch_env_var: str | None = None,
    ):
        self.drive_service_oauth: Any = drive_service_oauth
        self.drive_service_account: Any = drive_service_account
        self.sheets_service: Any = sheets_service
        self.folder_id = drive_folder_id
        self.engine = db_engine
        self.branch = get_branch(branch_env_var)
        self.model_map = self._get_model_map()

    def _get_model_map(self):
        model_map = {}
        for _, obj in inspect.getmembers(models):
            if inspect.isclass(obj):
                if hasattr(obj, "__tablename__"):
                    model_map[obj.__tablename__] = obj

        return model_map

    def _get_or_create_folder(self, folder_name: str, parent_id: str) -> str:
        folder_mime_type = self.MIME_TYPES["folder"]
        query = f"name = '{folder_name}' and mimeType = '{folder_mime_type}' and trashed = false and '{parent_id}' in parents"

        results = (
            self.drive_service_oauth.files()
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
            self.drive_service_oauth.files()
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
        self, file_name: str, current_parent_id: str | None = None
    ) -> str | None:
        query = f"name = '{file_name}' and trashed = false"

        if current_parent_id is not None:
            query += f" and '{current_parent_id}' in parents"

        existing_files = (
            self.drive_service_account.files()
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
                self.drive_service_oauth.files()
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
                self.drive_service_oauth.files()
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

    def upload_tables_drive(
        self,
        tables: Mapping[str, pd.DataFrame | dict[str, pd.DataFrame]],
        extension: str,
        table_type: str,
        pipeline_version: str,
        save_index: bool = True,
        replace_on_exists: bool = True,
    ):
        pbar = tqdm(
            tables.items(), leave=False, ascii=True, colour="green", unit="table"
        )
        for table_name, table in pbar:
            if isinstance(table, dict):
                self.upload_tables_drive(
                    tables=table,
                    extension=extension,
                    table_type=f"{table_type}/{table_name}",
                    pipeline_version=pipeline_version,
                    save_index=save_index,
                    replace_on_exists=replace_on_exists,
                )
            else:
                pbar.set_description(f"Loading {table_name} into drive")
                tn_clean = table_name.lower().replace(" ", "_").replace(",", "")
                target_path = f"/{self.branch}/{pipeline_version}/tables/{table_type}/{tn_clean}.{extension}"
                buffer = self._get_table_buffer(
                    table, extension, save_index, table_name
                )
                self._upload_buffer(
                    buffer,
                    target_path,
                    self.MIME_TYPES.get(extension.lower(), "text/csv"),
                    replace_on_exists,
                )

    def _col_to_letter(self, col: int):
        """Converts a 0-indexed column number to a Google Sheets letter (e.g., 0 -> A)."""
        letter = ""
        col += 1
        while col > 0:
            col, remainder = divmod(col - 1, 26)
            letter = chr(65 + remainder) + letter
        return letter

    def _resolve_column_addresses(
        self, cols: str | list[str], file_id: str, sheet_name: str
    ) -> str | dict[str, str]:
        header_result = (
            self.sheets_service.spreadsheets()
            .values()
            .get(spreadsheetId=file_id, range=f"'{sheet_name}'!1:1")
            .execute()
        )
        headers = header_result.get("values", [])[0]

        if isinstance(cols, str):
            return self._col_to_letter(headers.index(cols))

        letters = {}
        for col in cols:
            letters[col] = self._col_to_letter(headers.index(col))

        return letters

    def update_catch_col(
        self,
        file_name: str,
        sheet_name: str,
        stock_landings: pd.DataFrame,
        assessment_year: int,
        id_col: str = "stock_id",
        landings_col: str = "landings",
        catch_col: str = "catch",
        catch_year_col: str = "catch_year",
    ):
        file_id = self._get_existing_file_id(file_name)
        if file_id is None:
            raise FileNotFoundError(f"File '{file_name}' not found in Drive.")

        column_addresses = self._resolve_column_addresses(
            [id_col, catch_col, catch_year_col], file_id, sheet_name
        )
        assert isinstance(column_addresses, dict)
        id_a = column_addresses.get(id_col)
        catch_a = column_addresses.get(catch_col)
        catch_year_a = column_addresses.get(catch_year_col)

        response = (
            self.sheets_service.spreadsheets()
            .values()
            .batchGet(spreadsheetId=file_id, ranges=[f"'{sheet_name}'!{id_a}:{id_a}"])
            .execute()
        )

        sheet_ids = response.get("valueRanges", [])[0].get("values", [])

        id_map = {str(row[0]): i + 1 for i, row in enumerate(sheet_ids) if row}

        updates = []
        for _, row in stock_landings.iterrows():
            val_id = str(row[id_col])
            if val_id in id_map:
                updates += [
                    {
                        "range": f"'{sheet_name}'!{catch_a}{id_map[val_id]}",
                        "values": [[row[landings_col]]],
                    },
                    {
                        "range": f"'{sheet_name}'!{catch_year_a}{id_map[val_id]}",
                        "values": [[assessment_year]],
                    },
                ]

        body = {"valueInputOption": "RAW", "data": updates}
        try:
            self.sheets_service.spreadsheets().values().batchUpdate(
                spreadsheetId=file_id, body=body
            ).execute()
        except HttpError as e:
            match e.status_code:
                case 403:
                    print(
                        f"Must share {file_name} with service account email to update catch."
                    )
                case _:
                    raise e

    def sync_database_schema(self):
        app = create_app()
        with app.app_context():
            try:
                upgrade()
            except Exception as e:
                print(f"Error when syncing the database schema")
                raise e

    def _get_sql_dtypes(self, mapper: Mapper) -> dict:
        dtype_map = {}

        for column in mapper.attrs:
            if hasattr(column, "columns"):
                col_obj = column.columns[0]
                dtype_map[column.key] = col_obj.type

        return dtype_map

    def _get_pk(self, mapper: Mapper) -> str:
        return ", ".join(c.name for c in mapper.primary_key)

    def _upsert_table_db(
        self, table: pd.DataFrame, table_name: str, model: DeclarativeAttributeIntercept
    ) -> None:
        logger.info(f"Beginning upsert for {table_name}.")

        mapper = sqlinspect(model)
        pk = self._get_pk(mapper)
        sql_dtypes = self._get_sql_dtypes(mapper)

        valid_cols = [col for col in table.columns if col in sql_dtypes]
        temp_table = table[valid_cols].copy()
        temp_table_name = f"temp_{table_name}"

        try:
            with self.engine.begin() as conn:
                temp_table.to_sql(
                    temp_table_name,
                    conn,
                    if_exists="replace",
                    index=False,
                    dtype=sql_dtypes,
                )
                logger.debug(f"Temp table {temp_table_name} created.")

                update_cols = [c for c in valid_cols if c != pk]
                update_stmt = ", ".join(
                    [f'"{c}" = EXCLUDED."{c}"' for c in update_cols]
                )

                valid_cols_str = ", ".join([f'"{c}"' for c in valid_cols])
                upsert_query = text(f"""
                                    INSERT INTO {table_name} ({valid_cols_str})
                                    SELECT {valid_cols_str} FROM {temp_table_name}
                                    ON CONFLICT ({pk})
                                    DO UPDATE SET {update_stmt};
                                    """)

                conn.execute(upsert_query)
                logger.info(f"Upsert for {table_name} complete.")

                conn.execute(text(f"DROP TABLE {temp_table_name}"))
                logger.debug(f"Temp table {temp_table_name} dropped.")
        except Exception as e:
            print(f"An error occurred when performing upsert on {table_name}")
            raise e

    def upload_tables_db(
        self,
        tables: dict[str, pd.DataFrame],
    ) -> None:
        pbar = tqdm(
            tables.items(), leave=False, ascii=True, colour="green", unit="table"
        )
        for table_name, table in pbar:
            pbar.set_description(f"Loading {table_name} into database")
            logger.info(f"Loading {table_name} into database")
            clean_name = table_name.lower().replace(" ", "_").replace("-", "_")

            model = self.model_map.get(clean_name)
            if model is None:
                print(
                    f"Model not found for table {table_name}. Cannot upload to database"
                )
                continue

            self._upsert_table_db(table, clean_name, model)

    def save_tables(
        self,
        tables: Mapping[str, pd.DataFrame | dict[str, pd.DataFrame]],
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

        for name, content in tables.items():
            if isinstance(content, dict):
                self.save_tables(
                    tables=content,
                    extension=extension,
                    table_type=f"{table_type}/{name}",
                    output_dir=output_dir,
                    save_index=save_index,
                    replace_on_exists=replace_on_exists,
                )
            else:
                file_name = f"{name}.{extension}"
                if not replace_on_exists and (table_output_dir / file_name).is_file():
                    current_date = datetime.now().strftime(self.DATE_FORMAT)
                    file_name = f"{name}_{current_date}.{extension}"
                content.to_csv(table_output_dir / file_name, index=save_index)

    def upload_figures(
        self,
        figures: dict[str, Any],
        extension: str,
        dpi: int,
        pipeline_version: str,
        default_mime_type: str = "image/png",
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
                    target_path = f"/{self.branch}/{pipeline_version}/figures/{content_name}/{name_to_save}.{extension}"
                    buffer = self._get_figure_buffer(fig, extension, dpi, fig_name)
                    self._upload_buffer(
                        buffer,
                        target_path,
                        self.MIME_TYPES.get(extension.lower(), default_mime_type),
                    )
            else:
                target_path = f"/{self.branch}/{pipeline_version}/figures/{content_name}.{extension}"
                buffer = self._get_figure_buffer(contents, extension, dpi, content_name)
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
