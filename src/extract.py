"""
Data extractor class for all necessary datasets used in the
reporting of SOSI 2026.
"""

import pandas as pd
import requests
import zipfile
import io
import csv
import os
import warnings
import logging
from pathlib import Path
from typing import Any
from tqdm import tqdm
from googleapiclient.discovery import Resource
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

logger = logging.getLogger(__file__)


class SOSIExtractor:
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    CACHE_DIR = PROJECT_DIR / ".cache"
    SAVE_FILE_EXTENSION = "csv"
    EXTRACT_ALL_FLAG = "SELECT_ALL"

    def __init__(
        self,
        drive_service: Resource,
        sheets_service: Resource,
        cache_dir: Path | str = CACHE_DIR,
    ):
        """
        Used to extract all necessary datasets for SOSI 2026 data processing
        """
        self.drive_service: Any = drive_service
        self.sheets_service: Any = sheets_service
        self.cache_dir = Path(cache_dir)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _remove_cached_files(self, cache_dir: Path | str = CACHE_DIR):
        for file_path in Path(cache_dir).iterdir():
            os.remove(file_path)

    def _get_table_from_url(
        self,
        table_url: str,
        table_format: str,
        auth_token_env_var: str | None = None,
        sheet_name: str | int = 0,
        extract_to: Path | str | None = None,
    ) -> pd.DataFrame:
        """
        Retrieves table directly from download url
        """
        logger.info(f"-> Extracting table from url: {table_url}")
        if extract_to is None:
            extract_to = self.cache_dir

        logger.debug(f"Table url: {table_url}")
        logger.debug(f"Auth token: {auth_token_env_var}")

        try:
            headers = (
                {"Authorization": f"Bearer {os.getenv(auth_token_env_var)}"}
                if auth_token_env_var
                else None
            )

            response = requests.get(table_url, headers=headers)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error in downloading table from url: {table_url}")
            raise e

        file_content = io.BytesIO(response.content)
        df = pd.DataFrame()

        match table_format:
            case "xlsx":
                # Catch no default stylesheet warning
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df = pd.read_excel(file_content, sheet_name=sheet_name)
            case "csv":
                df = pd.read_csv(file_content)
            case _:
                raise ValueError(f"Invalid table format: {table_format}")

        file_content.close()
        logger.info("-> Successfully extracted table into Pandas DataFrame")
        return df

    def _get_table_from_path(
        self,
        file_path: Path,
        remove_file: bool = False,
    ) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        logger.info("-> Successfully extracted file into Pandas DataFrame")
        if remove_file:
            os.remove(file_path)
        else:
            logger.info(f"-> Saved file to {file_path}")

        return df

    def _get_tables_from_zip(
        self,
        zip_url: str,
        tables: dict[str, dict[str, str]],
    ) -> dict[str, pd.DataFrame]:
        """
        Loads datasets from a zip file
        Works for capture/aquaculture production & ASFIS datasets hosted on FAO
        """
        logger.info(
            f"-> Extracting tables {', '.join(tables.keys())} from url: {zip_url}"
        )
        # Set up temporary directory
        extract_to = self.PROJECT_DIR / "temp"

        try:
            response = requests.get(zip_url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error in downloading zip from url: {zip_url}")
            raise e

        zip_in_memory = io.BytesIO(response.content)

        with zipfile.ZipFile(zip_in_memory, "r") as zip_ref:
            for table_name, table_info in tables.items():
                file_name = table_info.get("file_name")
                if file_name not in zip_ref.namelist():
                    raise FileNotFoundError(
                        f"Target file {file_name} for table {table_name} not found in zip file at url: {zip_url}."
                    )
                zip_ref.extract(file_name, extract_to)

        zip_in_memory.close()

        dfs: dict[str, pd.DataFrame] = {}
        for table_name, table_info in tables.items():
            dfs[table_name] = self._get_table_from_path(
                extract_to / table_info["file_name"], remove_file=True
            )

        extract_to.rmdir()

        return dfs

    def _get_file_info_by_name(self, file_name: str) -> tuple[str, str]:
        """
        Retrieves the Google file id and mime type from the corresponding service's Google
        drive and the given file name
        """
        search_query = f"name = '{file_name}' and trashed = false"

        try:
            response = (
                self.drive_service.files()
                .list(
                    q=search_query,
                    spaces="drive",
                    fields="files(id, mimeType, name)",
                    pageSize=1,
                )
                .execute()
            )

            items = response.get("files", [])

            if not items:
                raise FileNotFoundError(
                    f"File named '{file_name}' not found or service account cannot access it."
                )

            file_id = items[0]["id"]
            mime_type = items[0]["mimeType"]
            logger.debug(
                f"-> Found file id: {file_id} corresponding to file name {file_name}"
            )
            return file_id, mime_type
        except HttpError as e:
            logger.error(f"API Error during retrieval of {file_name}")
            raise e
        except Exception as e:
            logger.error("An unexpected error occurred")
            raise e

    def _get_file_content_by_info_drive(
        self, file_id: str, mime_type: str
    ) -> io.BytesIO:
        file_content = io.BytesIO()
        if mime_type == "application/vnd.google-apps.spreadsheet":
            request = self.drive_service.files().export_media(
                fileId=file_id,
                mimeType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            request = self.drive_service.files().get_media(fileId=file_id)

        downloader = MediaIoBaseDownload(file_content, request)
        done = False
        while done is False:
            _, done = downloader.next_chunk()

        logger.debug(f"-> Download complete for ID: {file_id}")

        file_content.seek(0)

        return file_content

    def _get_file_content_by_info_sheets(
        self, file_id: str, sheet_name: str | int
    ) -> io.BytesIO:
        logger.debug(f"-> Fetching sheet: {sheet_name} from ID: {file_id}")

        result = (
            self.sheets_service.spreadsheets()
            .values()
            .get(
                spreadsheetId=file_id,
                range=sheet_name,
                valueRenderOption="UNFORMATTED_VALUE",  # Faster: skips formatting logic
            )
            .execute()
        )

        rows = result.get("values", [])

        # Convert list of lists to CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerows(rows)

        # Convert to BytesIO to remain compatible with your existing file-like flow
        file_content = io.BytesIO(output.getvalue().encode("utf-8"))
        file_content.seek(0)
        return file_content

    def _get_table_from_drive(
        self,
        file_name: str,
        service_name: str,
        sheet_name: str | int = 1,
        table_name: str = "",
    ) -> pd.DataFrame:
        """
        Retrieves df from file in Google Drive via Google Service Account
        Must pass service object created from SOSIAuthenticator class
        Used for accessing stock assessments remotely from SoSI2026-workspace

        Valid file formats:
            - csv
            - xlsx
        """
        logger.info(f"-> Extracting {file_name} from drive.")

        try:
            df: pd.DataFrame
            file_id, mime_type = self._get_file_info_by_name(file_name)
            match service_name:
                case "drive":
                    file_content = self._get_file_content_by_info_drive(
                        file_id, mime_type
                    )
                    # Catch no default stylesheet warning
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        df = pd.read_excel(file_content, sheet_name=sheet_name)
                case "sheets":
                    file_content = self._get_file_content_by_info_sheets(
                        file_id, sheet_name
                    )
                    df = pd.read_csv(file_content, encoding="utf-8")
                case _:
                    raise ValueError(
                        "service_name specified must either be 'drive' or 'sheets'"
                    )
            logger.info("-> Successfully loaded data into Pandas DataFrame.")

            file_content.close()
            return df

        except FileNotFoundError as e:
            logger.error(
                f"Could not find file: {file_name} in Drive. Did you specify the correct file for table: {table_name}"
            )
            raise e
        except ValueError as e:
            logger.error(
                f"A value error has occurred. Did you specify the correct sheet name: {sheet_name} for the table: {table_name}?"
            )
            raise e
        except HttpError as e:
            logger.error(f"API Error during retrieval of {table_name}")
            raise e
        except Exception as e:
            logger.error("An unexpected error occurred when retrieving {table_name}: ")
            raise e

    def _dispatch_source_extraction(
        self,
        source_type: str,
        source_tables: dict,
        source_url: str | None = None,
        source_name: str = "",
    ) -> dict[str, pd.DataFrame]:
        tables = {}

        match source_type:
            case "sheets":
                pbar = tqdm(
                    source_tables.items(),
                    leave=False,
                    colour="green",
                    ascii=True,
                    unit="tables",
                )
                for table_name, info in pbar:
                    pbar.set_description(f"Extracting {table_name}")
                    if "file_name" not in info:
                        raise KeyError(
                            f"'file_name' not specified for table {table_name}"
                        )
                    elif "sheet_name" not in info:
                        raise KeyError(
                            f"'sheet_name' not specified for table {table_name}"
                        )
                    tables[table_name] = self._get_table_from_drive(
                        info["file_name"],
                        "sheets",
                        info["sheet_name"],
                        table_name,
                    )
            case "drive":
                pbar = tqdm(
                    source_tables.items(),
                    leave=False,
                    colour="green",
                    ascii=True,
                    unit="tables",
                )
                for table_name, info in pbar:
                    pbar.set_description(f"Extracting {table_name}")
                    if "file_name" not in info:
                        raise KeyError(
                            f"'file_name' not specified for table {table_name}"
                        )
                    elif "sheet_name" not in info:
                        raise KeyError(
                            f"'sheet_name' not specified for table {table_name}"
                        )
                    tables[table_name] = self._get_table_from_drive(
                        info["file_name"],
                        "drive",
                        info["sheet_name"],
                        table_name,
                    )
            case "zip":
                if source_url is None:
                    raise KeyError(f"'url' not specified for source {source_name}")
                tables = self._get_tables_from_zip(
                    source_url,
                    source_tables,
                )
            case _:
                raise ValueError(
                    f"Unknown source type {source_type} specified for {source_name} in extraction."
                )

        return tables

    def _get_table_from_cache(
        self,
        table_name: str,
        cache_dir: Path,
        file_extension: str = SAVE_FILE_EXTENSION,
        table_info: dict[str, str] | None = None,
        source_type: str | None = None,
        source_url: str | None = None,
        source_name: str = "",
    ) -> pd.DataFrame:
        file_name = f"{table_name}.{file_extension}"

        logger.info(f"Extraction {table_name} from cache")

        if not os.path.exists(cache_dir / file_name):
            logger.info(f"Could not find {file_name} in cache.")
            print(
                f"Could not find {file_name} in cache. Retrieving from source: {source_name}",
                end="\r",
            )

            assert table_info is not None
            assert source_type is not None

            source_table = {table_name: table_info}
            table = self._dispatch_source_extraction(
                source_type, source_table, source_url, source_name
            )[table_name]

            file_name = f"{table_name}.{self.SAVE_FILE_EXTENSION}"
            table.to_csv(self.cache_dir / file_name, index=False)

            return table

        match file_extension:
            case "csv":
                return pd.read_csv(cache_dir / file_name)
            case "xlsx":
                return pd.read_excel(cache_dir / file_name)
            case _:
                raise ValueError(
                    f"Invalid file extension {file_extension} given to retrieve table {table_name} from cache"
                )

    def extract_tables(
        self,
        source_info: dict,
        source_name: str = "",
        extract_args: list | str | None = None,
        extract_all_flag: str = EXTRACT_ALL_FLAG,
    ) -> dict[str, pd.DataFrame]:
        """
        Extract table from source given the source information
        """
        tables: dict[str, pd.DataFrame] = {}

        if extract_args == extract_all_flag:
            tables = self._dispatch_source_extraction(
                source_info["source_type"],
                source_info["tables"],
                source_info.get("url"),
                source_name,
            )
            for table_name, table in tables.items():
                fp = self.cache_dir / f"{table_name}.{self.SAVE_FILE_EXTENSION}"
                table.to_csv(fp, index=False)
        else:
            for table_name, table_info in source_info["tables"].items():
                if extract_args is not None and table_name in extract_args:
                    table = self._dispatch_source_extraction(
                        source_info["source_type"],
                        {table_name: table_info},
                        source_info.get("url"),
                        source_name,
                    )[table_name]
                    fp = self.cache_dir / f"{table_name}.{self.SAVE_FILE_EXTENSION}"
                    table.to_csv(fp, index=False)
                    tables[table_name] = table
                else:
                    tables[table_name] = self._get_table_from_cache(
                        table_name,
                        self.cache_dir,
                        table_info=table_info,
                        source_type=source_info["source_type"],
                        source_url=source_info.get("url"),
                        source_name=source_name,
                    )

        return tables
