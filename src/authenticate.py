"""
Authenticator class to extract and load necessary data used in the
reporting of SOSI 2026.
Authenticates Google Service Account for extraction from Google Drive
and Dropbox for loading.
"""

import os
from google.oauth2 import service_account
from googleapiclient.discovery import build, Resource
from googleapiclient.errors import HttpError
from dropbox import Dropbox
import logging

logger = logging.getLogger(__name__)


class SOSIAuthenticator:
    def __init__(
        self,
        google_service_account_creds_env_var: str,
        google_scopes: list,
        dropbox_app_key_env_var: str,
        dropbox_app_secret_env_var: str,
        dropbox_app_refresh_token_env_var: str,
    ):
        self.google_service_account_creds_env_var = google_service_account_creds_env_var
        self.google_scopes = google_scopes
        self.dbx_key_env_var = dropbox_app_key_env_var
        self.dbx_secret_env_var = dropbox_app_secret_env_var
        self.dbx_refresh_token_env_var = dropbox_app_refresh_token_env_var

    def _get_google_service_credentials(self):
        try:
            creds_path = os.getenv(self.google_service_account_creds_env_var)

            if not creds_path:
                raise KeyError(
                    f"Set environment variable {self.google_service_account_creds_env_var} to point to path of Google Service Account credentials."
                )

            credentials = service_account.Credentials.from_service_account_file(
                creds_path, scopes=self.google_scopes
            )
            return credentials
        except FileNotFoundError as e:
            print(
                f"Could not find Google service account credentials at path: {creds_path}"
            )
            raise e
        except Exception as e:
            print("An unexpected error occurred: ")
            raise e

    def get_google_service(self, service_name: str) -> Resource:
        """
        Returns a Google Resource object to interact with the API
        used in extracting stock assessment files from Google Drive
        """
        if service_name not in ["drive", "sheets"]:
            raise ValueError(
                "Specify either 'drive' or 'sheets' for the type of service account."
            )

        versions = {"drive": "v3", "sheets": "v4"}
        try:
            creds = self._get_google_service_credentials()
            service = build(
                serviceName=service_name,
                version=versions[service_name],
                credentials=creds,
                static_discovery=True,
            )
            logger.info(
                f"Successfully created Google {service_name.capitalize()} client."
            )
        except HttpError as e:
            print("An error occurred connecting to the Google API")
            raise e
        except Exception as e:
            print("An unexpected error occurred: ")
            raise e

        return service

    def get_dropbox_client(self):
        key = os.getenv(self.dbx_key_env_var)
        secret = os.getenv(self.dbx_secret_env_var)
        refresh_token = os.getenv(self.dbx_refresh_token_env_var)
        if any(_ is None for _ in [key, secret, refresh_token]):
            raise KeyError(
                f"Dropbox app key, secret, or refresh token is not set as environment variables."
            )

        dbx = Dropbox(
            app_key=key, app_secret=secret, oauth2_refresh_token=refresh_token
        )

        try:
            dbx.users_get_current_account()
            logger.info("Dropbox client successfully connected")
        except Exception as e:
            raise ConnectionError(f"Dropbox authorization failed: {e}")

        return dbx
