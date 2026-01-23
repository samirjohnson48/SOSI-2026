"""
Authenticator class to extract and load necessary data used in the
reporting of SOSI 2026.
Authenticates Google Service Account for extraction from Google Drive
and Dropbox for loading.
"""

import os
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build, Resource
from googleapiclient.errors import HttpError
import logging

logger = logging.getLogger(__name__)


class SOSIAuthenticator:
    def __init__(
        self,
        google_service_account_creds_env_var: str,
        client_id_env_var: str,
        client_secret_env_var: str,
        refresh_token_env_var: str,
    ):
        self.google_service_account_creds_env_var = google_service_account_creds_env_var
        self.client_id_env_var = client_id_env_var
        self.client_secret_env_var = client_secret_env_var
        self.refresh_token_env_var = refresh_token_env_var

    def _get_google_service_credentials(self, scopes: list):
        try:
            creds_path = os.getenv(self.google_service_account_creds_env_var)

            if not creds_path:
                raise KeyError(
                    f"Set environment variable {self.google_service_account_creds_env_var} to point to path of Google Service Account credentials."
                )

            credentials = service_account.Credentials.from_service_account_file(
                creds_path, scopes=scopes
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

    # TODO: Better error handling with credentials and environment variables
    def _get_google_oauth_credentials(self) -> Credentials:
        creds = {
            "refresh_token": os.getenv(self.refresh_token_env_var),
            "client_id": os.getenv(self.client_id_env_var),
            "client_secret": os.getenv(self.client_secret_env_var),
        }
        missing = [k for k, v in creds.items() if v is None]
        if missing:
            raise ValueError(
                f"Environment variables not set for Google OAuth credentials: {', '.join(missing)}"
            )

        return Credentials(
            token=None,
            refresh_token=creds["refresh_token"],
            client_id=creds["client_id"],
            client_secret=creds["client_secret"],
            token_uri="https://oauth2.googleapis.com/token",
        )

    # TODO: Better naming convention with service_name = oauth
    def get_google_service(
        self,
        service_name: str,
        scopes: list | None = None,
        creds_type: str = "service_account",
    ) -> Resource:
        """
        Returns a Google Resource object to interact with the API
        used in extracting stock assessment files from Google Drive
        """
        if service_name not in ["drive", "sheets"]:
            raise ValueError(
                "Specify either 'drive' or 'sheets' for the type of service account."
            )

        versions = {"drive": "v3", "sheets": "v4"}

        match creds_type.lower():
            case "service_account" | "service account":
                if scopes is not None:
                    creds = self._get_google_service_credentials(scopes)
                else:
                    raise ValueError("Must specify scopes for service account object.")
            case "oauth" | "oauth2":
                creds = self._get_google_oauth_credentials()
            case _:
                raise ValueError(
                    f"Unknown credentials type {creds_type}. Please specify 'service_account' or 'oauth'"
                )
        try:
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
