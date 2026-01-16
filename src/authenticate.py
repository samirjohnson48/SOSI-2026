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


class SOSIAuthenticator:
    def __init__(
        self,
        google_service_account_creds_env_var: str,
        google_scopes: list,
    ):
        self.google_service_account_creds_env_var = google_service_account_creds_env_var
        self.google_scopes = google_scopes

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
            return service
        except HttpError as e:
            print("An error occurred connecting to the Google API")
            raise e
        except Exception as e:
            print("An unexpected error occurred: ")
            raise e
