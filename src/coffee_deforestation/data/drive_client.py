"""Google Drive polling and local file mirroring.

What: Downloads exported GeoTIFFs from Google Drive to the local filesystem.
Why: GEE exports land on Drive; we need them locally for rasterio processing.
Assumes: Google Drive API credentials are available (same service account as GEE,
or user's default credentials).
Produces: Local copies of exported files.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from coffee_deforestation.config import load_settings


def download_from_drive(filename: str, local_path: Path) -> bool:
    """Download a file from Google Drive by name.

    Uses the Google Drive API to find and download the file from the
    configured export folder.
    """
    settings = load_settings()
    folder_name = settings.google_drive_export_folder

    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload

        if settings.gee_service_account_key_path:
            creds = service_account.Credentials.from_service_account_file(
                settings.gee_service_account_key_path,
                scopes=["https://www.googleapis.com/auth/drive.readonly"],
            )
        else:
            # Fall back to default credentials
            import google.auth

            creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/drive.readonly"]
            )

        service = build("drive", "v3", credentials=creds)

        # Find the folder
        folder_query = (
            f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
        )
        folder_results = service.files().list(q=folder_query, spaces="drive").execute()
        folders = folder_results.get("files", [])

        if not folders:
            logger.error(f"Drive folder not found: {folder_name}")
            return False

        folder_id = folders[0]["id"]

        # Find the file in the folder
        file_query = f"name contains '{filename}' and '{folder_id}' in parents"
        file_results = service.files().list(q=file_query, spaces="drive").execute()
        files = file_results.get("files", [])

        if not files:
            logger.error(f"File not found in Drive: {filename}")
            return False

        file_id = files[0]["id"]
        request = service.files().get_media(fileId=file_id)

        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

        logger.info(f"Downloaded {filename} to {local_path}")
        return True

    except Exception as e:
        logger.error(f"Drive download failed for {filename}: {e}")
        return False
