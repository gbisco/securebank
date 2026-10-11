from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union


class JsonLogger:
    """
    Writes structured JSON logs for SecureBank system monitoring.

    Each log is saved as a separate JSON file using the required format:
    securebank/logs/log_{timestamp}.json
    """

    def __init__(self, log_dir: Union[str, Path] = "securebank/logs"):
        """
        Initializes the JSON logger.

        Parameters:
            log_dir (str | Path): Directory where log files should be written.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def write_log(
        self,
        endpoint: str,
        request_data: Optional[dict[str, Any]] = None,
        response_data: Optional[dict[str, Any]] = None,
        status: str = "success",
        error: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Path:
        """
        Writes a structured JSON log file.

        Parameters:
            endpoint (str): Endpoint or workflow being logged.
            request_data (Optional[dict[str, Any]] = None): Incoming request payload.
            response_data (Optional[dict[str, Any]] = None): Response or result payload.
            status (str): Log status, usually "success" or "error".
            error (Optional[str] = None): Error message when status is "error".
            metadata (Optional[dict[str, Any]] = None): Optional extra monitoring information.

        Returns:
            Path: Path to the written log file.
        """
        timestamp = self._timestamp()

        log_record = {
            "timestamp": timestamp,
            "endpoint": endpoint,
            "status": status,
            "request": request_data or {},
            "response": response_data or {},
            "error": error,
            "metadata": metadata or {},
        }

        log_path = self.log_dir / f"log_{timestamp}.json"

        with log_path.open("w", encoding="utf-8") as file:
            json.dump(log_record, file, indent=2, default=str)

        return log_path

    def _timestamp(self) -> str:
        """
        Returns a filesystem-safe UTC timestamp.
        """
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")