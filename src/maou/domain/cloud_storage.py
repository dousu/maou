"""Cloud Storage Interface."""

import abc
from pathlib import Path
from typing import Optional


class CloudStorage(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def upload_from_local(
        self, *, local_path: Path, cloud_path: str
    ) -> None:
        pass

    @abc.abstractmethod
    def upload_folder_from_local(
        self,
        *,
        local_folder: Path,
        cloud_folder: str,
        extensions: Optional[list[str]] = None,
    ) -> None:
        pass
