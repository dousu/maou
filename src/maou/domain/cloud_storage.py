"""Cloud Storage Interface."""

import abc
from pathlib import Path


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
        extensions: list[str] | None = None,
    ) -> None:
        pass
