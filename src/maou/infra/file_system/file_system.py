from pathlib import Path
from typing import Optional

from maou.interface import converter, learn, preprocess


class FileSystem(converter.FileSystem, learn.FileSystem, preprocess.FileSystem):
    @staticmethod
    def get_text(file_name: str, encoding: str = "utf-8") -> str:
        with open(file_name, encoding=encoding) as f:
            return f.read()

    @staticmethod
    def collect_files(p: Path, ext: Optional[str] = None) -> list[Path]:
        if p.is_file():
            return [p]
        elif p.is_dir():
            return [
                f
                for f in p.glob("**/*")
                if f.is_file()
                and (ext is None or ext is not None and ext in f.suffixes)
            ]
        else:
            raise ValueError(f"Path `{p}` is neither a file nor a directory.")
