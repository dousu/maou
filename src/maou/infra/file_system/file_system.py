from pathlib import Path

import maou.interface.converter as converter
import maou.interface.learn as learn
import maou.interface.preprocess as preprocess


class FileSystem(
    converter.FileSystem,
    learn.FileSystem,
    preprocess.FileSystem,
):
    @staticmethod
    def get_text(
        file_name: str, encoding: str = "utf-8"
    ) -> str:
        with open(file_name, encoding=encoding) as f:
            return f.read()

    @staticmethod
    def collect_files(
        p: Path, ext: str | None = None
    ) -> list[Path]:
        if p.is_file():
            if ext is not None and ext not in p.suffixes:
                msg = (
                    f"ファイルは {ext} 形式で"
                    f"なければなりません: {p}"
                )
                raise ValueError(msg)
            return [p]
        elif p.is_dir():
            return [
                f
                for f in p.glob("**/*")
                if f.is_file()
                and (
                    ext is None
                    or ext is not None
                    and ext in f.suffixes
                )
            ]
        else:
            raise ValueError(
                f"Path `{p}` is neither a file nor a directory."
            )
