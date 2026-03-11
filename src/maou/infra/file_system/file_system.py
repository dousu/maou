from pathlib import Path

import maou.interface.converter as converter
import maou.interface.learn as learn
import maou.interface.preprocess as preprocess
from maou.infra.file_system.path_utils import (
    collect_files as _collect_files,
)


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
        """指定パスからファイルを収集する．

        詳細は :func:`maou.infra.file_system.path_utils.collect_files` を参照．
        """
        return _collect_files(p, ext=ext)
