from pathlib import Path

import maou.interface.converter as converter
import maou.interface.preprocess as preprocess
from maou.infra.file_system.path_utils import (
    collect_files as _collect_files,
)


# NOTE: maou.interface.learn の FileSystem ABC は継承しない．
# learn インターフェースは module-level で torch を import するため，
# 継承すると torch 無しの base install で hcpe-convert 等が壊れる．
# learn 系の呼び出しは collect_files を静的に使うのみで，
# learn.FileSystem を型として要求する消費者は存在しない．
class FileSystem(
    converter.FileSystem,
    preprocess.FileSystem,
):
    @staticmethod
    def get_text(filename: str, encoding: str = "utf-8") -> str:
        with open(filename, encoding=encoding) as f:
            return f.read()

    @staticmethod
    def collect_files(
        p: Path, ext: str | None = None
    ) -> list[Path]:
        """指定パスからファイルを収集する．

        詳細は :func:`maou.infra.file_system.path_utils.collect_files` を参照．
        """
        return _collect_files(p, ext=ext)
