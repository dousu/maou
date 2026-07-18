"""ファイルパス操作のユーティリティ関数．

torch等の重い依存を持たない軽量モジュール．
``visualize`` や ``build-game-graph`` など，
torch不要なコマンドから安全にインポートできる．
"""

from pathlib import Path

# クラウドストレージ (gcsfuse / gsutil 等) がダウンロード中・中断で残す一時
# ファイルのサフィックス．これらは不完全なことがあり，データファイルとして
# 読むと壊れる (例: `foo.feather_.gstmp` を Arrow IPC として読むと
# ``OSError: failed to fill whole buffer``)．ディレクトリ走査時に除外する．
_TEMP_ARTIFACT_SUFFIXES: tuple[str, ...] = (
    ".gstmp",  # gcsfuse ダウンロード一時ファイル
    ".tmp",  # 汎用一時ファイル
    ".partial",  # 部分ダウンロード
    ".crc",  # gsutil / hadoop チェックサム副産物
)


def _is_temp_artifact(f: Path) -> bool:
    """クラウドストレージのダウンロード一時ファイル (不完全な可能性) か．

    ``foo.feather_.gstmp`` のような一時ファイルはデータとして読むと壊れるため，
    ディレクトリ収集から除外する．拡張子 (ファイル名末尾) で判定する．
    """
    return f.name.endswith(_TEMP_ARTIFACT_SUFFIXES)


def collect_files(
    p: Path, ext: str | None = None
) -> list[Path]:
    """指定パスからファイルを収集する．

    ディレクトリ走査時は，クラウドストレージのダウンロード一時ファイル
    (``.gstmp`` など，[`_TEMP_ARTIFACT_SUFFIXES`] 参照) を除外する
    (不完全なファイルをデータとして読むと壊れるため)．

    Args:
        p: ファイルまたはディレクトリのパス
        ext: フィルタする拡張子(例: ".feather")．
            ``None`` の場合は(一時ファイルを除く)全ファイルを返す．

    Returns:
        マッチしたファイルパスのリスト

    Raises:
        ValueError: パスがファイルでもディレクトリでもない場合，
            または拡張子が一致しないファイルが指定された場合
    """
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
            and (ext is None or ext in f.suffixes)
            and not _is_temp_artifact(f)
        ]
    else:
        msg = (
            f"パスがファイルでもディレクトリでもありません: {p}"
        )
        raise ValueError(msg)
