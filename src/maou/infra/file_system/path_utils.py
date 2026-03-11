"""ファイルパス操作のユーティリティ関数．

torch等の重い依存を持たない軽量モジュール．
``visualize`` や ``build-game-graph`` など，
torch不要なコマンドから安全にインポートできる．
"""

from pathlib import Path


def collect_files(
    p: Path, ext: str | None = None
) -> list[Path]:
    """指定パスからファイルを収集する．

    Args:
        p: ファイルまたはディレクトリのパス
        ext: フィルタする拡張子(例: ".feather")．
            ``None`` の場合は全ファイルを返す．

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
        ]
    else:
        msg = (
            f"パスがファイルでもディレクトリでもありません: {p}"
        )
        raise ValueError(msg)
