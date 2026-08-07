"""探索による value 教師生成のインターフェース層．

`maou utility search-values` と app 層 (`SearchValueCollector`) の間で
オプションを組み立てる．
"""

from __future__ import annotations

import logging
from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)


def collect_search_values(
    *,
    input_path: Path,
    output_path: Path,
    model_path: Path | None = None,
    min_ply: int = 60,
    max_positions: int = 0,
    seed: int = 0,
    max_playouts: int = 800,
    time_ms: int | None = None,
    threads: int = 1,
    batch_size: int = 8,
    root_dfpn: bool = True,
    leaf_mate: bool = True,
    cuda: bool = False,
    tensorrt: bool = False,
    trt_engine_cache_dir: Path | None = None,
    resume: bool = False,
    overwrite: bool = False,
    flush_interval: int = 500,
) -> dict[str, str]:
    """HCPE の局面を探索して value 教師を作る．

    Args:
        input_path: HCPE (`.feather`) のディレクトリまたはファイル．
        output_path: 出力する feather のパス．
        model_path: ONNX モデルのパス．None なら mock 評価器．
        min_ply: この手数以上の局面のみ対象にする．
        max_positions: 対象局面数の上限 (0 で無制限)．
        seed: 上限超過時の標本抽出の乱数種．
        max_playouts: 1 局面あたりの playout 上限．
        time_ms: 1 局面あたりの時間上限 (ミリ秒)．
        threads: 探索スレッド数．
        batch_size: 評価バッチサイズ．
        root_dfpn: ルート並行 dfpn 詰み探索を行うか．
        leaf_mate: 葉の短手詰み探索を行うか．
        cuda: CUDA Execution Provider を使うか．
        tensorrt: TensorRT Execution Provider を使うか．
        trt_engine_cache_dir: TensorRT エンジンキャッシュ保存先．
        resume: 既存の出力にある局面を飛ばして続きから探索するか．
        overwrite: 既存の出力を破棄して作り直すか．
        flush_interval: 途中結果を書き出す局面数の間隔．

    Returns:
        表示用の要約 dict．

    Raises:
        ValueError: `input_path` に `.feather` が無い場合．
    """
    from maou.app.pre_process.search_value import (
        SearchValueCollector,
        SearchValueOption,
    )

    if input_path.is_dir() and not any(
        input_path.glob("**/*.feather")
    ):
        raise ValueError(f"no .feather under {input_path}")

    option = SearchValueOption(
        input_path=input_path,
        output_path=output_path,
        model_path=model_path,
        min_ply=min_ply,
        max_positions=max_positions,
        seed=seed,
        max_playouts=max_playouts,
        time_ms=time_ms,
        threads=threads,
        batch_size=batch_size,
        root_dfpn=root_dfpn,
        leaf_mate=leaf_mate,
        cuda=cuda,
        tensorrt=tensorrt,
        trt_engine_cache_dir=trt_engine_cache_dir,
        resume=resume,
        overwrite=overwrite,
        flush_interval=flush_interval,
    )
    return SearchValueCollector().collect(option)
