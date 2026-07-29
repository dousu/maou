import logging
from pathlib import Path

from maou._rust.maou_search import evaluate as _rust_evaluate
from maou.domain.board.shogi import Board

logger: logging.Logger = logging.getLogger(__name__)


def infer(
    *,
    model_path: Path | None = None,
    num_moves: int,
    sfen: str,
    cuda: bool = False,
    tensorrt: bool = False,
    trt_cache_dir: Path | None = None,
) -> str:
    """局面を 0 手読みで評価して結果を文字列で返す．

    推論本体は Rust (``maou._rust.maou_search.evaluate``) が GIL を解放して
    実行する．本関数はオプションの受け渡しと表示用整形だけを担う．

    Args:
        model_path: ONNXモデルファイルパス．未指定ならmock評価器．
        num_moves: 上位候補手数．
        sfen: SFEN文字列．
        cuda: CUDA Execution Provider を使うか．
        tensorrt: TensorRT Execution Provider を使うか．
        trt_cache_dir: TensorRTエンジンキャッシュ保存先．

    Returns:
        推論結果のフォーマット済み文字列．

    Raises:
        ValueError: num_movesが負の場合．
    """
    if num_moves < 0:
        raise ValueError(
            f"num_moves must be non-negative: {num_moves}"
        )

    result = _rust_evaluate(
        sfen,
        model_path=str(model_path)
        if model_path is not None
        else None,
        num_moves=num_moves,
        use_cuda=cuda,
        use_tensorrt=tensorrt,
        trt_engine_cache_dir=str(trt_cache_dir)
        if trt_cache_dir is not None
        else None,
    )

    if result.moves:
        policy = ", ".join(
            f"{m.usi} ({m.prior:.4f})" for m in result.moves
        )
    elif result.legal_move_count == 0:
        policy = "(no legal moves)"
    else:
        # num_moves=0 が指定された: 合法手はあるが表示しない
        policy = f"(suppressed; {result.legal_move_count} legal moves)"

    # 盤面表示はドメインのBoardに委ねる (推論経路には関与しない)
    board = Board()
    board.set_sfen(sfen)

    return f"""

Policy: {policy}
Eval: {result.eval_score:.2f}
WinRate: {result.winrate:.4f}
{board.to_pretty_board()}"""
