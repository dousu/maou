import logging
from dataclasses import dataclass
from pathlib import Path

from maou._rust.maou_search import search as _rust_search
from maou.app.inference.eval import Evaluation
from maou.domain.board.shogi import Board

logger: logging.Logger = logging.getLogger(__name__)


class SearchRunner:
    """MCTS 探索エンジン (Rust maou_search) を駆動するユースケース．

    探索本体は Rust 側 (`maou._rust.maou_search.search`) が GIL を解放して
    実行する．本クラスはオプションの受け渡しと，評価値スコア
    (:class:`maou.app.inference.eval.Evaluation` — `maou evaluate` と同一の
    Ponanza 係数 600 の変換) を含む表示用整形を担う．
    """

    @dataclass(kw_only=True, frozen=True)
    class SearchOption:
        """探索オプション．

        Attributes:
            sfen: 基準局面の SFEN 文字列．
            moves: sfen から root 局面までの指し手列 (USI 形式)．
                途中局面は千日手判定の対局履歴として使われる．
            model_path: ONNX モデルのパス．None なら決定論的な mock 評価器
                (API 検証/開発用)．
            threads: 探索スレッド数．
            batch_size: 評価バッチサイズ．
            max_playouts: playout 数上限 (None で無制限)．
            time_ms: 時間上限ミリ秒 (None で無制限)．
            num_moves: 表示する上位候補手数．
            root_dfpn: ルート並行 dfpn 詰み探索を有効にするか．
            cuda: CUDA Execution Provider を使うか．
            tensorrt: TensorRT Execution Provider を使うか．
            trt_engine_cache_dir: TensorRT エンジンキャッシュ保存先．
            board_view: 盤面 ASCII 表示を含めるか．
        """

        sfen: str
        moves: tuple[str, ...] = ()
        model_path: Path | None = None
        threads: int = 1
        batch_size: int = 8
        max_playouts: int | None = None
        time_ms: int | None = None
        num_moves: int = 5
        root_dfpn: bool = False
        cuda: bool = False
        tensorrt: bool = False
        trt_engine_cache_dir: Path | None = None
        board_view: bool = True

    def run(self, config: SearchOption) -> dict[str, str]:
        """探索を実行し，表示用に整形した結果 dict を返す．

        Args:
            config: 探索オプション．

        Returns:
            ``Bestmove`` / ``Eval`` / ``WinRate`` / ``PV`` / ``Candidates`` /
            ``Stats`` (+ ``board_view`` 時 ``Board``) をキーとする dict．
            Eval は勝率からの逆シグモイド変換 (600 × logit) で，探索後の
            平均勝率に基づく (モデル 1 回の生出力である ``maou evaluate``
            とはスケールは同じだが同一値にはならない)．
        """
        result = _rust_search(
            config.sfen,
            moves=list(config.moves) if config.moves else None,
            model_path=(
                str(config.model_path)
                if config.model_path is not None
                else None
            ),
            threads=config.threads,
            batch_size=config.batch_size,
            max_playouts=config.max_playouts,
            time_ms=config.time_ms,
            root_dfpn=config.root_dfpn,
            use_cuda=config.cuda,
            use_tensorrt=config.tensorrt,
            trt_engine_cache_dir=(
                str(config.trt_engine_cache_dir)
                if config.trt_engine_cache_dir is not None
                else None
            ),
        )

        winrate = float(result.winrate)
        eval_score = float(
            Evaluation.get_eval_from_winrate(winrate)
        )

        # 候補手: best_move を先頭に，残りは訪問回数の降順に上位 num_moves 件
        # (勝敗確定時の best_move は訪問回数が少ないことがあり，訪問数順
        # だけではリストから漏れ得る)
        children = sorted(
            result.root_children,
            key=lambda c: int(c.visits),
            reverse=True,
        )
        best_first = [
            c for c in children if c.usi == result.best_move
        ]
        rest = [
            c for c in children if c.usi != result.best_move
        ]
        candidates: list[str] = []
        for child in (best_first + rest)[: config.num_moves]:
            if int(child.visits) == 0:
                # 未訪問の winrate 0 は「データなし」であり敗勢ではないため
                # 数値を表示しない
                candidates.append(
                    f"{child.usi} (visits=0, "
                    f"prior={float(child.prior):.4f})"
                )
                continue
            child_winrate = float(child.winrate)
            child_eval = float(
                Evaluation.get_eval_from_winrate(child_winrate)
            )
            candidates.append(
                f"{child.usi} (visits={child.visits}, "
                f"winrate={child_winrate:.4f}, eval={child_eval:.2f}, "
                f"prior={float(child.prior):.4f})"
            )

        output: dict[str, str] = {
            "Bestmove": (
                result.best_move
                if result.best_move is not None
                else "none"
            ),
            "Eval": f"{eval_score:.2f}",
            "WinRate": f"{winrate:.4f}",
            "PV": " ".join(result.pv),
            "Candidates": "\n".join(candidates),
            "Stats": (
                f"playouts={result.playouts} nps={result.nps:.0f} "
                f"elapsed_ms={result.elapsed_ms} max_depth={result.max_depth} "
                f"repetitions={result.repetitions} "
                f"proven_nodes={result.proven_nodes} stop={result.stop}"
            ),
        }
        if config.board_view:
            board = Board()
            board.set_sfen(config.sfen)
            for usi in config.moves:
                board.push_move(board.move_from_usi(usi))
            output["Board"] = board.to_pretty_board()
        return output
