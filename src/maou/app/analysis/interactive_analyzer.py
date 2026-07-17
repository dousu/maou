"""GUI 常駐の対話解析ユースケース (app 層)．

`maou analyze-gui` の「この局面を解析」「全局面解析」を担う．評価器
(:class:`maou._rust.maou_search.SearchEngine`) はサーバープロセスで
1 個だけ遅延構築して使い回す (docs/design/game-analysis/gui.md §11)．
解析結果は analyze-game の ``positions[i]`` と同一スキーマの dict
(:func:`maou.app.analysis.game_analyzer.position_record`) で返し，
CLI と GUI でレポートの相互運用を保つ (同 §9)．
"""

import logging
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from maou.app.analysis.analysis_session import (
    GameDocument,
    VariationTree,
    path_moves_usi,
)
from maou.app.analysis.game_analyzer import (
    FixedPlayoutsAllocator,
    FixedTimeAllocator,
    build_search_engine,
    position_record,
    summarize_positions,
)

logger: logging.Logger = logging.getLogger(__name__)

# 予算未指定時のデフォルト (1 局面あたり，docs/design/game-analysis/gui.md §3)
DEFAULT_TIME_MS = 1000


@dataclass(frozen=True)
class EngineSettings:
    """GUI 常駐エンジンの構築・探索設定．

    :class:`maou.app.analysis.game_analyzer.GameAnalyzer.AnalyzeOption`
    のエンジン系オプションと同じ意味・デフォルトを持つ．

    Attributes:
        model_path: ONNX モデルのパス．None なら決定論的な mock 評価器
            (開発検証専用．GUI に明示される)．
        threads: 探索スレッド数．
        batch_size: 評価バッチサイズ．
        num_candidates: 記録する候補手数．
        root_dfpn: ルート並行 dfpn 詰み探索を有効にするか．
        root_dfpn_nodes: ルート dfpn のノード予算．
        root_dfpn_depth: ルート dfpn の探索深さ上限 (最大 2047)．
        leaf_mate: MCTS の葉の短手詰み探索 (専用スレッド) を行うか．
        leaf_mate_nodes: leaf-mate 1 回あたりのノード予算．
        leaf_mate_threads: leaf-mate 専用スレッド数．
        cuda: CUDA Execution Provider を使うか．
        tensorrt: TensorRT Execution Provider を使うか．
        trt_engine_cache_dir: TensorRT エンジンキャッシュ保存先．
    """

    model_path: Path | None = None
    threads: int = 1
    batch_size: int = 8
    num_candidates: int = 5
    root_dfpn: bool = True
    root_dfpn_nodes: int = 2_000_000
    root_dfpn_depth: int = 2047
    leaf_mate: bool = True
    leaf_mate_nodes: int = 50
    leaf_mate_threads: int = 1
    cuda: bool = False
    tensorrt: bool = False
    trt_engine_cache_dir: Path | None = None

    def describe(self) -> dict[str, Any]:
        """レポートの ``engine`` セクション (analyze-game 互換) を返す．"""
        return {
            "model_path": (
                str(self.model_path)
                if self.model_path is not None
                else None
            ),
            "threads": self.threads,
            "batch_size": self.batch_size,
            "cuda": self.cuda,
            "tensorrt": self.tensorrt,
            "root_dfpn": self.root_dfpn,
            "root_dfpn_nodes": self.root_dfpn_nodes,
            "root_dfpn_depth": self.root_dfpn_depth,
            "leaf_mate": self.leaf_mate,
            "leaf_mate_nodes": self.leaf_mate_nodes,
            "leaf_mate_threads": self.leaf_mate_threads,
        }


def _normalize_budget(
    time_ms: int | None, max_playouts: int | None
) -> tuple[int | None, int | None]:
    """予算を正規化する (両方 None ならデフォルト時間予算)．"""
    if time_ms is None and max_playouts is None:
        return DEFAULT_TIME_MS, None
    return time_ms, max_playouts


class InteractiveAnalyzer:
    """GUI 常駐の対話解析エンジン．

    評価器は初回の解析要求時に 1 回だけ構築する (モデルロードを起動時
    から解析時まで遅延)．呼び出し側 (Gradio イベント) は
    ``concurrency_limit=1`` で直列化する前提であり，本クラス自体は
    スレッドセーフではない．
    """

    def __init__(self, settings: EngineSettings) -> None:
        """設定を保持する (エンジンはまだ構築しない)．

        Args:
            settings: エンジンの構築・探索設定．
        """
        self._settings = settings
        self._engine: Any | None = None

    @property
    def settings(self) -> EngineSettings:
        """エンジン設定．"""
        return self._settings

    @property
    def is_mock(self) -> bool:
        """mock 評価器 (モデル未指定，開発検証専用) かどうか．"""
        return self._settings.model_path is None

    def _ensure_engine(self) -> Any:
        """SearchEngine を遅延構築して返す．"""
        if self._engine is None:
            logger.info(
                "Building SearchEngine (model=%s)",
                self._settings.model_path,
            )
            self._engine = build_search_engine(
                model_path=self._settings.model_path,
                threads=self._settings.threads,
                batch_size=self._settings.batch_size,
                cuda=self._settings.cuda,
                tensorrt=self._settings.tensorrt,
                trt_engine_cache_dir=(
                    self._settings.trt_engine_cache_dir
                ),
            )
        return self._engine

    def _search(
        self,
        root_sfen: str,
        moves: list[str],
        time_ms: int | None,
        max_playouts: int | None,
    ) -> Any:
        """初期局面 + USI 経路で 1 局面探索を実行する．"""
        engine = self._ensure_engine()
        s = self._settings
        return engine.search(
            root_sfen,
            moves=moves or None,
            max_playouts=max_playouts,
            time_ms=time_ms,
            root_dfpn=s.root_dfpn,
            root_dfpn_nodes=s.root_dfpn_nodes,
            root_dfpn_depth=s.root_dfpn_depth,
            leaf_mate=s.leaf_mate,
            leaf_mate_nodes=s.leaf_mate_nodes,
            leaf_mate_threads=s.leaf_mate_threads,
        )

    def analyze_position(
        self,
        document: GameDocument,
        tree: VariationTree,
        node_id: int | None = None,
        *,
        time_ms: int | None = None,
        max_playouts: int | None = None,
    ) -> dict[str, Any]:
        """分岐木の 1 ノードの局面を解析する．

        エンジンには「初期局面 SFEN + root からの USI 経路」を渡す
        (千日手履歴を正しく効かせるため — GameAnalyzer と同じ規約)．

        Args:
            document: 棋譜の GameDocument．
            tree: 分岐木．
            node_id: 対象ノード (省略時は現在ノード)．
            time_ms: 時間予算 (ミリ秒)．
            max_playouts: playout 数予算．両方 None なら
                :data:`DEFAULT_TIME_MS`．

        Returns:
            analyze-game の ``positions[i]`` スキーマの解析記録．
            本譜ノードで次の手が既知なら実戦手比較
            (``played_move`` / ``match`` / ``winrate_loss``) を含む．

        Raises:
            ValueError: 探索の失敗 (不正局面等)．
        """
        node = tree.nodes[
            tree.current_id if node_id is None else node_id
        ]
        time_ms, max_playouts = _normalize_budget(
            time_ms, max_playouts
        )
        moves = path_moves_usi(tree, node.node_id)
        result = self._search(
            document.snapshots[0].sfen,
            moves,
            time_ms,
            max_playouts,
        )
        ply = node.snapshot.ply
        played: str | None = None
        record_time_s: int | None = None
        record_score: int | None = None
        record_comment: str | None = None
        if node.is_mainline and ply < document.n_moves:
            played = document.moves_usi[ply]
            if ply < len(document.times):
                record_time_s = document.times[ply]
            if ply < len(document.scores):
                record_score = document.scores[ply]
            if ply < len(document.comments):
                record_comment = document.comments[ply] or None
        return position_record(
            ply=ply + 1,
            side=node.snapshot.turn,
            sfen=node.snapshot.sfen,
            played_usi=played,
            result=result,
            num_candidates=self._settings.num_candidates,
            record_time_s=record_time_s,
            record_score=record_score,
            record_comment=record_comment,
        )

    def analyze_mainline(
        self,
        document: GameDocument,
        *,
        time_ms: int | None = None,
        max_playouts: int | None = None,
        cancel: threading.Event | None = None,
    ) -> Iterator[tuple[int, int, dict[str, Any]]]:
        """本譜の全局面を順に解析するジェネレータ．

        1 局面解析するごとに ``(i, n_moves, 解析記録)`` を yield する
        (i は 0 始まりの局面番号)．``cancel`` がセットされた時点で
        以降の局面を解析せずに終了する (実行中の 1 局面は完了を待つ)．

        Args:
            document: 棋譜の GameDocument．
            time_ms: 1 局面あたりの時間予算 (ミリ秒)．
            max_playouts: 1 局面あたりの playout 数予算．
            cancel: 協調キャンセル用のイベント．

        Yields:
            ``(局面番号, 総局面数, 解析記録)``．
        """
        time_ms, max_playouts = _normalize_budget(
            time_ms, max_playouts
        )
        root_sfen = document.snapshots[0].sfen
        n = document.n_moves
        for i in range(n):
            if cancel is not None and cancel.is_set():
                logger.info(
                    "Mainline analysis cancelled at %d/%d", i, n
                )
                return
            snapshot = document.snapshots[i]
            result = self._search(
                root_sfen,
                document.moves_usi[:i],
                time_ms,
                max_playouts,
            )
            record = position_record(
                ply=i + 1,
                side=snapshot.turn,
                sfen=snapshot.sfen,
                played_usi=document.moves_usi[i],
                result=result,
                num_candidates=self._settings.num_candidates,
                record_time_s=(
                    document.times[i]
                    if i < len(document.times)
                    else None
                ),
                record_score=(
                    document.scores[i]
                    if i < len(document.scores)
                    else None
                ),
                record_comment=(
                    document.comments[i] or None
                    if i < len(document.comments)
                    else None
                ),
            )
            yield i, n, record

    def build_report(
        self,
        document: GameDocument,
        positions: list[dict[str, Any]],
        *,
        source_name: str,
        time_ms: int | None = None,
        max_playouts: int | None = None,
    ) -> dict[str, Any]:
        """全局面解析の結果から analyze-game 互換のレポートを組み立てる．

        Args:
            document: 棋譜の GameDocument．
            positions: :meth:`analyze_mainline` が返した全解析記録
                (本譜の手数と同数であること)．
            source_name: レポートの ``input.path`` に記録する棋譜名
                (GUI アップロードではファイル名)．
            time_ms: 使用した時間予算 (ミリ秒)．
            max_playouts: 使用した playout 数予算．

        Returns:
            analyze-game の JSON 出力と同一スキーマの dict．

        Raises:
            ValueError: positions の件数が本譜の手数と一致しない場合．
        """
        if len(positions) != document.n_moves:
            raise ValueError(
                f"解析記録の件数 {len(positions)} が本譜の手数 "
                f"{document.n_moves} と一致しません"
            )
        time_ms, max_playouts = _normalize_budget(
            time_ms, max_playouts
        )
        allocator = (
            FixedPlayoutsAllocator(playouts=max_playouts)
            if max_playouts is not None
            else FixedTimeAllocator(
                time_ms=time_ms or DEFAULT_TIME_MS
            )
        )
        budget = allocator.describe()
        budget["per_position"] = {
            "max_playouts": max_playouts,
            "time_ms": time_ms,
        }
        return {
            "input": {
                "path": source_name,
                "format": document.input_format,
                "names": list(document.names),
                "ratings": list(document.ratings),
                "win": document.win,
                "endgame": document.endgame,
                "n_moves": document.n_moves,
            },
            "engine": self._settings.describe(),
            "budget": budget,
            "positions": positions,
            "summary": summarize_positions(positions),
        }
