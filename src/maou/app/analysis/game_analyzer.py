"""棋譜 1 局の自動解析ユースケース．

CSA / KIF の 1 局を受け取り，各指し手の直前局面を 1 局面探索エンジン
(`maou._rust.maou_search.SearchEngine`) で解析して per-position 記録と
サマリを返す．対局全体の時間配分 (:class:`BudgetAllocator`) は本モジュール
の責務であり，探索エンジンは与えられた予算内で 1 局面を探索するのみ
(docs/design/game-analysis/index.md)．
"""

import abc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from maou._rust.maou_search import SearchEngine
from maou._rust.maou_shogi import parse_csa_str, parse_kif_str
from maou.app.inference.eval import Evaluation
from maou.domain.board.shogi import Board, Turn, move_to_usi

logger: logging.Logger = logging.getLogger(__name__)


def decode_kifu_bytes(data: bytes) -> str:
    """棋譜ファイルの bytes を UTF-8 先行で文字列にデコードする．

    cp932 はほぼ任意のバイト列を受理してしまうため，先に UTF-8 を厳格に
    試し，失敗した場合のみ cp932 にフォールバックする．

    Args:
        data: 棋譜ファイルの生バイト列．

    Returns:
        デコード済み文字列．

    Raises:
        UnicodeDecodeError: UTF-8 でも cp932 でも解釈できない場合．
    """
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp932")


@dataclass(frozen=True)
class PositionBudget:
    """1 局面分の探索予算 (playout 数と時間の少なくとも一方)．

    Attributes:
        max_playouts: playout 数上限 (None で制限なし)．
        time_ms: 時間上限ミリ秒 (None で制限なし)．
    """

    max_playouts: int | None
    time_ms: int | None


class BudgetAllocator(metaclass=abc.ABCMeta):
    """対局全体の予算を局面ごとの探索予算に配分する戦略．

    探索エンジンは配分済みの :class:`PositionBudget` を受け取るのみで，
    時間配分の判断はこの階層に閉じる．将来の傾斜配分 (難局面・終盤重視)
    は本クラスの実装追加のみで導入できる．
    """

    @abc.abstractmethod
    def allocate(
        self, n_positions: int
    ) -> list[PositionBudget]:
        """n_positions 局面分の予算リストを返す．"""

    @abc.abstractmethod
    def describe(self) -> dict[str, Any]:
        """JSON 出力用の予算メタデータ (mode + パラメータ) を返す．"""


@dataclass(frozen=True)
class FixedTimeAllocator(BudgetAllocator):
    """全局面に同一の時間予算を割り当てる．

    Attributes:
        time_ms: 1 局面あたりの時間 (ミリ秒)．
    """

    time_ms: int

    def allocate(
        self, n_positions: int
    ) -> list[PositionBudget]:
        """n_positions 局面分の予算リストを返す．"""
        budget = PositionBudget(
            max_playouts=None, time_ms=self.time_ms
        )
        return [budget] * n_positions

    def describe(self) -> dict[str, Any]:
        """JSON 出力用の予算メタデータを返す．"""
        return {"mode": "fixed_time", "time_ms": self.time_ms}


@dataclass(frozen=True)
class EqualDivisionAllocator(BudgetAllocator):
    """対局全体の時間を局面数で等分して割り当てる．

    1 局面あたり ``max(total_time_ms // n, 1)`` ミリ秒 (床関数．端数は
    切り捨て = 全体上限を超えない側に倒す)．

    Attributes:
        total_time_ms: 対局全体の時間 (ミリ秒)．
    """

    total_time_ms: int

    def allocate(
        self, n_positions: int
    ) -> list[PositionBudget]:
        """n_positions 局面分の予算リストを返す．"""
        if n_positions <= 0:
            return []
        per_position = max(self.total_time_ms // n_positions, 1)
        budget = PositionBudget(
            max_playouts=None, time_ms=per_position
        )
        return [budget] * n_positions

    def describe(self) -> dict[str, Any]:
        """JSON 出力用の予算メタデータを返す．"""
        return {
            "mode": "equal_division",
            "total_time_ms": self.total_time_ms,
        }


@dataclass(frozen=True)
class FixedPlayoutsAllocator(BudgetAllocator):
    """全局面に同一の playout 数予算を割り当てる．

    Attributes:
        playouts: 1 局面あたりの playout 数．
    """

    playouts: int

    def allocate(
        self, n_positions: int
    ) -> list[PositionBudget]:
        """n_positions 局面分の予算リストを返す．"""
        budget = PositionBudget(
            max_playouts=self.playouts, time_ms=None
        )
        return [budget] * n_positions

    def describe(self) -> dict[str, Any]:
        """JSON 出力用の予算メタデータを返す．"""
        return {
            "mode": "fixed_playouts",
            "playouts": self.playouts,
        }


def _turn_char(turn: Turn) -> str:
    """手番を SFEN の手番文字 ("b"/"w") にする．"""
    return "b" if turn == Turn.BLACK else "w"


class GameAnalyzer:
    """棋譜 1 局を 1 手ずつ 1 局面探索で解析するユースケース．

    評価器は :class:`SearchEngine` として解析開始時に 1 回だけ構築し，
    全局面の探索で使い回す (ONNX モデルのロードは 1 回のみ)．
    """

    @dataclass(kw_only=True, frozen=True)
    class AnalyzeOption:
        """解析オプション．

        Attributes:
            input_path: 棋譜ファイルのパス．
            input_format: 棋譜形式 ("csa" / "kif")．
            allocator: 局面ごとの探索予算を決める配分戦略．
            model_path: ONNX モデルのパス．None なら決定論的な mock 評価器
                (API 検証/開発用)．
            threads: 探索スレッド数．
            batch_size: 評価バッチサイズ．
            num_candidates: JSON に記録する候補手数．
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

        input_path: Path
        input_format: str
        allocator: BudgetAllocator
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

    def analyze(self, option: AnalyzeOption) -> dict[str, Any]:
        """棋譜 1 局を解析して JSON 化可能な結果 dict を返す．

        Args:
            option: 解析オプション．

        Returns:
            ``input`` / ``engine`` / ``budget`` / ``positions`` / ``summary``
            をキーとする dict (スキーマは docs/design/game-analysis/index.md §7)．

        Raises:
            ValueError: 棋譜のパース失敗，複数局の CSA，指し手ゼロの棋譜，
                または棋譜中の不正手．
        """
        content = decode_kifu_bytes(
            option.input_path.read_bytes()
        )
        record = self._parse_record(
            content, option.input_format
        )
        moves: list[int] = list(record.moves)
        if not moves:
            raise ValueError(
                f"棋譜に指し手がありません: {option.input_path}"
            )
        usi_moves = [move_to_usi(m) for m in moves]
        budgets = option.allocator.allocate(len(moves))

        engine = SearchEngine(
            model_path=(
                str(option.model_path)
                if option.model_path is not None
                else None
            ),
            threads=option.threads,
            batch_size=option.batch_size,
            use_cuda=option.cuda,
            use_tensorrt=option.tensorrt,
            trt_engine_cache_dir=(
                str(option.trt_engine_cache_dir)
                if option.trt_engine_cache_dir is not None
                else None
            ),
        )

        record_times: list[int] = list(record.times)
        record_scores: list[int] = list(record.scores)
        record_comments: list[str] = list(record.comments)

        board = Board()
        board.set_sfen(record.sfen)
        positions: list[dict[str, Any]] = []
        for i, (move, played_usi, budget) in enumerate(
            tqdm(
                zip(moves, usi_moves, budgets),
                total=len(moves),
                desc="analyze",
                unit="pos",
            )
        ):
            ply = i + 1
            side = _turn_char(board.get_turn())
            sfen = board.get_sfen()
            try:
                result = engine.search(
                    record.sfen,
                    moves=usi_moves[:i] if i > 0 else None,
                    max_playouts=budget.max_playouts,
                    time_ms=budget.time_ms,
                    root_dfpn=option.root_dfpn,
                    root_dfpn_nodes=option.root_dfpn_nodes,
                    root_dfpn_depth=option.root_dfpn_depth,
                    leaf_mate=option.leaf_mate,
                    leaf_mate_nodes=option.leaf_mate_nodes,
                    leaf_mate_threads=option.leaf_mate_threads,
                )
            except ValueError as e:
                raise ValueError(
                    f"{option.input_path} の {ply} 手目 (直前局面 {sfen}) "
                    f"の探索に失敗: {e}"
                ) from e
            positions.append(
                self._position_record(
                    ply=ply,
                    side=side,
                    sfen=sfen,
                    played_usi=played_usi,
                    result=result,
                    num_candidates=option.num_candidates,
                    record_time_s=(
                        record_times[i]
                        if i < len(record_times)
                        else None
                    ),
                    record_score=(
                        record_scores[i]
                        if i < len(record_scores)
                        else None
                    ),
                    record_comment=(
                        record_comments[i] or None
                        if i < len(record_comments)
                        else None
                    ),
                )
            )
            try:
                board.push_move(move)
            except ValueError as e:
                raise ValueError(
                    f"{option.input_path} の {ply} 手目 {played_usi} を "
                    f"適用できません: {e}"
                ) from e

        return {
            "input": {
                "path": str(option.input_path),
                "format": option.input_format,
                "names": list(record.names),
                "ratings": list(record.ratings),
                "win": record.win,
                "endgame": record.endgame,
                "n_moves": len(moves),
            },
            "engine": {
                "model_path": (
                    str(option.model_path)
                    if option.model_path is not None
                    else None
                ),
                "threads": option.threads,
                "batch_size": option.batch_size,
                "cuda": option.cuda,
                "tensorrt": option.tensorrt,
                "root_dfpn": option.root_dfpn,
                "root_dfpn_nodes": option.root_dfpn_nodes,
                "root_dfpn_depth": option.root_dfpn_depth,
                "leaf_mate": option.leaf_mate,
                "leaf_mate_nodes": option.leaf_mate_nodes,
                "leaf_mate_threads": option.leaf_mate_threads,
            },
            "budget": self._budget_meta(
                option.allocator, budgets
            ),
            "positions": positions,
            "summary": self._summary(positions),
        }

    def _parse_record(
        self, content: str, input_format: str
    ) -> Any:
        """棋譜文字列をパースして GameRecord を返す．

        CSA はファイル内に複数局を含み得るが，本コマンドは 1 局のみ対応
        (複数局は ValueError)．
        """
        if input_format == "csa":
            records = parse_csa_str(content)
            if len(records) != 1:
                raise ValueError(
                    f"CSA ファイルに {len(records)} 局が含まれています．"
                    "analyze-game は 1 局のみ対応です "
                    "(複数局は 1 局ずつのファイルに分割してください)"
                )
            return records[0]
        if input_format == "kif":
            return parse_kif_str(content)
        raise ValueError(
            f"未対応の棋譜形式です: {input_format} (csa / kif のみ)"
        )

    def _budget_meta(
        self,
        allocator: BudgetAllocator,
        budgets: list[PositionBudget],
    ) -> dict[str, Any]:
        """JSON 出力用の予算セクション (mode + 代表 per-position 値) を作る．

        現状の配分戦略は全局面同一のため，先頭要素を代表値として載せる．
        """
        meta = allocator.describe()
        if budgets:
            meta["per_position"] = {
                "max_playouts": budgets[0].max_playouts,
                "time_ms": budgets[0].time_ms,
            }
        return meta

    def _position_record(
        self,
        *,
        ply: int,
        side: str,
        sfen: str,
        played_usi: str,
        result: Any,
        num_candidates: int,
        record_time_s: int | None,
        record_score: int | None,
        record_comment: str | None,
    ) -> dict[str, Any]:
        """1 局面分の解析記録 (JSON 化可能な dict) を作る．

        ``played_move_winrate`` / ``winrate_loss`` は同一局面の
        root_children 統計から取る (未訪問の手は winrate が「データなし」
        のため null)．``winrate_loss`` は best との差を 0 で下限クランプ
        する (訪問数基準の最終手選択では played の Q が best を僅かに
        上回り得るが，その場合も「損失なし」と扱う)．
        """
        best_move = result.best_move
        children = list(result.root_children)
        played_child = next(
            (c for c in children if c.usi == played_usi), None
        )
        played_winrate = (
            float(played_child.winrate)
            if played_child is not None
            and int(played_child.visits) > 0
            else None
        )
        winrate = float(result.winrate)
        best_child = (
            next(
                (c for c in children if c.usi == best_move),
                None,
            )
            if best_move is not None
            else None
        )
        best_winrate = (
            float(best_child.winrate)
            if best_child is not None
            and int(best_child.visits) > 0
            else winrate
        )
        match = (
            best_move is not None and played_usi == best_move
        )
        winrate_loss: float | None
        if match:
            winrate_loss = 0.0
        elif played_winrate is not None:
            winrate_loss = max(
                best_winrate - played_winrate, 0.0
            )
        else:
            winrate_loss = None

        # 候補手: best を先頭に，残りは訪問回数の降順 (SearchRunner と同じ整列)
        children_sorted = sorted(
            children, key=lambda c: int(c.visits), reverse=True
        )
        best_first = [
            c for c in children_sorted if c.usi == best_move
        ]
        rest = [
            c for c in children_sorted if c.usi != best_move
        ]
        candidates: list[dict[str, Any]] = []
        for child in (best_first + rest)[:num_candidates]:
            candidates.append(
                {
                    "usi": child.usi,
                    "visits": int(child.visits),
                    "winrate": (
                        float(child.winrate)
                        if int(child.visits) > 0
                        else None
                    ),
                    "prior": float(child.prior),
                    "proven": (
                        float(child.proven)
                        if child.proven is not None
                        else None
                    ),
                }
            )

        return {
            "ply": ply,
            "side_to_move": side,
            "sfen": sfen,
            "played_move": played_usi,
            "best_move": best_move,
            "match": match,
            "winrate": winrate,
            "eval_cp": float(
                Evaluation.get_eval_from_winrate(winrate)
            ),
            "played_move_winrate": played_winrate,
            "winrate_loss": winrate_loss,
            "pv": list(result.pv),
            "candidates": candidates,
            "mate_found": result.stop == "root_proven",
            "playouts": int(result.playouts),
            "elapsed_ms": int(result.elapsed_ms),
            "stop": result.stop,
            "record_time_s": record_time_s,
            "record_score": record_score,
            "record_comment": record_comment,
        }

    def _summary(
        self, positions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """per-position 記録から対局サマリを作る．

        worst_moves は winrate_loss の大きい順に上位 5 手 (null は除外)．
        """
        sides = {"b": "black", "w": "white"}
        match_rate: dict[str, float | None] = {}
        mean_winrate_loss: dict[str, float | None] = {}
        for side, label in sides.items():
            side_positions = [
                p
                for p in positions
                if p["side_to_move"] == side
            ]
            match_rate[label] = (
                sum(1 for p in side_positions if p["match"])
                / len(side_positions)
                if side_positions
                else None
            )
            losses = [
                p["winrate_loss"]
                for p in side_positions
                if p["winrate_loss"] is not None
            ]
            mean_winrate_loss[label] = (
                sum(losses) / len(losses) if losses else None
            )
        worst = sorted(
            (
                p
                for p in positions
                if p["winrate_loss"] is not None
            ),
            key=lambda p: p["winrate_loss"],
            reverse=True,
        )[:5]
        return {
            "match_rate": match_rate,
            "mean_winrate_loss": mean_winrate_loss,
            "worst_moves": [
                {
                    "ply": p["ply"],
                    "side": p["side_to_move"],
                    "played": p["played_move"],
                    "best": p["best_move"],
                    "winrate_loss": p["winrate_loss"],
                }
                for p in worst
            ],
            "mates_found": [
                {"ply": p["ply"], "side": p["side_to_move"]}
                for p in positions
                if p["mate_found"]
            ],
            "total_elapsed_ms": sum(
                p["elapsed_ms"] for p in positions
            ),
            "total_playouts": sum(
                p["playouts"] for p in positions
            ),
        }
