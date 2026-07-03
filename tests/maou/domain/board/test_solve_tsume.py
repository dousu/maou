"""`maou._rust.maou_shogi.solve_tsume` の Python バインディング境界テスト．

Rust 側 (rust/maou_shogi/src/dfpn/tests.rs) が探索アルゴリズムの正しさを網羅する
ため，ここでは **Python バインディング固有の契約**のみを検証する:

- 正常系の詰み/不詰/結果オブジェクトのセマンティクス
- 引数バリデーション (不正 SFEN・depth 範囲) が Python 例外になること
- GIL 解放下での並行呼び出し安全性
"""

from __future__ import annotations

import threading

import pytest

from maou._rust.maou_shogi import solve_tsume

# 後手玉 1一，先手金 2三，先手持ち駒: 金．G*1b (または G*2b) の 1 手詰．
MATE_1TE = "8k/9/7G1/9/9/9/9/9/9 b G 1"
# 玉のみ (攻め方に王手手段なし) → 不詰．
NO_MATE = "4k4/9/9/9/9/9/9/9/4K4 b - 1"


class TestSolveTsumeBasic:
    """正常系の結果セマンティクス．"""

    def test_mate_1te(self) -> None:
        result = solve_tsume(MATE_1TE, depth=3, nodes=100_000)
        assert result.status == "checkmate"
        assert len(result.moves) == 1
        assert result.moves[0] in ("G*1b", "G*2b")
        assert result.nodes_searched > 0
        assert bool(result) is True
        assert result.is_proven is True

    def test_no_mate(self) -> None:
        result = solve_tsume(NO_MATE, depth=31, nodes=100_000)
        assert result.status == "no_checkmate"
        assert result.moves == []
        assert bool(result) is False
        assert result.is_proven is False

    def test_defaults_are_optional(self) -> None:
        """depth/nodes 等を省略してもデフォルトで解ける (シグネチャの後方互換)．"""
        result = solve_tsume(MATE_1TE)
        assert result.status == "checkmate"

    def test_find_shortest_toggle(self) -> None:
        """find_shortest=False でも詰みは返る (最短保証なし)．"""
        result = solve_tsume(
            MATE_1TE,
            depth=3,
            nodes=100_000,
            find_shortest=False,
        )
        assert result.status == "checkmate"
        assert len(result.moves) == 1


class TestSolveTsumeValidation:
    """引数バリデーションが Python 例外 (ValueError) になること．"""

    def test_invalid_sfen_raises_valueerror(self) -> None:
        with pytest.raises(ValueError):
            solve_tsume(
                "not a valid sfen", depth=3, nodes=1_000
            )

    @pytest.mark.parametrize("bad_depth", [0, 48, 100])
    def test_out_of_range_depth_raises_valueerror(
        self, bad_depth: int
    ) -> None:
        """depth は 1..=47．範囲外は PanicException ではなく ValueError を送出する．

        Rust 側 `DfPnSolver::with_timeout` は depth >= 48 (PATH_CAPACITY) で
        panic するため，バインディングが事前に弾いて ValueError に変換する．
        """
        with pytest.raises(ValueError):
            solve_tsume(MATE_1TE, depth=bad_depth, nodes=1_000)

    def test_depth_upper_bound_ok(self) -> None:
        """depth=47 (上限) は正常に受け付けられる．"""
        result = solve_tsume(MATE_1TE, depth=47, nodes=100_000)
        assert result.status == "checkmate"


class TestSolveTsumeConcurrency:
    """GIL 解放下 (py.detach) での並行呼び出し安全性．

    ソルバーは呼び出しごとにローカル状態を構築し共有可変状態を持たないため，
    複数スレッドからの同時 solve が競合なく完了することを確認する．
    """

    def test_parallel_calls(self) -> None:
        results: list[str] = []
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                r = solve_tsume(
                    MATE_1TE, depth=3, nodes=100_000
                )
                results.append(r.status)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=worker) for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert results == ["checkmate"] * 4
