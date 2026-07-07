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
# 29 手詰: first-mate は ~7K node で 31 手，最短 29 手の確定には ~396K node 必要．
# find_shortest のセマンティクス (予算不足時の unknown / first-mate 早期返却) の検証に使う．
MATE_29TE = "l2+P5/2k4+L1/2n1p2B1/p1pp1spN1/4Ps3/PlPP2P2/1P1Sb4/1KG2+p3/LN7 w R2GPrgsn4p 1"


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
        assert result.stop_reason == "solved"
        assert result.mate_len_found == 1
        assert result.shortest_confirmed is True
        assert result.root_pn == 0  # 詰み証明ゆえ pn=0
        assert result.elapsed_ms >= 0
        # checkmate 時は手順は moves 側; best_mate は空．
        assert result.best_mate == []
        # collect_progress 未指定 → progress は空 (アロケーション無し)．
        assert result.progress == []

    def test_no_mate(self) -> None:
        result = solve_tsume(NO_MATE, depth=31, nodes=100_000)
        assert result.status == "no_checkmate"
        assert result.moves == []
        assert bool(result) is False
        assert result.is_proven is False
        # 不詰は dn=0 で証明され，stop_reason は disproven．
        assert result.stop_reason == "disproven"
        assert result.root_dn == 0
        assert result.mate_len_found is None

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


class TestSolveTsumeFindShortestSemantics:
    """find_shortest の予算セマンティクス (呼び出し側の速度⇄最短性トレードオフ)．"""

    def test_find_shortest_unknown_when_budget_insufficient(
        self,
    ) -> None:
        """find_shortest=True で最小性を証明しきれない予算では unknown を返す．

        29 手詰は first-mate(31 手) は安く見つかるが，最短 29 手の確定には ~396K node
        を要する．予算 50K では最小性を証明できないため，**非最小の詰みでなく unknown**．
        """
        result = solve_tsume(
            MATE_29TE,
            depth=31,
            nodes=50_000,
            find_shortest=True,
        )
        assert result.status == "unknown"
        # 最短確定手順 (moves) は未確定ゆえ空．
        assert result.moves == []
        # unknown でも「詰み自体は検証済 (最短だけ未確定)」を伝える．
        # 予算追加で solved になる最有力ケース．
        assert result.stop_reason == "minimality_unconfirmed"
        assert result.mate_len_found is not None
        assert (
            result.mate_len_found % 2 == 1
        )  # 詰将棋の手数は奇数
        assert result.mate_len_found >= 29
        assert result.shortest_confirmed is False
        # 予算切れで最短だけ未確定でも，そこまでに見つけた検証済み詰み手順を best_mate に
        # 残す (大きなリソースを費した探索の成果を最低限保持する)．手数は mate_len_found と一致．
        assert len(result.best_mate) == result.mate_len_found
        assert all(
            isinstance(m, str) and m for m in result.best_mate
        )

    def test_find_shortest_checkmate_when_budget_sufficient(
        self,
    ) -> None:
        """十分な予算 (最小性を確定できる) では最短 29 手の checkmate を返す．"""
        result = solve_tsume(
            MATE_29TE,
            depth=31,
            nodes=500_000,
            find_shortest=True,
        )
        assert result.status == "checkmate"
        assert len(result.moves) == 29
        # 最短 29 手を確定できたので solved / shortest_confirmed．
        assert result.stop_reason == "solved"
        assert result.mate_len_found == 29
        assert result.shortest_confirmed is True
        # checkmate 時は手順は moves 側にあり，best_mate は空 (重複させない)．
        assert result.best_mate == []

    def test_find_first_returns_early_without_exhausting_budget(
        self,
    ) -> None:
        """find_shortest=False は最初の詰みを発見時点で返し，予算を使い切らない．

        29 手詰の first-mate は ~7K node で見つかる．予算 500K を与えても，
        発見時点で即返るため nodes_searched << 予算 となる (早いレスポンス重視)．
        """
        result = solve_tsume(
            MATE_29TE,
            depth=31,
            nodes=500_000,
            find_shortest=False,
        )
        assert result.status == "checkmate"
        # first-mate は 31 手 (最短の 29 手ではない)．
        assert len(result.moves) == 31
        # 予算 (500K) を使い切らず，発見時点 (~7K node) で返っている．
        assert result.nodes_searched < 100_000


class TestSolveTsumeValidation:
    """引数バリデーションが Python 例外 (ValueError) になること．"""

    def test_invalid_sfen_raises_valueerror(self) -> None:
        with pytest.raises(ValueError):
            solve_tsume(
                "not a valid sfen", depth=3, nodes=1_000
            )

    @pytest.mark.parametrize("bad_depth", [0, 2048, 3000])
    def test_out_of_range_depth_raises_valueerror(
        self, bad_depth: int
    ) -> None:
        """depth は 1..=2047．範囲外は PanicException ではなく ValueError を送出する．

        Rust 側 `DfPnSolver::with_timeout` は depth >= 2048 (PATH_CAPACITY) で
        panic するため，バインディングが事前に弾いて ValueError に変換する
        (上限 2047 = 長手数詰将棋対応)．
        """
        with pytest.raises(ValueError):
            solve_tsume(MATE_1TE, depth=bad_depth, nodes=1_000)

    def test_depth_upper_bound_ok(self) -> None:
        """depth=2047 (上限) は正常に受け付けられる．"""
        result = solve_tsume(
            MATE_1TE, depth=2047, nodes=100_000
        )
        assert result.status == "checkmate"


class TestSolveTsumeProgress:
    """collect_progress による進捗トラジェクトリ．

    Colab 等で native stderr のログが見えない環境でも，返り値経由で pn/dn の推移を
    純 Python で観測できることを保証する (fd 捕捉不要)．
    """

    def test_progress_empty_by_default(self) -> None:
        """collect_progress 未指定では progress は空 (性能・メモリに影響しない)．"""
        result = solve_tsume(MATE_29TE, depth=31, nodes=50_000)
        assert result.progress == []

    def test_progress_collected_when_enabled(self) -> None:
        """collect_progress=True で root 反復ごとのサンプルが記録される．"""
        result = solve_tsume(
            MATE_29TE,
            depth=31,
            nodes=50_000,
            find_shortest=True,
            collect_progress=True,
        )
        assert len(result.progress) > 0
        # 各サンプルは属性アクセスでき，nodes は単調非減少 (累積カウンタ)．
        prev_nodes = 0
        for s in result.progress:
            assert s.nodes >= prev_nodes
            assert s.pn >= 0
            assert s.dn >= 0
            assert s.elapsed_ms >= 0
            assert s.mate_len >= 1
            prev_nodes = s.nodes
        # 最終サンプルのノード数は総探索ノード数以下 (途中断面ゆえ)．
        assert (
            result.progress[-1].nodes <= result.nodes_searched
        )

    def test_progress_shows_mate_proof_convergence(
        self,
    ) -> None:
        """詰みが解けた探索では，どこかの反復で pn=0 (詰み証明) に到達する．"""
        result = solve_tsume(
            MATE_29TE,
            depth=31,
            nodes=500_000,
            find_shortest=True,
            collect_progress=True,
        )
        assert result.status == "checkmate"
        assert any(s.pn == 0 for s in result.progress)


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
