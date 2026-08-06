"""探索による value 教師 (`maou utility search-values`) のテスト．

探索そのものは Rust 側なので，ここでは**選定とマージの意味**を固定する．
選定がラベルに依存しないこと，部分適用が成立することが要点である．
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from maou.app.pre_process.search_value import (
    SEARCH_VALUE_SCHEMA,
    _ply_of,
    apply_search_values,
    select_positions,
)


class TestPlyOf:
    """HCPE の ``id`` から手数を取り出す規約を固定する．"""

    def test_parses_trailing_ply(self) -> None:
        assert (
            _ply_of("wdoor+floodgate-a+b+20260303.csa.hcpe_73")
            == 73
        )

    def test_handles_dots_in_game_name(self) -> None:
        assert _ply_of("a.b.c.csa.hcpe_5") == 5

    def test_returns_minus_one_without_suffix(self) -> None:
        assert _ply_of("no-suffix") == -1

    def test_returns_minus_one_for_non_numeric(self) -> None:
        assert _ply_of("game.hcpe_x") == -1


def _df(ids: list[str]) -> pl.DataFrame:
    return pl.DataFrame({"id": ids})


class TestSelectPositions:
    """選定は手数と重複だけで決まり，ラベルを見ない．"""

    def test_filters_by_min_ply(self) -> None:
        df = _df(
            ["g.hcpe_10", "g.hcpe_59", "g.hcpe_60", "g.hcpe_61"]
        )
        hashes = np.array([1, 2, 3, 4], dtype=np.uint64)
        rows = select_positions(
            df, hashes, min_ply=60, max_positions=0, seed=0
        )
        assert rows.tolist() == [2, 3]

    def test_keeps_one_row_per_hash(self) -> None:
        # 同一局面は前処理で 1 行へ集約されるので探索も 1 回でよい
        df = _df(["g.hcpe_60", "h.hcpe_70", "i.hcpe_80"])
        hashes = np.array([7, 7, 9], dtype=np.uint64)
        rows = select_positions(
            df, hashes, min_ply=60, max_positions=0, seed=0
        )
        assert rows.tolist() == [0, 2]

    def test_skips_already_done(self) -> None:
        df = _df(["g.hcpe_60", "g.hcpe_70", "g.hcpe_80"])
        hashes = np.array([1, 2, 3], dtype=np.uint64)
        rows = select_positions(
            df,
            hashes,
            min_ply=60,
            max_positions=0,
            seed=0,
            already_done=np.array([2], dtype=np.uint64),
        )
        assert rows.tolist() == [0, 2]

    def test_cap_is_deterministic_for_a_seed(self) -> None:
        df = _df([f"g.hcpe_{60 + i}" for i in range(50)])
        hashes = np.arange(50, dtype=np.uint64)
        first = select_positions(
            df, hashes, min_ply=60, max_positions=10, seed=3
        )
        second = select_positions(
            df, hashes, min_ply=60, max_positions=10, seed=3
        )
        assert first.tolist() == second.tolist()
        assert len(first) == 10
        # 昇順で返る (入力順に読めるようにするため)
        assert first.tolist() == sorted(first.tolist())

    def test_returns_empty_when_nothing_matches(self) -> None:
        df = _df(["g.hcpe_1", "g.hcpe_2"])
        hashes = np.array([1, 2], dtype=np.uint64)
        rows = select_positions(
            df, hashes, min_ply=60, max_positions=0, seed=0
        )
        assert rows.tolist() == []


def _pre_df(
    ids: list[int], values: list[float]
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": pl.Series("id", ids, dtype=pl.UInt64),
            "resultValue": pl.Series(
                "resultValue", values, dtype=pl.Float32
            ),
        }
    )


def _values(
    ids: list[int], win_rates: list[float]
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": pl.Series("id", ids, dtype=pl.UInt64),
            "searchWinRate": pl.Series(
                "searchWinRate", win_rates, dtype=pl.Float32
            ),
            "playouts": pl.Series(
                "playouts", [800] * len(ids), dtype=pl.Int32
            ),
            "stop": pl.Series(
                "stop",
                ["playout_limit"] * len(ids),
                dtype=pl.String,
            ),
        },
        schema=SEARCH_VALUE_SCHEMA,
    )


class TestApplySearchValues:
    """部分適用が成立することを固定する．

    floodgate の全局面を探索するのは GPU 予算上ありえないので，
    **探索できた局面だけ差し替わり，残りは対局結果のまま**でなければ
    ならない．これが崩れると部分実行が使い物にならなくなる．
    """

    def test_replaces_only_covered_positions(self) -> None:
        df = _pre_df([1, 2, 3], [1.0, 0.0, 1.0])
        out, applied = apply_search_values(
            df, _values([2], [0.42])
        )
        assert applied == 1
        assert out["resultValue"].to_list() == pytest.approx(
            [1.0, 0.42, 1.0], abs=1e-6
        )

    def test_keeps_row_order_and_columns(self) -> None:
        df = _pre_df([10, 20], [0.0, 1.0])
        out, _ = apply_search_values(df, _values([20], [0.3]))
        assert out.columns == df.columns
        assert out["id"].to_list() == [10, 20]

    def test_result_value_stays_float32(self) -> None:
        df = _pre_df([1], [1.0])
        out, _ = apply_search_values(df, _values([1], [0.25]))
        assert out.schema["resultValue"] == pl.Float32

    def test_empty_values_is_a_noop(self) -> None:
        df = _pre_df([1, 2], [1.0, 0.0])
        out, applied = apply_search_values(
            df, pl.DataFrame(schema=SEARCH_VALUE_SCHEMA)
        )
        assert applied == 0
        assert out.equals(df)

    def test_unmatched_values_change_nothing(self) -> None:
        df = _pre_df([1, 2], [1.0, 0.0])
        out, applied = apply_search_values(
            df, _values([99], [0.5])
        )
        assert applied == 0
        assert out["resultValue"].to_list() == pytest.approx(
            [1.0, 0.0], abs=1e-6
        )
