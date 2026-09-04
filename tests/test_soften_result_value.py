"""``scripts/soften_result_value.py`` (Arm 0 の教師生成) のテスト．

実データを使う試験は ``scratchpad/`` の測定物に依存するので，無ければ
``skip`` する (``scratchpad/`` は .gitignore 配下で CI には無い)．
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts")
)

from soften_result_value import (
    aggregate_positions,
    empirical_pools,
    select_targets,
    soften_empirical,
    soften_uniform,
)

INPERIOD_HCPE = Path("scratchpad/calib/inperiod_hcpe")
INPERIOD_PRE = Path("scratchpad/calib/inperiod_pre_nosv")
REFERENCE = Path("scratchpad/measure/sv_vs_outcome.feather")


class TestAggregatePositions:
    """局面単位の集約が ``pre-process`` と同じ値になること．"""

    def test_duplicates_are_averaged(self) -> None:
        agg = aggregate_positions(
            np.array([7, 7, 9], dtype=np.uint64),
            np.array([1.0, 0.0, 1.0]),
            np.array([50, 80, 60]),
        )
        row = agg.filter(pl.col("id") == 7)
        assert row["resultValue"][0] == pytest.approx(0.5)
        assert row["count"][0] == 2
        assert row["min_ply"][0] == 50
        assert row["max_ply"][0] == 80

    def test_single_occurrence_keeps_label(self) -> None:
        agg = aggregate_positions(
            np.array([9], dtype=np.uint64),
            np.array([1.0]),
            np.array([60]),
        )
        assert agg["resultValue"][0] == 1.0
        assert agg["count"][0] == 1
        assert agg["min_ply"][0] == agg["max_ply"][0] == 60

    def test_draw_stays_half(self) -> None:
        agg = aggregate_positions(
            np.array([1], dtype=np.uint64),
            np.array([0.5]),
            np.array([70]),
        )
        assert agg["resultValue"][0] == pytest.approx(0.5)


class TestSelectTargets:
    """帯の境界と，触ってはいけない行の除外．"""

    @staticmethod
    def _agg(
        rows: list[tuple[int, float, int, int]],
    ) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "id": pl.Series(
                    [r[0] for r in rows], dtype=pl.UInt64
                ),
                "resultValue": [r[1] for r in rows],
                "count": [1] * len(rows),
                "min_ply": [r[2] for r in rows],
                "max_ply": [r[3] for r in rows],
            }
        )

    def test_band_is_half_open(self) -> None:
        agg = self._agg(
            [
                (1, 1.0, 59, 59),
                (2, 1.0, 60, 60),
                (3, 1.0, 99, 99),
                (4, 1.0, 100, 100),
            ]
        )
        got = set(select_targets(agg, 60, 100)["id"].to_list())
        assert got == {2, 3}, "下限は含み，上限は含まない"

    def test_any_occurrence_in_band_qualifies(self) -> None:
        """帯の外と内にまたがる局面は対象 (search-values に合わせる)．"""
        agg = self._agg([(1, 1.0, 30, 70), (2, 1.0, 10, 20)])
        got = set(select_targets(agg, 60, 100)["id"].to_list())
        assert got == {1}

    def test_draws_and_aggregates_are_left_alone(self) -> None:
        agg = self._agg(
            [
                (1, 0.5, 70, 70),
                (2, 0.25, 70, 70),
                (3, 0.0, 70, 70),
                (4, 1.0, 70, 70),
            ]
        )
        got = set(select_targets(agg, 60, 100)["id"].to_list())
        assert got == {3, 4}, "既に統計の行と引き分けは触らない"

    def test_rejects_inverted_band(self) -> None:
        agg = self._agg([(1, 1.0, 70, 70)])
        with pytest.raises(ValueError, match="max-ply"):
            select_targets(agg, 100, 60)


class TestSoftenUniform:
    def test_symmetric_and_bounded(self) -> None:
        out = soften_uniform(
            np.array([0.0, 1.0, 1.0, 0.0]), 0.15
        )
        assert out.tolist() == pytest.approx(
            [0.15, 0.85, 0.85, 0.15]
        )
        assert out.dtype == np.float32

    def test_preserves_side(self) -> None:
        v = np.array([0.0, 1.0])
        out = soften_uniform(v, 0.3)
        assert (out[v == 1.0] > 0.5).all()
        assert (out[v == 0.0] < 0.5).all()

    @pytest.mark.parametrize("bad", [0.0, 0.5, 0.7, -0.1])
    def test_rejects_out_of_range_epsilon(
        self, bad: float
    ) -> None:
        with pytest.raises(ValueError, match="epsilon"):
            soften_uniform(np.array([0.0, 1.0]), bad)


class TestSoftenEmpirical:
    @staticmethod
    def _pools() -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(1)
        return (
            rng.uniform(0.0, 0.4, 5000).astype(np.float32),
            rng.uniform(0.6, 1.0, 5000).astype(np.float32),
        )

    def test_samples_from_the_matching_pool(self) -> None:
        lo, hi = self._pools()
        v = np.array([0.0] * 500 + [1.0] * 500)
        out = soften_empirical(
            v, lo, hi, np.random.default_rng(0)
        )
        assert (out[v == 0.0] <= 0.4).all(), (
            "元 0 は 0 側の分布から"
        )
        assert (out[v == 1.0] >= 0.6).all(), (
            "元 1 は 1 側の分布から"
        )

    def test_reproduces_reference_mean(self) -> None:
        lo, hi = self._pools()
        v = np.array([0.0] * 20000 + [1.0] * 20000)
        out = soften_empirical(
            v, lo, hi, np.random.default_rng(0)
        )
        assert out[v == 0.0].mean() == pytest.approx(
            lo.mean(), abs=0.01
        )
        assert out[v == 1.0].mean() == pytest.approx(
            hi.mean(), abs=0.01
        )

    def test_deterministic_for_a_seed(self) -> None:
        lo, hi = self._pools()
        v = np.array([0.0, 1.0] * 100)
        a = soften_empirical(
            v, lo, hi, np.random.default_rng(7)
        )
        b = soften_empirical(
            v, lo, hi, np.random.default_rng(7)
        )
        assert np.array_equal(a, b)

    def test_handles_single_sided_input(self) -> None:
        lo, hi = self._pools()
        out = soften_empirical(
            np.array([1.0, 1.0]),
            lo,
            hi,
            np.random.default_rng(0),
        )
        assert (out >= 0.6).all()


class TestEmpiricalPools:
    def test_requires_both_columns(self) -> None:
        with pytest.raises(ValueError, match="resultValue"):
            empirical_pools(
                pl.DataFrame({"searchWinRate": [0.1] * 200}),
                60,
                100,
            )

    def test_rejects_thin_reference(self) -> None:
        ref = pl.DataFrame(
            {
                "searchWinRate": [0.2] * 10,
                "resultValue": [0.0] * 10,
            }
        )
        with pytest.raises(
            ValueError, match="標本が少なすぎる"
        ):
            empirical_pools(ref, 60, 100)

    def test_filters_by_ply_when_present(self) -> None:
        n = 200
        ref = pl.DataFrame(
            {
                "searchWinRate": [0.2] * n
                + [0.8] * n
                + [0.05] * n
                + [0.95] * n,
                "resultValue": [0.0] * n
                + [1.0] * n
                + [0.0] * n
                + [1.0] * n,
                "ply": [70] * (2 * n) + [200] * (2 * n),
            }
        )
        lo, hi = empirical_pools(ref, 60, 100)
        assert len(lo) == len(hi) == n
        assert lo.max() == pytest.approx(0.2), (
            "帯外 (ply 200) を含めない"
        )


@pytest.mark.skipif(
    not INPERIOD_HCPE.exists() or not INPERIOD_PRE.exists(),
    reason="scratchpad の測定物が無い",
)
class TestAgainstRealPreprocessing:
    """集約が本物の ``maou pre-process`` 出力と一致すること (回帰)．

    この一致が崩れると，Arm 0 の教師が Arm 1 と別の行に当たることになり，
    対照が成立しなくなる．
    """

    def test_aggregate_matches_preprocess_output(self) -> None:
        from soften_result_value import load_positions

        ids, results, plies = load_positions(INPERIOD_HCPE)
        agg = aggregate_positions(ids, results, plies)
        real = pl.concat(
            [
                pl.scan_ipc(f, memory_map=False).select(
                    ["id", "resultValue"]
                )
                for f in sorted(
                    glob.glob(str(INPERIOD_PRE / "*.feather"))
                )
            ]
        ).collect()

        assert len(agg) == len(real), "局面数が一致すること"
        joined = agg.join(
            real, on="id", how="inner", suffix="_real"
        )
        assert len(joined) == len(real), "id 集合が一致すること"
        diff = np.abs(
            joined["resultValue"].to_numpy()
            - joined["resultValue_real"].to_numpy()
        )
        assert diff.max() < 1e-6, (
            f"resultValue が一致しない (最大 {diff.max():.2e})"
        )


@pytest.mark.skipif(
    not INPERIOD_HCPE.exists() or not REFERENCE.exists(),
    reason="scratchpad の測定物が無い",
)
class TestEndToEnd:
    """実データで最後まで通し，出力が pre-process に渡せる形であること．"""

    def test_produces_usable_search_value_file(
        self, tmp_path: Path
    ) -> None:
        from soften_result_value import load_positions

        ids, results, plies = load_positions(INPERIOD_HCPE)
        agg = aggregate_positions(ids, results, plies)
        targets = select_targets(agg, 60, 100)
        assert len(targets) > 1000

        lo, hi = empirical_pools(
            pl.read_ipc(REFERENCE, memory_map=False), 60, 100
        )
        original = targets["resultValue"].to_numpy()
        softened = soften_empirical(
            original, lo, hi, np.random.default_rng(0)
        )
        out = pl.DataFrame(
            {
                "id": targets["id"].cast(pl.UInt64),
                "searchWinRate": pl.Series(
                    softened, dtype=pl.Float32
                ),
            }
        )
        path = tmp_path / "arm0.feather"
        out.write_ipc(path, compression="lz4")

        back = pl.read_ipc(path, memory_map=False)
        assert back.schema["id"] == pl.UInt64
        assert back.schema["searchWinRate"] == pl.Float32
        assert back["id"].n_unique() == len(back), (
            "id が重複しない"
        )
        v = back["searchWinRate"].to_numpy()
        assert ((v >= 0.0) & (v <= 1.0)).all()
        assert not ((v == 0.0) | (v == 1.0)).all(), (
            "全部が厳密 0/1 では軟化になっていない"
        )

    def test_only_confident_rows_are_targeted(self) -> None:
        from soften_result_value import load_positions

        ids, results, plies = load_positions(INPERIOD_HCPE)
        agg = aggregate_positions(ids, results, plies)
        targets = select_targets(agg, 60, 100)
        v = targets["resultValue"].to_numpy()
        assert set(np.unique(v)) <= {0.0, 1.0}

    def test_targets_are_a_subset_of_the_corpus(self) -> None:
        from soften_result_value import load_positions

        ids, results, plies = load_positions(INPERIOD_HCPE)
        agg = aggregate_positions(ids, results, plies)
        targets = select_targets(agg, 60, 100)
        assert set(targets["id"].to_list()) <= set(
            agg["id"].to_list()
        )


class TestDebias:
    """``--debias`` が偏りだけを打ち消すこと．"""

    def test_matches_the_original_mean(self) -> None:
        from soften_result_value import debias

        original = np.array([0.0] * 500 + [1.0] * 500)
        softened = np.where(original == 1.0, 0.88, 0.18).astype(
            np.float32
        )
        assert softened.mean() > original.mean(), (
            "前提: 楽観側に偏っている"
        )

        out = debias(softened, original)
        assert out.mean() == pytest.approx(
            original.mean(), abs=1e-4
        )

    def test_preserves_spread(self) -> None:
        """平行移動なので分散は変えない (丸めが効かない範囲では)．"""
        from soften_result_value import debias

        rng = np.random.default_rng(0)
        original = np.array([0.0] * 1000 + [1.0] * 1000)
        softened = np.where(
            original == 1.0,
            rng.uniform(0.55, 0.8, 2000),
            rng.uniform(0.2, 0.45, 2000),
        ).astype(np.float32)
        out = debias(softened, original)
        assert out.std() == pytest.approx(
            softened.std(), rel=0.02
        )

    def test_stays_in_unit_interval(self) -> None:
        from soften_result_value import debias

        original = np.array([1.0] * 100)
        softened = np.full(100, 0.99, dtype=np.float32)
        out = debias(softened, original)
        assert ((out >= 0.0) & (out <= 1.0)).all()

    def test_is_a_noop_when_already_unbiased(self) -> None:
        from soften_result_value import debias

        original = np.array([0.0] * 100 + [1.0] * 100)
        softened = np.where(original == 1.0, 0.8, 0.2).astype(
            np.float32
        )
        out = debias(softened, original)
        assert np.allclose(out, softened, atol=1e-6)


class TestEmbeddedReference:
    """外部ファイル無しで ``--mode empirical`` が使えること．

    Colab などへスクリプト 1 枚だけ持っていく運用のため，測定済みの条件分布を
    分位点表として埋め込んである．表が壊れると対照実験の教師が別物になる．
    """

    def test_quantile_tables_are_well_formed(self) -> None:
        from soften_result_value import (
            EMBEDDED_QUANTILES_HI,
            EMBEDDED_QUANTILES_LO,
        )

        for q in (EMBEDDED_QUANTILES_LO, EMBEDDED_QUANTILES_HI):
            assert len(q) == 101, "0, 0.01, ..., 1 の 101 点"
            assert ((q >= 0.0) & (q <= 1.0)).all()
            assert (np.diff(q) >= -1e-6).all(), (
                "分位点は単調非減少"
            )

    def test_reproduces_the_measured_distribution(self) -> None:
        """埋め込み表が測定値を再現すること (回帰).

        実測 (floodgate 2025-03-02 / ply 60-99): 0 側 mean 0.2949 /
        1 側 mean 0.7772 / ほぼ確信 28.6%.
        """
        from soften_result_value import embedded_pools

        lo, hi = embedded_pools(
            200_000, np.random.default_rng(0)
        )
        assert lo.mean() == pytest.approx(0.2949, abs=0.005)
        assert hi.mean() == pytest.approx(0.7772, abs=0.005)
        both = np.concatenate([lo, hi])
        conf = ((both < 0.1) | (both > 0.9)).mean()
        assert conf == pytest.approx(0.286, abs=0.01)

    def test_is_deterministic_for_a_seed(self) -> None:
        from soften_result_value import embedded_pools

        a = embedded_pools(500, np.random.default_rng(3))
        b = embedded_pools(500, np.random.default_rng(3))
        assert np.array_equal(a[0], b[0])
        assert np.array_equal(a[1], b[1])

    def test_sides_are_separated(self) -> None:
        """0 側の分布は 1 側より低い方に寄っていること．"""
        from soften_result_value import embedded_pools

        lo, hi = embedded_pools(
            20_000, np.random.default_rng(0)
        )
        assert lo.mean() < 0.5 < hi.mean()

    def test_embedded_band_matches_the_documented_one(
        self,
    ) -> None:
        from soften_result_value import EMBEDDED_BAND

        assert EMBEDDED_BAND == (60, 100)
