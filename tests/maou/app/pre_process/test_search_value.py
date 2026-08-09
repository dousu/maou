"""探索による value 教師 (`maou utility search-values`) のテスト．

探索そのものは Rust 側なので，ここでは**選定とマージの意味**を固定する．
選定がラベルに依存しないこと，部分適用が成立することが要点である．
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from maou.app.pre_process.hcpe_transform import DataSource
from maou.app.pre_process.search_value import (
    SEARCH_VALUE_REQUIRED_SCHEMA,
    SEARCH_VALUE_SCHEMA,
    SearchValueCollector,
    SearchValueOption,
    _merge,
    _ply_of,
    _with_current_schema,
    apply_search_values,
    load_search_values,
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
            "elapsedMs": pl.Series(
                "elapsedMs", [120] * len(ids), dtype=pl.Int32
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

    def test_duplicate_ids_do_not_add_rows(self) -> None:
        """重複 id で行数が増えないこと．

        左 join は右側に同じキーが 2 行あると左の行を複製する．
        これを許すと**前処理出力の行数が増えて学習データが静かに壊れる**．
        実際に修正前は 3 行が 4 行になっていた．
        """
        df = _pre_df([1, 2, 3], [1.0, 0.0, 1.0])
        out, applied = apply_search_values(
            df, _values([2, 2], [0.4, 0.6])
        )
        assert len(out) == 3
        assert applied == 1
        # 後勝ち (新しい探索を残す)
        assert out["resultValue"].to_list() == pytest.approx(
            [1.0, 0.6, 1.0], abs=1e-6
        )


class TestMerge:
    """`--resume` の蓄積で重複が生まれないことを固定する．"""

    def test_appends_new_rows(self) -> None:
        merged = _merge(
            _values([1], [0.1]), _values([2], [0.2])
        )
        assert sorted(merged["id"].to_list()) == [1, 2]

    def test_drops_duplicates_keeping_the_new_one(self) -> None:
        merged = _merge(
            _values([1], [0.1]), _values([1], [0.9])
        )
        assert len(merged) == 1
        assert merged[
            "searchWinRate"
        ].to_list() == pytest.approx([0.9], abs=1e-6)

    def test_empty_done_returns_fresh(self) -> None:
        fresh = _values([5], [0.5])
        assert _merge(
            pl.DataFrame(schema=SEARCH_VALUE_SCHEMA), fresh
        ).equals(fresh)


class TestOverwriteGuard:
    """既存の出力を黙って捨てないこと．

    出力は数日分の GPU 時間そのものなので，`--resume` も `--overwrite` も
    無いまま上書きさせない (修正前は 52 行が 5 行に上書きされた)．
    """

    def _option(
        self, out: Path, **kw: object
    ) -> SearchValueOption:
        base: dict[str, object] = {
            "input_path": out.parent,
            "output_path": out,
        }
        base.update(kw)
        return SearchValueOption(**base)  # type: ignore[arg-type]

    def test_raises_when_output_exists(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "sv.feather"
        _values([1], [0.5]).write_ipc(out)
        with pytest.raises(ValueError, match="already exists"):
            SearchValueCollector().collect(self._option(out))

    def test_resume_is_allowed(self, tmp_path: Path) -> None:
        out = tmp_path / "sv.feather"
        _values([1], [0.5]).write_ipc(out)
        # 対象 HCPE が無いので探索は 0 件だが，ガードは通る
        result = SearchValueCollector().collect(
            self._option(out, resume=True)
        )
        assert result["searched"] == "0"

    def test_overwrite_is_allowed(self, tmp_path: Path) -> None:
        out = tmp_path / "sv.feather"
        _values([1], [0.5]).write_ipc(out)
        result = SearchValueCollector().collect(
            self._option(out, overwrite=True)
        )
        assert result["searched"] == "0"

    def test_resume_and_overwrite_are_mutually_exclusive(
        self, tmp_path: Path
    ) -> None:
        """両方指定は暗黙に resume が勝つのでなくエラーにする．

        実測では両方を付けると `--overwrite` が黙って無視され，
        作り直したつもりの実行で古い値が残っていた．
        """
        out = tmp_path / "sv.feather"
        _values([1], [0.5]).write_ipc(out)
        with pytest.raises(
            ValueError, match="mutually exclusive"
        ):
            SearchValueCollector().collect(
                self._option(out, resume=True, overwrite=True)
            )

    def test_mutual_exclusion_applies_without_an_output(
        self, tmp_path: Path
    ) -> None:
        # 出力の有無に関わらず矛盾した指定は受け付けない
        with pytest.raises(
            ValueError, match="mutually exclusive"
        ):
            SearchValueCollector().collect(
                self._option(
                    tmp_path / "sv.feather",
                    resume=True,
                    overwrite=True,
                )
            )

    def test_no_guard_when_output_is_absent(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "sv.feather"
        result = SearchValueCollector().collect(
            self._option(out)
        )
        assert result["searched"] == "0"


class TestOlderOutputFormat:
    """0.82.0 より前の出力 (``elapsedMs`` 無し) を捨てさせない．

    既に数十万件を貯めている実行があるので，列が増えたら読めなくなる，
    では困る．欠けている列は null で補って `--resume` を続けられること．
    """

    def _old(self) -> pl.DataFrame:
        return _values([1, 2], [0.4, 0.6]).drop("elapsedMs")

    def test_backfills_missing_column(self) -> None:
        out = _with_current_schema(self._old())
        assert list(out.columns) == list(SEARCH_VALUE_SCHEMA)
        assert out["elapsedMs"].null_count() == 2

    def test_keeps_existing_values(self) -> None:
        out = _with_current_schema(self._old())
        assert out["id"].to_list() == [1, 2]
        assert out["searchWinRate"].to_list() == pytest.approx(
            [0.4, 0.6], abs=1e-6
        )

    def test_current_format_is_untouched(self) -> None:
        cur = _values([1], [0.5])
        assert _with_current_schema(cur).equals(cur)

    def test_merges_with_new_rows(self) -> None:
        merged = _merge(
            _with_current_schema(self._old()),
            _values([3], [0.7]),
        )
        assert sorted(merged["id"].to_list()) == [1, 2, 3]
        assert merged["elapsedMs"].null_count() == 2


class TestLoadSearchValues:
    """``--search-value-path`` の受け付ける形と，早期に落ちる条件を固定する．

    差し替え自体は HCPE 変換と集約が終わった後にしか走らない．
    入力の不備をそこまで運んでしまうと数時間の実行が丸ごと無駄になるので，
    **読み込みの時点で落ちる**ことがこのクラスの主題である．
    """

    def _write(self, path: Path, df: pl.DataFrame) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_ipc(path, compression="lz4")
        return path

    def test_reads_a_single_file(self, tmp_path: Path) -> None:
        self._write(
            tmp_path / "v.feather", _values([1, 2], [0.4, 0.6])
        )
        out = load_search_values(tmp_path / "v.feather")
        assert out["id"].to_list() == [1, 2]
        assert out["searchWinRate"].to_list() == pytest.approx(
            [0.4, 0.6], abs=1e-6
        )

    def test_projects_to_the_columns_pre_processing_uses(
        self, tmp_path: Path
    ) -> None:
        """診断列は落とす (union をスキーマ差で失敗させないため)．"""
        self._write(tmp_path / "v.feather", _values([1], [0.4]))
        out = load_search_values(tmp_path / "v.feather")
        assert list(out.columns) == list(
            SEARCH_VALUE_REQUIRED_SCHEMA
        )
        assert out.schema["id"] == pl.UInt64
        assert out.schema["searchWinRate"] == pl.Float32

    def test_directory_unions_every_feather(
        self, tmp_path: Path
    ) -> None:
        self._write(
            tmp_path / "a.feather", _values([1, 2], [0.1, 0.2])
        )
        self._write(tmp_path / "b.feather", _values([3], [0.3]))
        out = load_search_values(tmp_path)
        assert sorted(out["id"].to_list()) == [1, 2, 3]

    def test_directory_recurses(self, tmp_path: Path) -> None:
        self._write(tmp_path / "a.feather", _values([1], [0.1]))
        self._write(
            tmp_path / "nested" / "b.feather",
            _values([2], [0.2]),
        )
        assert sorted(
            load_search_values(tmp_path)["id"].to_list()
        ) == [1, 2]

    def test_directory_ignores_non_feather(
        self, tmp_path: Path
    ) -> None:
        self._write(tmp_path / "a.feather", _values([1], [0.1]))
        (tmp_path / "notes.txt").write_text("not a feather")
        assert load_search_values(tmp_path)["id"].to_list() == [
            1
        ]

    def test_unions_across_schema_versions(
        self, tmp_path: Path
    ) -> None:
        """``elapsedMs`` は 0.82.0 で追加された．

        新旧の出力が同じディレクトリに並ぶのは普通に起こる．診断列の
        構成差で union が失敗すると，過去の GPU 時間が使えなくなる．
        """
        old = _values([1], [0.1]).drop("elapsedMs")
        self._write(tmp_path / "old.feather", old)
        self._write(
            tmp_path / "new.feather", _values([2], [0.2])
        )
        assert sorted(
            load_search_values(tmp_path)["id"].to_list()
        ) == [1, 2]

    def test_directory_result_is_usable(
        self, tmp_path: Path
    ) -> None:
        """union した結果がそのまま差し替えに使えること．"""
        self._write(
            tmp_path / "a.feather", _values([1], [0.25])
        )
        self._write(
            tmp_path / "b.feather", _values([2], [0.75])
        )
        out, applied = apply_search_values(
            _pre_df([1, 2, 3], [1.0, 0.0, 1.0]),
            load_search_values(tmp_path),
        )
        assert applied == 2
        assert out["resultValue"].to_list() == pytest.approx(
            [0.25, 0.75, 1.0], abs=1e-6
        )

    def test_duplicate_ids_across_files_do_not_add_rows(
        self, tmp_path: Path
    ) -> None:
        """union は重複を生みうるが，行数不変は崩れてはならない．"""
        self._write(tmp_path / "a.feather", _values([2], [0.4]))
        self._write(tmp_path / "b.feather", _values([2], [0.6]))
        out, applied = apply_search_values(
            _pre_df([1, 2], [1.0, 0.0]),
            load_search_values(tmp_path),
        )
        assert len(out) == 2
        assert applied == 1
        # パス順で後勝ち
        assert out["resultValue"].to_list() == pytest.approx(
            [1.0, 0.6], abs=1e-6
        )

    def test_empty_directory_raises(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(
            ValueError, match="no .feather file"
        ):
            load_search_values(tmp_path / "empty")

    def test_missing_column_raises_naming_the_file(
        self, tmp_path: Path
    ) -> None:
        """HCPE ディレクトリを渡す取り違えがここで止まること．"""
        self._write(
            tmp_path / "ok.feather", _values([1], [0.1])
        )
        self._write(
            tmp_path / "hcpe.feather",
            pl.DataFrame(
                {"id": ["g.hcpe_1"], "hcp": [b"x" * 32]}
            ),
        )
        with pytest.raises(
            ValueError, match="searchWinRate"
        ) as excinfo:
            load_search_values(tmp_path)
        assert "hcpe.feather" in str(excinfo.value)

    def test_unusable_dtype_raises(
        self, tmp_path: Path
    ) -> None:
        self._write(
            tmp_path / "v.feather",
            pl.DataFrame(
                {
                    "id": pl.Series(
                        "id", ["1"], dtype=pl.String
                    ),
                    "searchWinRate": pl.Series(
                        "searchWinRate", [0.5], dtype=pl.Float32
                    ),
                }
            ),
        )
        with pytest.raises(ValueError, match="cannot be used"):
            load_search_values(tmp_path / "v.feather")

    def test_unreadable_file_raises(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "broken.feather").write_bytes(
            b"not arrow ipc"
        )
        with pytest.raises(
            ValueError, match="could not be read"
        ):
            load_search_values(tmp_path)

    def test_checks_every_file_before_reading_any(
        self, tmp_path: Path
    ) -> None:
        """検査は全ファイル分を先に済ませる．

        1 ファイル目を読んでから 2 ファイル目で落ちる作りだと，
        ファイル数が増えるほど失敗が遅れる．
        """
        for i in range(3):
            self._write(
                tmp_path / f"{i}_ok.feather",
                _values([i], [0.1]),
            )
        self._write(
            tmp_path / "9_bad.feather",
            pl.DataFrame({"id": [1]}),
        )
        with pytest.raises(ValueError, match="searchWinRate"):
            load_search_values(tmp_path)


class _NoDataSource(DataSource):
    """``__init__`` の検査だけを見るためのダミー．

    早期検証は ``transform`` より前に走るので，データソースは触られない．
    """

    def __len__(self) -> int:
        return 0

    def iter_batches(self) -> Iterator[tuple[str, np.ndarray]]:
        return iter(())


class TestEarlyValidation:
    """探索値の不備が ``transform`` より前に落ちること．

    回帰: 以前は差し替え直前まで遅延ロードしていたため，HCPE 変換と
    集約が全部終わってから落ちていた．パスの取り違え 1 つで数時間の実行が
    丸ごと無駄になる．検査は `PreProcess` の構築時点で済んでいなければ
    ならない．
    """

    def _preprocess(self, path: Path) -> None:
        from maou.app.pre_process.hcpe_transform import (
            PreProcess,
        )

        PreProcess(
            datasource=_NoDataSource(),
            feature_store=None,
            search_value_path=path,
        )

    def test_rejects_wrong_schema_at_construction(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "wrong.feather").write_bytes(b"")
        pl.DataFrame({"id": ["g.hcpe_1"]}).write_ipc(
            tmp_path / "wrong.feather", compression="lz4"
        )
        with pytest.raises(ValueError, match="searchWinRate"):
            self._preprocess(tmp_path)

    def test_rejects_empty_directory_at_construction(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(
            ValueError, match="no .feather file"
        ):
            self._preprocess(tmp_path)

    def test_accepts_a_directory_of_values(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "a.feather").write_bytes(b"")
        _values([1], [0.5]).write_ipc(
            tmp_path / "a.feather", compression="lz4"
        )
        self._preprocess(tmp_path)
