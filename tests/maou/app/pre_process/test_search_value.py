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
    NODE_CAPACITY_MARGIN,
    PENDING_PATTERN,
    SEARCH_VALUE_REQUIRED_SCHEMA,
    SEARCH_VALUE_SCHEMA,
    SHARD_PATTERN,
    SearchValueCollector,
    SearchValueOption,
    _merge,
    _next_index,
    _node_capacity,
    _ordered_value_paths,
    _ply_of,
    _with_current_schema,
    apply_search_values,
    load_search_values,
    select_positions,
    validate_search_value_source,
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
            "warmupMs": pl.Series(
                "warmupMs", [7] * len(ids), dtype=pl.Int32
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

    def _seeded(self, tmp_path: Path) -> Path:
        """シャードが 1 枚ある出力ディレクトリを作る．"""
        out = tmp_path / "sv"
        out.mkdir()
        _values([1], [0.5]).write_ipc(
            out / "part_00000001.feather"
        )
        return out

    def test_raises_when_output_exists(
        self, tmp_path: Path
    ) -> None:
        out = self._seeded(tmp_path)
        with pytest.raises(
            ValueError,
            match="already holds 1 search value file",
        ):
            SearchValueCollector().collect(self._option(out))

    def test_resume_is_allowed(self, tmp_path: Path) -> None:
        out = self._seeded(tmp_path)
        # 対象 HCPE が無いので探索は 0 件だが，ガードは通る
        result = SearchValueCollector().collect(
            self._option(out, resume=True)
        )
        assert result["searched"] == "0"

    def test_overwrite_is_allowed(self, tmp_path: Path) -> None:
        out = self._seeded(tmp_path)
        result = SearchValueCollector().collect(
            self._option(out, overwrite=True)
        )
        assert result["searched"] == "0"
        assert not out.exists(), (
            "--overwrite はシャードを残してはいけない "
            "(残骸を次の --resume が拾う)"
        )

    def test_resume_and_overwrite_are_mutually_exclusive(
        self, tmp_path: Path
    ) -> None:
        """両方指定は暗黙に resume が勝つのでなくエラーにする．

        実測では両方を付けると `--overwrite` が黙って無視され，
        作り直したつもりの実行で古い値が残っていた．
        """
        out = self._seeded(tmp_path)
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


class TestNodeCapacity:
    """ノードプール容量を playout 予算から決める規則を固定する．

    ノードは「未展開の子へ降りた playout」1 回につき 1 個しか確保されない
    ので，予算が必要数の上界になる．**上回っていれば GC は走らず探索は
    不変**という前提でこの縮小が正当化されるため，予算を下回る既定値を
    返してはならない．
    """

    def _option(self, **kwargs: object) -> SearchValueOption:
        return SearchValueOption(
            input_path=Path("in"),
            output_path=Path("out"),
            **kwargs,  # type: ignore[arg-type]
        )

    def test_derives_from_the_playout_budget(self) -> None:
        assert (
            _node_capacity(self._option(max_playouts=800))
            == 2 * 800 + NODE_CAPACITY_MARGIN
        )

    def test_stays_above_the_playout_budget(self) -> None:
        """予算を下回ると GC が走り，木が刈られて値が比較できなくなる．"""
        for budget in (1, 800, 100_000):
            assert (
                _node_capacity(
                    self._option(max_playouts=budget)
                )
                > budget
            )

    def test_explicit_value_wins(self) -> None:
        assert (
            _node_capacity(
                self._option(max_playouts=800, node_capacity=32)
            )
            == 32
        )

    def test_is_far_below_the_rust_default(self) -> None:
        """既定 2^20 のままだと 1 局面あたり約 50MB を確保し直す．"""
        assert (
            _node_capacity(self._option(max_playouts=800))
            < (1 << 20) // 100
        )


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

    def test_backfills_warmup_ms(self) -> None:
        """``warmupMs`` は 0.97.0 で足した．

        既に走っている実行の出力 (この列が無い) を `--resume` で
        続けられること．
        """
        old = _values([1, 2], [0.4, 0.6]).drop("warmupMs")
        out = _with_current_schema(old)
        assert list(out.columns) == list(SEARCH_VALUE_SCHEMA)
        assert out["warmupMs"].null_count() == 2
        assert out["elapsedMs"].to_list() == [120, 120]


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
        out = load_search_values([tmp_path / "v.feather"])
        assert out["id"].to_list() == [1, 2]
        assert out["searchWinRate"].to_list() == pytest.approx(
            [0.4, 0.6], abs=1e-6
        )

    def test_projects_to_the_columns_pre_processing_uses(
        self, tmp_path: Path
    ) -> None:
        """診断列は落とす (union をスキーマ差で失敗させないため)．"""
        self._write(tmp_path / "v.feather", _values([1], [0.4]))
        out = load_search_values([tmp_path / "v.feather"])
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
        out = load_search_values([tmp_path])
        assert sorted(out["id"].to_list()) == [1, 2, 3]

    def test_directory_recurses(self, tmp_path: Path) -> None:
        self._write(tmp_path / "a.feather", _values([1], [0.1]))
        self._write(
            tmp_path / "nested" / "b.feather",
            _values([2], [0.2]),
        )
        assert sorted(
            load_search_values([tmp_path])["id"].to_list()
        ) == [1, 2]

    def test_directory_ignores_non_feather(
        self, tmp_path: Path
    ) -> None:
        self._write(tmp_path / "a.feather", _values([1], [0.1]))
        (tmp_path / "notes.txt").write_text("not a feather")
        assert load_search_values([tmp_path])[
            "id"
        ].to_list() == [1]

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
            load_search_values([tmp_path])["id"].to_list()
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
            load_search_values([tmp_path]),
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
            load_search_values([tmp_path]),
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
            ValueError, match="no .feather or .arrow file"
        ):
            load_search_values([tmp_path / "empty"])

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
            load_search_values([tmp_path])
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
            load_search_values([tmp_path / "v.feather"])

    def test_unreadable_file_raises(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "broken.feather").write_bytes(
            b"not arrow ipc"
        )
        with pytest.raises(
            ValueError, match="could not be read"
        ):
            load_search_values([tmp_path])

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
            load_search_values([tmp_path])


class _NoDataSource(DataSource):
    """``__init__`` の検査だけを見るためのダミー．

    早期検証は ``transform`` より前に走るので，データソースは触られない．

    ``total_pages`` は「触られないから」という理由で長らく欠けていた．
    ``DataSource`` が ``abc.ABC`` になったので，欠けたままだと構築時に
    ``TypeError`` になる — それが ABC を入れた目的そのものなので，
    ダミーであっても抽象メソッドは全部埋める．2026-08-14 に
    ``iter_batches_df`` も abstract になったので (backlog 行 N6-2)，
    ここにも足してある．
    """

    def __len__(self) -> int:
        return 0

    def iter_batches(self) -> Iterator[tuple[str, np.ndarray]]:
        return iter(())

    def iter_batches_df(
        self,
    ) -> Iterator[tuple[str, pl.DataFrame]]:
        return iter(())

    def total_pages(self) -> int:
        return 0


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
            search_value_paths=[path],
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
            ValueError, match="no .feather or .arrow file"
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


class TestDedupHappensOnce:
    """重複排除は load で 1 回だけ済ませる．

    回帰: ``apply_search_values`` は出力チャンクごとに呼ばれるので，
    そちらに重複解決を任せると union 全体の排除と同一の警告を
    チャンク数だけ繰り返す．resume を重ねた出力を union すると
    重複は普通に起こるため，この経路は通常経路である．
    """

    def _dir(self, tmp_path: Path) -> Path:
        (tmp_path / "a.feather").write_bytes(b"")
        _values([1, 2], [0.4, 0.9]).write_ipc(
            tmp_path / "a.feather", compression="lz4"
        )
        (tmp_path / "b.feather").write_bytes(b"")
        _values([2], [0.6]).write_ipc(
            tmp_path / "b.feather", compression="lz4"
        )
        return tmp_path

    def test_load_returns_unique_ids(
        self, tmp_path: Path
    ) -> None:
        out = load_search_values([self._dir(tmp_path)])
        assert sorted(out["id"].to_list()) == [1, 2]

    def test_load_resolves_last_in_path_order(
        self, tmp_path: Path
    ) -> None:
        out = load_search_values([self._dir(tmp_path)])
        got = dict(
            zip(
                out["id"].to_list(),
                out["searchWinRate"].to_list(),
                strict=True,
            )
        )
        assert got[2] == pytest.approx(0.6, abs=1e-6)

    def test_apply_finds_nothing_left_to_drop(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """チャンクごとの適用で警告が出ないこと．"""
        values = load_search_values([self._dir(tmp_path)])
        caplog.clear()
        with caplog.at_level("WARNING"):
            for _ in range(3):  # チャンク 3 回ぶん
                apply_search_values(
                    _pre_df([1, 2], [1.0, 0.0]), values
                )
        assert "duplicate id" not in caplog.text


class TestValidateSearchValueSource:
    """検査だけを切り出した入口 (データを読まない)．"""

    def test_returns_paths_in_order(
        self, tmp_path: Path
    ) -> None:
        for name in ("b.feather", "a.feather"):
            (tmp_path / name).write_bytes(b"")
            _values([1], [0.5]).write_ipc(
                tmp_path / name, compression="lz4"
            )
        assert [
            p.name
            for p in validate_search_value_source([tmp_path])
        ] == ["a.feather", "b.feather"]

    def test_raises_on_bad_schema(self, tmp_path: Path) -> None:
        (tmp_path / "x.feather").write_bytes(b"")
        pl.DataFrame({"id": [1]}).write_ipc(
            tmp_path / "x.feather", compression="lz4"
        )
        with pytest.raises(ValueError, match="searchWinRate"):
            validate_search_value_source([tmp_path])

    def test_picks_up_arrow_files(self, tmp_path: Path) -> None:
        """`.arrow` も拾う．

        取りこぼしはエラーにならず**無言の部分適用**になるので，
        実際に読める形式は拾っておく．
        """
        (tmp_path / "v.arrow").write_bytes(b"")
        _values([1], [0.5]).write_ipc(
            tmp_path / "v.arrow", compression="lz4"
        )
        assert [
            p.name
            for p in validate_search_value_source([tmp_path])
        ] == ["v.arrow"]

    def test_ignores_flush_temporaries(
        self, tmp_path: Path
    ) -> None:
        """`_write` の中間ファイル (`*.feather.tmp`) は拾わない．"""
        (tmp_path / "v.feather").write_bytes(b"")
        _values([1], [0.5]).write_ipc(
            tmp_path / "v.feather", compression="lz4"
        )
        (tmp_path / "v.feather.tmp").write_bytes(b"partial")
        assert [
            p.name
            for p in validate_search_value_source([tmp_path])
        ] == ["v.feather"]


class TestShardedOutput:
    """出力はシャードのディレクトリであること．

    累積全体を毎回書き直す方式は書き込み量が行数の二乗で伸びる (実測 1.25M 行
    で 1 回 76ms / flush 間隔 500 局面)．flush は `pending_*` を足すだけにし，
    貯まったら `part_*` へまとめて pending を消す．
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

    def test_a_file_output_is_rejected_with_migration_steps(
        self, tmp_path: Path
    ) -> None:
        """旧形式を黙ってディレクトリ扱いしない．

        数日分の既存出力をどう扱ったのかが利用者から見えなくなる．
        """
        old = tmp_path / "search_values.feather"
        _values([1], [0.5]).write_ipc(old)
        with pytest.raises(
            ValueError, match="now a directory of shards"
        ):
            SearchValueCollector().collect(
                self._option(old, resume=True)
            )
        assert old.exists(), "エラー時に既存を消してはいけない"

    def test_carries_pending_written_before_warmup_ms(
        self, tmp_path: Path
    ) -> None:
        """0.96.0 以前が残した `pending_*` を引き継げること．

        `--resume` は中断のたびに使われるので，列が増えた版へ上げた直後は
        **新旧の列構成が 1 回の実行の中で同居する**．引き継ぎで落ちると，
        まだ確定していない行 (最大 `--shard-rows` 分) が失われる．
        """
        out = tmp_path / "sv"
        out.mkdir()
        old = _values([11, 22], [0.25, 0.75]).drop("warmupMs")
        old.write_ipc(out / "pending_00000001.feather")
        carry = SearchValueCollector()._read_pending(
            [out / "pending_00000001.feather"]
        )
        assert list(carry.columns) == list(SEARCH_VALUE_SCHEMA)
        assert carry["warmupMs"].null_count() == 2
        assert sorted(carry["id"].to_list()) == [11, 22]

    def test_resume_reads_a_legacy_file_moved_into_the_dir(
        self, tmp_path: Path
    ) -> None:
        """移行手順どおり移せば，旧形式が既探索として効くこと．"""
        out = tmp_path / "sv"
        out.mkdir()
        _values([11, 22], [0.25, 0.75]).write_ipc(
            out / "search_values.feather"
        )
        done = SearchValueCollector()._load_done(
            self._option(out, resume=True)
        )
        assert sorted(done["id"].to_list()) == [11, 22]

    def test_resume_unions_legacy_and_shards(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "sv"
        out.mkdir()
        _values([11], [0.25]).write_ipc(
            out / "search_values.feather"
        )
        _values([22], [0.75]).write_ipc(
            out / "part_00000001.feather"
        )
        done = SearchValueCollector()._load_done(
            self._option(out, resume=True)
        )
        assert sorted(done["id"].to_list()) == [11, 22]

    def test_shard_numbering_continues_from_the_maximum(
        self, tmp_path: Path
    ) -> None:
        """個数でなく最大値の次を取る (途中を消しても既存を潰さない)．"""
        out = tmp_path / "sv"
        out.mkdir()
        (out / "part_00000007.feather").touch()
        assert _next_index(out, SHARD_PATTERN) == 8
        assert _next_index(out, PENDING_PATTERN) == 1

    def test_resume_finalises_leftover_pending_shards(
        self, tmp_path: Path
    ) -> None:
        """中断で残った pending が，次の resume で確定シャードにまとまること．

        回帰: `pending_paths` を空で初期化していたため前回の未確定分を
        引き継がず，(a) 前回の pending が永久に確定されない，(b) 蓄積カウンタが
        実行ごとに 0 へ戻る，の 2 つが起きていた．中断を繰り返すほど小さな
        ファイルが溜まり続ける．
        """
        out = tmp_path / "sv"
        out.mkdir()
        _values([1, 2], [0.1, 0.2]).write_ipc(
            out / "pending_00000001.feather"
        )
        _values([3], [0.3]).write_ipc(
            out / "pending_00000002.feather"
        )

        # 入力に HCPE が無いので探索は 0 件．引き継ぎ分だけが確定する
        result = SearchValueCollector().collect(
            self._option(out, resume=True)
        )

        assert result["searched"] == "0"
        assert sorted(p.name for p in out.iterdir()) == [
            "part_00000001.feather"
        ], "pending が確定シャードへまとまり，元は消えること"
        done = SearchValueCollector()._load_done(
            self._option(out, resume=True)
        )
        assert sorted(done["id"].to_list()) == [1, 2, 3]

    def test_a_finished_run_leaves_no_pending(
        self, tmp_path: Path
    ) -> None:
        """目標行数に届かなくても実行の終わりで確定させること．

        端数を pending のまま残す案は採らない: `--resume` のたびに
        `--shard-rows` が同じとは限らず，値が変われば「既存の確定シャードも
        分割し直すのか」という決着のつかない問題になる．
        """
        out = tmp_path / "sv"
        out.mkdir()
        _values([9], [0.9]).write_ipc(
            out / "pending_00000001.feather"
        )
        SearchValueCollector().collect(
            self._option(out, resume=True, shard_rows=1_000_000)
        )
        assert not list(out.glob("pending_*.feather"))
        assert list(out.glob("part_*.feather"))

    def test_ordering_puts_legacy_first_and_pending_last(
        self, tmp_path: Path
    ) -> None:
        """後勝ちの向きが「旧形式 → 確定 → 未確定」になること．

        素の辞書順では `search_values.feather` が `part_*` より後ろに来て，
        移行直後に**旧い値が新しいシャードを上書きする**．
        """
        names = [
            "part_00000010.feather",
            "pending_00000001.feather",
            "search_values.feather",
            "part_00000002.feather",
        ]
        ordered = _ordered_value_paths(
            [tmp_path / n for n in names]
        )
        assert [p.name for p in ordered] == [
            "search_values.feather",
            "part_00000002.feather",
            "part_00000010.feather",
            "pending_00000001.feather",
        ]
