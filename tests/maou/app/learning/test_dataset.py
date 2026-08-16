"""Tests for dataset handling in the learning module."""

from __future__ import annotations

import abc
import pathlib

import numpy as np
import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader

from maou.app.common.data_io_service import DataIOService
from maou.app.learning.dataset import DataSource, KifDataset
from maou.domain.data.array_io import save_preprocessing_df
from maou.domain.data.schema import (
    MOVE_LABELS_NUM,
    convert_numpy_to_preprocessing_df,
    create_empty_preprocessing_array,
    create_empty_preprocessing_df,
)


class _ArrayDataSource(DataSource):
    """Minimal ``DataSource`` backed by a numpy structured array."""

    def __init__(self, data: np.ndarray) -> None:
        self._data = data

    def __getitem__(self, idx: int) -> np.ndarray:
        return self._data[idx]

    def __len__(self) -> int:
        return len(self._data)


def test_preprocessed_batches_omit_legal_move_mask() -> None:
    """前処理済みレコードは legal_move_mask を含まない．

    この経路が作れるマスクは常に全 1 で消費側では no-op なの
    に，バッチ毎に moveLabel と同じサイズを転送していた．
    moveWinRate がない旧形式では targets は 2 要素になる．
    """

    dtype = np.dtype(
        [
            ("boardIdPositions", np.uint8, (9, 9)),
            ("piecesInHand", np.uint8, (14,)),
            ("moveLabel", np.float32, (5,)),
            ("resultValue", np.float32),
        ]
    )
    data = np.array(
        [
            (
                np.eye(9, dtype=np.uint8),
                np.zeros(14, dtype=np.uint8),
                np.array([1, 0, 0, 0, 0], dtype=np.float32),
                np.float32(1.0),
            ),
            (
                np.fliplr(np.eye(9, dtype=np.uint8)),
                np.zeros(14, dtype=np.uint8),
                np.array([0, 1, 0, 0, 0], dtype=np.float32),
                np.float32(-1.0),
            ),
        ],
        dtype=dtype,
    )

    dataset = KifDataset(datasource=_ArrayDataSource(data))

    loader = DataLoader(dataset, batch_size=2)
    (boards, pieces), targets = next(iter(loader))

    assert isinstance(boards, torch.Tensor)
    assert boards.dtype == torch.uint8
    assert boards.shape == (2, 9, 9)
    assert isinstance(pieces, torch.Tensor)
    assert pieces.dtype == torch.uint8
    assert pieces.shape == (2, 14)
    # moveWinRate がないので (move_label, result_value) のみ．
    assert len(targets) == 2
    move_label, result_value = targets
    assert move_label.shape == (2, 5)
    assert result_value.shape == (2, 1)


def test_dataset_accepts_float16_move_labels() -> None:
    """Structured arrays with float16 policy labels remain loadable."""

    dtype = np.dtype(
        [
            ("boardIdPositions", np.uint8, (2, 2)),
            ("piecesInHand", np.uint8, (4,)),
            ("moveLabel", np.float16, (3,)),
            ("resultValue", np.float32),
        ]
    )
    data = np.array(
        [
            (
                np.ones((2, 2), dtype=np.uint8),
                np.zeros(4, dtype=np.uint8),
                np.array([0.5, 0.25, 0.25], dtype=np.float16),
                np.float32(0.0),
            )
        ],
        dtype=dtype,
    )

    dataset = KifDataset(datasource=_ArrayDataSource(data))

    (_, _), (policy, _) = dataset[0]

    assert policy.dtype == torch.float16


def test_dataset_returns_move_win_rate_when_present() -> None:
    """moveWinRate field is returned as the 3rd target element."""

    dtype = np.dtype(
        [
            ("boardIdPositions", np.uint8, (2, 2)),
            ("piecesInHand", np.uint8, (4,)),
            ("moveLabel", np.float16, (3,)),
            ("resultValue", np.float32),
            ("moveWinRate", np.float32, (3,)),
        ]
    )
    data = np.array(
        [
            (
                np.ones((2, 2), dtype=np.uint8),
                np.zeros(4, dtype=np.uint8),
                np.array([0.5, 0.25, 0.25], dtype=np.float16),
                np.float32(0.0),
                np.array([0.8, 0.6, 0.1], dtype=np.float32),
            )
        ],
        dtype=dtype,
    )

    dataset = KifDataset(datasource=_ArrayDataSource(data))

    (_, _), (_, _, move_win_rate) = dataset[0]

    assert move_win_rate is not None
    assert move_win_rate.dtype == torch.float32
    assert move_win_rate.shape == (3,)
    assert torch.allclose(
        move_win_rate,
        torch.tensor([0.8, 0.6, 0.1], dtype=torch.float32),
    )


def test_dataset_returns_2_element_tuple_when_no_win_rate() -> (
    None
):
    """Target tuple has 2 elements when moveWinRate field is absent."""

    dtype = np.dtype(
        [
            ("boardIdPositions", np.uint8, (2, 2)),
            ("piecesInHand", np.uint8, (4,)),
            ("moveLabel", np.float16, (3,)),
            ("resultValue", np.float32),
        ]
    )
    data = np.array(
        [
            (
                np.ones((2, 2), dtype=np.uint8),
                np.zeros(4, dtype=np.uint8),
                np.array([0.5, 0.25, 0.25], dtype=np.float16),
                np.float32(0.0),
            )
        ],
        dtype=dtype,
    )

    dataset = KifDataset(datasource=_ArrayDataSource(data))

    (_, _), targets = dataset[0]

    assert len(targets) == 2


def test_dataset_requires_board_identifiers() -> None:
    """Datasets missing board ID grids should raise a helpful error."""

    dtype = np.dtype(
        [
            ("features", np.uint8, (4, 9, 9)),
            ("moveLabel", np.float16, (5,)),
            ("resultValue", np.float16),
        ]
    )
    data = np.zeros(1, dtype=dtype)

    dataset = KifDataset(datasource=_ArrayDataSource(data))

    with pytest.raises(ValueError):
        dataset[0]


def test_numpy_to_tensor_requires_writeable_buffer() -> None:
    """Read-only buffers should surface actionable guidance."""

    dtype = np.dtype(
        [
            ("boardIdPositions", np.uint8, (2, 2)),
            ("piecesInHand", np.uint8, (2,)),
            ("moveLabel", np.float32, (1,)),
            ("resultValue", np.float32),
        ]
    )
    data = np.zeros(1, dtype=dtype)
    data.setflags(write=False)

    dataset = KifDataset(datasource=_ArrayDataSource(data))

    with pytest.raises(ValueError, match="read-only"):
        dataset[0]


@pytest.mark.skip(reason="Needs update for DataFrame-based I/O")
def test_numpy_to_tensor_preserves_memmap_zero_copy(
    tmp_path: pathlib.Path,
) -> None:
    """Copy-on-write preprocessing memmaps stay writeable for tensors."""

    prep_array = create_empty_preprocessing_array(1)
    prep_array["boardIdPositions"] = np.ones(
        prep_array["boardIdPositions"].shape,
        dtype=np.uint8,
    )

    file_path = tmp_path / "zero_copy.feather"
    # Convert numpy array to Polars DataFrame and save as .feather
    df = convert_numpy_to_preprocessing_df(prep_array)
    save_preprocessing_df(df, file_path)

    loaded_array = DataIOService.load_array(
        file_path,
        array_type="preprocessing",
        bit_pack=False,
    )

    assert isinstance(loaded_array, np.memmap)
    assert loaded_array.flags.writeable

    record = loaded_array[0]
    board_np = record["boardIdPositions"]
    tensor = KifDataset._numpy_to_tensor(
        board_np,
        field_name="boardIdPositions",
        expected_dtype=np.uint8,
    )

    ptr = tensor.data_ptr()
    board_np[0, 0] = 9

    assert tensor.data_ptr() == ptr
    assert int(tensor[0, 0]) == 9


# ============================================================================
# /audit-backlog 2026-08-13 — backlog 行 N3 の回帰テスト
# ============================================================================


class TestDataSourceIsAbstract:
    """``DataSource`` が実際に ABC であること．

    N3: 以前は ``@abc.abstractmethod`` を付けながら ``ABCMeta`` を
    使っていなかったので，未実装の実装が**構築時に一切捕まらなかった**．
    `BigQueryDataSource.__getitem__` が ``pl.DataFrame`` を返していた
    不具合 (O1) が実行時まで露見しなかった根本原因．
    """

    def test_base_uses_abcmeta(self) -> None:
        assert isinstance(DataSource, abc.ABCMeta)
        assert DataSource.__abstractmethods__ == frozenset(
            {"__getitem__", "__len__"}
        )

    def test_incomplete_subclass_fails_at_construction(
        self,
    ) -> None:
        """抽象メソッドを埋めない実装は構築時に落ちること.

        **trap**: デコレータだけでは何も起きない．この assert が
        ``TypeError`` を期待しなくなったら ABC が外れている．
        """

        class _MissingLen(DataSource):
            def __getitem__(self, idx: int) -> np.ndarray:
                raise NotImplementedError

        with pytest.raises(TypeError, match="__len__"):
            _MissingLen()  # type: ignore[abstract]

    def test_complete_subclass_still_constructs(self) -> None:
        """全部埋めた実装はこれまでどおり構築できること (挙動不変)."""

        class _Complete(DataSource):
            def __getitem__(self, idx: int) -> np.ndarray:
                return np.zeros(1)

            def __len__(self) -> int:
                return 0

        assert len(_Complete()) == 0

    def test_learning_datasource_stays_abstract(self) -> None:
        """``LearningDataSource`` は中間抽象のまま構築できないこと.

        両メソッドとも未定義の中間クラスで，production では一度も
        構築されない．ABC 化でここが構築不能になるのは意図どおり．
        """
        from maou.app.learning.dl import LearningDataSource

        with pytest.raises(TypeError):
            LearningDataSource()  # type: ignore[abstract]


# ============================================================================
# /audit-backlog 2026-08-15 — backlog 行 D13 (2)(4) の回帰テスト
# ============================================================================


def _varied_preprocessing_df(rows: int) -> pl.DataFrame:
    """全フィールドが行ごとに異なる preprocessing DataFrame を作る．

    ``create_empty_preprocessing_df`` のままだと全部ゼロなので，
    フィールドの取り違えや脱落があっても等価性テストが通ってしまう
    (characterization test の非空虚性が保てない)．
    """
    return create_empty_preprocessing_df(rows).with_columns(
        [
            pl.Series("id", list(range(rows))),
            pl.Series(
                "boardIdPositions",
                [
                    [
                        [
                            (r * 81 + c * 9 + k) % 256
                            for k in range(9)
                        ]
                        for c in range(9)
                    ]
                    for r in range(rows)
                ],
            ),
            pl.Series(
                "piecesInHand",
                [
                    [(r * 14 + k) % 256 for k in range(14)]
                    for r in range(rows)
                ],
            ),
            pl.Series(
                "moveLabel",
                [
                    [
                        float((r + k) % 7)
                        for k in range(MOVE_LABELS_NUM)
                    ]
                    for r in range(rows)
                ],
            ),
            pl.Series(
                "moveWinRate",
                [
                    [
                        float((r + k) % 5)
                        for k in range(MOVE_LABELS_NUM)
                    ]
                    for r in range(rows)
                ],
            ),
            pl.Series(
                "resultValue",
                [float(r % 3) - 1.0 for r in range(rows)],
            ),
        ]
    )


class _StructuredOnly(DataSource):
    """列アクセサを持たないソース (ABC の既定実装のまま)．

    同じデータを **structured array 経路だけ**で読ませるための
    ラッパー．速い口を持つソースと突き合わせることで，
    「2 つの経路が同じサンプルを返す」を同一データ上で固定できる．
    """

    def __init__(self, inner: DataSource) -> None:
        self._inner = inner

    def __getitem__(self, idx: int) -> np.ndarray:
        return self._inner[idx]

    def __len__(self) -> int:
        return len(self._inner)


def _preprocessing_source(
    tmp_path: pathlib.Path, rows: int
) -> DataSource:
    from maou.infra.file_system.file_data_source import (
        FileDataSource,
    )

    path = tmp_path / "varied.feather"
    save_preprocessing_df(_varied_preprocessing_df(rows), path)
    return FileDataSource(
        file_paths=[path], array_type="preprocessing"
    )


class TestColumnarFastPath:
    """列アクセサ経由のサンプル生成が structured 経路と等価であること．

    `/audit-backlog` 2026-08-15 / backlog 行 D13 (2)(4)．
    ``KifDataset`` は ``DataSource.columnar_record`` があればそちらを
    使い，無ければ従来の structured array 経路へ落ちる．固定したい
    のは **2 つの経路が同じサンプルを返すこと**と，**ABC の既定実装が
    従来のソースを壊さないこと**である．
    """

    def test_abc_default_returns_none(self) -> None:
        """列アクセサは既定で ``None`` を返し，抽象にはしないこと．

        BigQuery / ObjectStorage / テストの fake は override して
        いない．抽象メソッドに**すると**それら全部が構築不能になる
        ので，既定実装であることごと固定する．
        """
        source = _ArrayDataSource(
            create_empty_preprocessing_array(1)
        )

        assert source.columnar_record(0) is None
        assert (
            "columnar_record"
            not in DataSource.__abstractmethods__
        )

    def test_columnar_and_structured_paths_agree(
        self, tmp_path: pathlib.Path
    ) -> None:
        """同じデータを 2 経路で読んで同じサンプルが返ること．

        characterization test — 右辺 (``_StructuredOnly``) は修正前の
        ``KifDataset`` がやっていたことそのものなので，「挙動不変」の
        根拠になる．
        """
        source = _preprocessing_source(tmp_path, 4)
        fast = KifDataset(datasource=source)
        slow = KifDataset(datasource=_StructuredOnly(source))

        assert len(fast) == len(slow) == 4
        for idx in range(4):
            (fb, fp), ft = fast[idx]
            (sb, sp), st = slow[idx]
            assert torch.equal(fb, sb), (
                f"board differs at {idx}"
            )
            assert torch.equal(fp, sp), (
                f"pieces differ at {idx}"
            )
            assert len(ft) == len(st), (
                f"target arity differs at {idx}: "
                f"{len(ft)} vs {len(st)}"
            )
            for tidx, (f, s) in enumerate(zip(ft, st)):
                assert torch.equal(f, s), (
                    f"target {tidx} differs at row {idx}"
                )

    def test_fast_path_shares_the_batch_storage(
        self, tmp_path: pathlib.Path
    ) -> None:
        """速い口のテンソルが元バッチとストレージを共有すること．

        この所見そのもの — 共有していなければサンプル毎のコピーが
        まだ残っているということで，修正の意味が消える．
        ``_StructuredOnly`` 経由では共有され**ない**ことも併せて
        固定し，このテストが経路を取り違えていないことを示す．
        """
        source = _preprocessing_source(tmp_path, 4)
        entry = (
            source._FileDataSource__file_manager._file_entries[  # type: ignore[attr-defined]
                0
            ]
        )
        batch = entry.cached_columnar
        assert batch is not None

        (board, pieces), _ = KifDataset(datasource=source)[2]
        assert np.shares_memory(
            board.numpy(), batch.board_positions
        )
        assert np.shares_memory(
            pieces.numpy(), batch.pieces_in_hand
        )

        (slow_board, _), _ = KifDataset(
            datasource=_StructuredOnly(source)
        )[2]
        assert not np.shares_memory(
            slow_board.numpy(), batch.board_positions
        )

    def test_duck_typed_source_still_works(self) -> None:
        """ABC を継承しないソースでも従来どおり動くこと．

        **trap**: ``PolarsDataFrameSource`` は structured array では
        なく ``_PolarsRow`` を返すので ABC を正直には名乗れない．
        ``self.__datasource.columnar_record(idx)`` を無条件に呼ぶと
        そこで ``AttributeError`` になる．契約の実装有無は構築時に
        ``isinstance`` で判定している．
        """
        data = create_empty_preprocessing_array(2)
        data["boardIdPositions"] = 1
        data["piecesInHand"] = 2

        class _DuckTyped:
            def __getitem__(self, idx: int) -> np.ndarray:
                return data[idx]

            def __len__(self) -> int:
                return len(data)

        dataset = KifDataset(datasource=_DuckTyped())  # type: ignore[arg-type]

        (board, _), _ = dataset[0]
        assert board.shape == (9, 9)

    def test_missing_named_fields_still_raise(self) -> None:
        """フィールド名を持たない配列は従来どおり ``ValueError`` にすること．

        ``dtype.names is None`` のガードは速い口の導入で
        :func:`_record_fields` へ移した．移し忘れると，名前の無い
        配列が ``"boardIdPositions" not in None`` で ``TypeError`` に
        化ける．
        """
        dataset = KifDataset(
            datasource=_ArrayDataSource(
                np.zeros((2, 3), dtype=np.uint8)
            )
        )

        with pytest.raises(
            ValueError, match="lacks named fields"
        ):
            dataset[0]

    def test_stage_dataset_uses_the_fast_path(
        self, tmp_path: pathlib.Path
    ) -> None:
        """``_StageDataset`` 側も速い口を通り，同じ値を返すこと．

        ``KifDataset`` だけを直すと stage1/stage2 の事前学習だけが
        サンプル毎のコピーを払い続ける．
        """
        from maou.app.learning.dataset import Stage1Dataset
        from maou.domain.data.rust_io import save_stage1_df
        from maou.domain.data.schema import (
            create_empty_stage1_df,
        )
        from maou.infra.file_system.file_data_source import (
            FileDataSource,
        )

        rows = 3
        df = create_empty_stage1_df(rows).with_columns(
            [
                pl.Series("id", list(range(rows))),
                pl.Series(
                    "boardIdPositions",
                    [
                        [
                            [
                                (r + c + k) % 256
                                for k in range(9)
                            ]
                            for c in range(9)
                        ]
                        for r in range(rows)
                    ],
                ),
                pl.Series(
                    "piecesInHand",
                    [
                        [(r + k) % 256 for k in range(14)]
                        for r in range(rows)
                    ],
                ),
                pl.Series(
                    "reachableSquares",
                    [
                        [
                            [(r * c * k) % 2 for k in range(9)]
                            for c in range(9)
                        ]
                        for r in range(rows)
                    ],
                ),
            ]
        )
        path = tmp_path / "stage1.feather"
        save_stage1_df(df, path)

        source = FileDataSource(
            file_paths=[path], array_type="stage1"
        )
        assert source.columnar_record(0) is not None

        fast = Stage1Dataset(datasource=source)
        slow = Stage1Dataset(datasource=_StructuredOnly(source))

        for idx in range(rows):
            (fb, fp), ft = fast[idx]
            (sb, sp), st = slow[idx]
            assert torch.equal(fb, sb)
            assert torch.equal(fp, sp)
            assert torch.equal(ft, st)

        # 値の一致だけでは経路を判別できない (両経路は同じ値を返すのが
        # 正しいため)．``_StageDataset`` が実際に速い口を通っている
        # ことは，元バッチとのストレージ共有でしか観測できない．
        batch = (
            source._FileDataSource__file_manager._file_entries[  # type: ignore[attr-defined]
                0
            ].cached_columnar
        )
        assert batch is not None
        (fast_board, _), _ = fast[1]
        (slow_board, _), _ = slow[1]
        assert np.shares_memory(
            fast_board.numpy(), batch.board_positions
        )
        assert not np.shares_memory(
            slow_board.numpy(), batch.board_positions
        )
