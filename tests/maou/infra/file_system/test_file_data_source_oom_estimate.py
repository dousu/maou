"""``cache_mode='memory'`` の OOM 見積りが全フィールドを数えることの回帰テスト．

`/audit-backlog` 2026-08-13 / backlog 行 FS-D5 (見積り部分を分離して消化)．

``FileManager._concatenate_columnar`` は合計バイト数を **6 フィールド
手書き**で足しており，7 番目の ``move_win_rate``
((N, MOVE_LABELS_NUM) float32) を数え落としていた．``move_win_rate`` は
``move_label`` (float16) のちょうど 2 倍あり，preprocessing データでは
最大級のフィールドなので，40GB 級の入力が 18GB と報告されて 32GB 閾値に
掛からず，**OOM 警告が出ないまま OOM する**状態だった．

実データ 32GB は用意できないので，閾値定数
``OOM_WARNING_THRESHOLD_GB`` を差し替えて警告経路を検証する．罠として，
**旧実装の 6 フィールド合計では閾値を超えず，新実装の 7 フィールド合計
では超える**ように閾値を選んである — 手書き列挙に戻すとこのテストが落ちる．

BigQuery 契約テスト (``test_bq_get_item_contract.py``) と同じく，
``FileManager`` は ``object.__new__`` で生成してファイル I/O を回避する．
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from maou.domain.data.columnar_batch import ColumnarBatch
from maou.domain.move.label import MOVE_LABELS_NUM
from maou.infra.file_system import file_data_source
from maou.infra.file_system.file_data_source import (
    FileDataSource,
)


class _RecordingHandler(logging.Handler):
    """``FileManager.logger`` に直付けして警告を拾うハンドラ．

    ``caplog`` は使えない — このプロジェクトのロギング設定では
    ``maou`` ロガーが root へ伝播しないので ``caplog.records`` が空に
    なる (警告自体は stderr に出ている)．
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def warnings_log() -> Iterator[_RecordingHandler]:
    """``FileManager`` が出す WARNING を集める．"""
    handler = _RecordingHandler()
    logger = FileDataSource.FileManager.logger
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def _make_columnar(n: int) -> ColumnarBatch:
    """``move_win_rate`` を持つ preprocessing 相当のバッチ．"""
    return ColumnarBatch(
        board_positions=np.zeros((n, 9, 9), dtype=np.uint8),
        pieces_in_hand=np.zeros((n, 14), dtype=np.uint8),
        move_label=np.zeros(
            (n, MOVE_LABELS_NUM), dtype=np.float16
        ),
        result_value=np.zeros(n, dtype=np.float16),
        move_win_rate=np.zeros(
            (n, MOVE_LABELS_NUM), dtype=np.float32
        ),
    )


def _make_manager(batches: list[ColumnarBatch]) -> Any:
    """ファイルを読まずに ``_concatenate_columnar`` だけを動かせる状態を作る．"""
    fm = object.__new__(FileDataSource.FileManager)
    fm.array_type = "preprocessing"
    fm._file_entries = [
        FileDataSource.FileManager._FileEntry(
            name=f"f{i}",
            path=None,  # type: ignore[arg-type]
            length=len(b),
            cached_array=None,
            cached_columnar=b,
        )
        for i, b in enumerate(batches)
    ]
    fm.total_rows = sum(len(b) for b in batches)
    fm.total_pages = len(batches)
    fm._concatenated_columnar = None
    return fm


def _six_field_bytes(batches: list[ColumnarBatch]) -> int:
    """旧実装が数えていた 6 フィールドの合計 (``move_win_rate`` を除く)．"""
    return sum(
        b.board_positions.nbytes
        + b.pieces_in_hand.nbytes
        + (
            b.move_label.nbytes
            if b.move_label is not None
            else 0
        )
        + (
            b.result_value.nbytes
            if b.result_value is not None
            else 0
        )
        + (
            b.reachable_squares.nbytes
            if b.reachable_squares is not None
            else 0
        )
        + (
            b.legal_moves_label.nbytes
            if b.legal_moves_label is not None
            else 0
        )
        for b in batches
    )


def test_oom_warning_counts_move_win_rate(
    monkeypatch: pytest.MonkeyPatch,
    warnings_log: _RecordingHandler,
) -> None:
    """``move_win_rate`` を含めた合計で閾値判定されること．

    閾値は「旧 6 フィールド合計 < 閾値 < 新 7 フィールド合計」に置く．
    旧実装ではこの入力で警告が出ず，新実装では出る．
    """
    batches = [_make_columnar(256) for _ in range(4)]
    old_bytes = _six_field_bytes(batches)
    new_bytes = sum(b.nbytes for b in batches)
    assert old_bytes < new_bytes, (
        "テストの前提が崩れている: move_win_rate が合計に効いていない"
    )

    gib = 1024**3
    # 2 つの合計のちょうど中間を閾値にする
    monkeypatch.setattr(
        file_data_source,
        "OOM_WARNING_THRESHOLD_GB",
        ((old_bytes + new_bytes) / 2) / gib,
    )

    _make_manager(batches)._concatenate_columnar()

    assert any(
        "may cause OOM" in message
        for message in warnings_log.messages
    ), (
        "move_win_rate を数え落とすと閾値を超えず，この警告が出ない"
    )


def test_no_warning_below_threshold(
    monkeypatch: pytest.MonkeyPatch,
    warnings_log: _RecordingHandler,
) -> None:
    """閾値以下では警告を出さないこと (警告が常時出るようになっていない)．"""
    batches = [_make_columnar(64) for _ in range(2)]
    gib = 1024**3
    monkeypatch.setattr(
        file_data_source,
        "OOM_WARNING_THRESHOLD_GB",
        (sum(b.nbytes for b in batches) * 2) / gib,
    )

    _make_manager(batches)._concatenate_columnar()

    assert not [
        m for m in warnings_log.messages if "may cause OOM" in m
    ]


def test_numpy_path_shares_the_threshold(
    monkeypatch: pytest.MonkeyPatch,
    warnings_log: _RecordingHandler,
) -> None:
    """hcpe 経路も同じ定数・同じ文面で警告すること．

    ``_concatenate_numpy`` と ``_concatenate_columnar`` は同じ警告文を
    二重に持っていたので ``_warn_if_oom_risk`` に一本化した．
    """
    arrays = [
        np.zeros(1024, dtype=np.dtype([("a", np.float64)]))
        for _ in range(3)
    ]
    fm = object.__new__(FileDataSource.FileManager)
    fm.array_type = "hcpe"
    fm._file_entries = [
        FileDataSource.FileManager._FileEntry(
            name=f"f{i}",
            path=None,  # type: ignore[arg-type]
            length=len(a),
            cached_array=a,
        )
        for i, a in enumerate(arrays)
    ]
    fm.total_rows = sum(len(a) for a in arrays)
    fm.total_pages = len(arrays)
    fm._concatenated_array = None

    gib = 1024**3
    monkeypatch.setattr(
        file_data_source,
        "OOM_WARNING_THRESHOLD_GB",
        (sum(a.nbytes for a in arrays) / 2) / gib,
    )

    fm._concatenate_numpy()

    assert any(
        "may cause OOM" in message
        for message in warnings_log.messages
    )


def test_concatenation_result_is_unchanged() -> None:
    """警告経路を触っても結合結果そのものは変わらないこと (P3 の特性テスト)．"""
    batches = [_make_columnar(8), _make_columnar(5)]
    fm = _make_manager(batches)
    fm._concatenate_columnar()

    merged = fm._concatenated_columnar
    assert merged is not None
    assert len(merged) == 13
    assert merged.move_win_rate is not None
    assert merged.move_win_rate.shape == (13, MOVE_LABELS_NUM)
    # 結合後に入力キャッシュが解放されること (ピークメモリの前提)
    assert all(
        e.cached_columnar is None for e in fm._file_entries
    )
