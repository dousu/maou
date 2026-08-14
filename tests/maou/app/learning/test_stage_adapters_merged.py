"""Stage1/Stage2 アダプタの統合を固定する回帰テスト．

`audits/coverage.md` § Deferred backlog の
**2026-08-08 app/learning Deferred 3** 行に対するもの．

6 つのアダプタクラスは 3 つの重複した対で，2026-08-14 のユーザ判断に
より 3 クラスへ統合した．旧 6 名は別名として残す (公開名は消えない)．

固定するのは 3 点:

1. **別名であること** — 旧 6 名が新 3 クラスと同一オブジェクトである．
   これが崩れると「別名なので import が壊れない」という前提が消える．
2. **``set_epoch`` のガード** — 統合前は Stage 1 側だけが ``hasattr``
   で守り，Stage 2 側は無防備に呼んでいた．**安全側 (ガードを付ける
   側) で揃えた**ので，``set_epoch`` を持たない dataset を包んでも
   ``AttributeError`` にならない．これが本統合で唯一変わった挙動
   (= クラス P4 の理由) なので，明示的に固定する．
3. **head の型で選び分けていた分岐が消えたこと** — アダプタは head を
   そのまま保持するだけで，型を見ない．
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

from maou.app.learning.multi_stage_training import (
    Stage1DatasetAdapter,
    Stage1ModelAdapter,
    Stage2DatasetAdapter,
    Stage2ModelAdapter,
    StageDatasetAdapter,
    StageModelAdapter,
)
from maou.app.learning.network import (
    HeadlessNetwork,
    LegalMovesHead,
    ReachableSquaresHead,
)
from maou.app.learning.streaming_dataset import (
    Stage1StreamingAdapter,
    Stage2StreamingAdapter,
    StageStreamingAdapter,
)


@pytest.mark.parametrize(
    ("alias", "merged"),
    [
        (Stage1ModelAdapter, StageModelAdapter),
        (Stage2ModelAdapter, StageModelAdapter),
        (Stage1DatasetAdapter, StageDatasetAdapter),
        (Stage2DatasetAdapter, StageDatasetAdapter),
        (Stage1StreamingAdapter, StageStreamingAdapter),
        (Stage2StreamingAdapter, StageStreamingAdapter),
    ],
)
def test_old_names_are_aliases_of_the_merged_class(
    alias: type, merged: type
) -> None:
    """旧 6 名は新 3 クラスの別名である."""
    assert alias is merged


class _DatasetWithoutSetEpoch:
    """``set_epoch`` を持たないストリーミング dataset のモック."""

    def __len__(self) -> int:
        return 1

    def __iter__(
        self,
    ) -> Iterator[
        tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]
    ]:
        board = torch.randint(0, 32, (2, 9, 9))
        hand = torch.randn(2, 14)
        yield (board, hand), torch.zeros(2, 81)


class _DatasetWithSetEpoch(_DatasetWithoutSetEpoch):
    """``set_epoch`` を持つストリーミング dataset のモック."""

    def __init__(self) -> None:
        self.epoch: int | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def test_set_epoch_is_guarded_for_datasets_without_it() -> None:
    """``set_epoch`` を持たない dataset を包んでも落ちない.

    統合前の ``Stage2StreamingAdapter`` は無防備に
    ``self._dataset.set_epoch(epoch)`` を呼んでいたので，ここは
    ``AttributeError`` になっていた．ガードを付ける側で揃えたことで
    黙って何もしなくなる — 本統合で唯一変わった挙動．
    """
    adapter = StageStreamingAdapter(_DatasetWithoutSetEpoch())  # type: ignore[arg-type]

    adapter.set_epoch(3)  # 例外にならないことが主張


def test_set_epoch_is_still_delegated_when_present() -> None:
    """ガードは委譲そのものを殺していない."""
    dataset = _DatasetWithSetEpoch()
    adapter = StageStreamingAdapter(dataset)  # type: ignore[arg-type]

    adapter.set_epoch(7)

    assert dataset.epoch == 7


@pytest.mark.parametrize(
    "head_factory",
    [
        lambda dim: ReachableSquaresHead(input_dim=dim),
        lambda dim: LegalMovesHead(input_dim=dim),
    ],
    ids=["stage1-head", "stage2-head"],
)
def test_model_adapter_takes_either_head(head_factory) -> None:  # type: ignore[no-untyped-def]
    """統合後のモデルアダプタは head の型を見ない.

    統合前は ``StageComponentFactory._build_model`` が
    ``isinstance(head, ReachableSquaresHead)`` で 2 つの同一クラスを
    選び分け，第 3 の型を ``TypeError`` で弾いていた．その ``else``
    腕は 2 度の監査でいずれも到達不能と確認されており，分岐ごと除去
    した．
    """
    backbone = HeadlessNetwork(
        board_vocab_size=32,
        embedding_dim=64,
        hand_projection_dim=0,
        architecture="resnet",
        out_channels=(16, 32, 64, 64),
    )
    head = head_factory(backbone.embedding_dim)
    adapter = StageModelAdapter(backbone, head)

    assert adapter.head is head

    board = torch.randint(0, 32, (2, 9, 9))
    hand = torch.randn(2, 14)
    policy, value = adapter((board, hand))

    assert policy.shape[0] == 2
    torch.testing.assert_close(value, torch.zeros(2, 1))
