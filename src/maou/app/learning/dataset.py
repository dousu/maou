import abc
import logging
from collections.abc import Sized
from typing import cast

import numpy as np
import torch
from numpy.typing import DTypeLike
from torch.utils.data import Dataset

_EXPECTED_DTYPE_CACHE: dict[object, tuple[np.dtype, ...]] = {}


def _resolve_expected_dtypes(
    expected_dtype: DTypeLike | tuple[DTypeLike, ...],
) -> tuple[np.dtype, ...]:
    """許容dtype指定を ``np.dtype`` のタプルへ正規化する (キャッシュ付き)．

    ``_numpy_to_tensor`` はサンプルごとに 4〜5 回呼ばれ，その度に
    ``np.dtype()`` を作り直していた．呼び出し側が渡す指定は
    モジュール定数の少数パターンしかないため，結果をキャッシュする．

    Args:
        expected_dtype: 単一dtype，またはdtypeのタプル

    Returns:
        正規化された ``np.dtype`` のタプル
    """
    key: object = expected_dtype
    cached = _EXPECTED_DTYPE_CACHE.get(key)
    if cached is not None:
        return cached
    resolved = (
        tuple(
            np.dtype(cast(DTypeLike, dtype))
            for dtype in expected_dtype
        )
        if isinstance(expected_dtype, tuple)
        else (np.dtype(cast(DTypeLike, expected_dtype)),)
    )
    _EXPECTED_DTYPE_CACHE[key] = resolved
    return resolved


class DataSource(abc.ABC):
    """学習用データソースの抽象基底．

    ``abc.ABC`` を継承しているのは，未実装のメソッドを**構築時に**
    捕まえるため．以前は ``@abc.abstractmethod`` を付けながら
    ``ABCMeta`` を使っていなかったので，これらのデコレータは
    documentation 以上の意味を持たず，`BigQueryDataSource.__getitem__`
    が ``pl.DataFrame`` を返していた不具合が実行時まで露見しなかった．
    """

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> np.ndarray:
        """
        指定されたインデックスのレコードをnumpy structured arrayとして返す

        Returns:
            np.ndarray: structured arrayの単一レコード（0次元配列）
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        pass


class KifDataset(Dataset, Sized):
    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(
        self,
        *,
        datasource: DataSource,
    ):
        self.__datasource = datasource
        self.logger.info(f"{len(self.__datasource)} samples")
        self._has_move_win_rate: bool | None = None

    def __len__(self) -> int:
        return len(self.__datasource)

    def __getitem__(
        self, idx: int
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, ...],
    ]:
        # 前処理済みのデータを使う（structured arrayから直接アクセス）
        data = self.__datasource[
            idx
        ]  # numpy structured array (0次元)

        # torch.from_numpy()を使用してゼロコピー変換（read-onlyの場合はcopy()で回避）
        # Dataset内ではCUDA操作を避け、DataLoaderのpin_memory機能を活用
        if data.dtype.names is None:
            raise ValueError(
                "Preprocessed record lacks named fields"
            )

        if "boardIdPositions" not in data.dtype.names:
            raise ValueError(
                "Preprocessed record lacks boardIdPositions"
            )

        if "piecesInHand" not in data.dtype.names:
            raise ValueError(
                "Preprocessed record lacks piecesInHand"
            )

        board_tensor = self._structured_field_to_tensor(
            data,
            field_name="boardIdPositions",
            expected_dtype=np.uint8,
        )
        pieces_in_hand_tensor = (
            self._structured_field_to_tensor(
                data,
                field_name="piecesInHand",
                expected_dtype=np.uint8,
            )
        )
        move_label_tensor = self._structured_field_to_tensor(
            data,
            field_name="moveLabel",
            expected_dtype=(np.float16, np.float32),
        )
        result_value_tensor = torch.tensor(
            data["resultValue"].item(), dtype=torch.float32
        ).reshape(1)

        # DataLoaderのpin_memory機能と競合を避けるため、Dataset内ではCPUテンソルを返す
        # GPU転送はDataLoaderが自動的に処理する
        #
        # legal_move_mask は返さない．この経路が作れるマスクは
        # 常に torch.ones_like(moveLabel) であり，消費側の 5 つの
        # カーネルすべてで no-op でありながら，バッチ毎に
        # moveLabel と同じサイズ (B=1024 で約 9MB) を PCIe 上に
        # 流していた．TrainingLoop._unpack_batch() は
        # legal_move_mask=None として扱う．
        #
        # moveWinRateが存在する場合のみ3要素tupleを返す．
        # Noneを含むtupleはPyTorchのdefault_collateに非対応のため，
        # 2要素tupleを維持する．
        if self._has_move_win_rate is None:
            self._has_move_win_rate = (
                data.dtype.names is not None
                and "moveWinRate" in data.dtype.names
            )
        if self._has_move_win_rate:
            move_win_rate_tensor = (
                self._structured_field_to_tensor(
                    data,
                    field_name="moveWinRate",
                    expected_dtype=(
                        np.float16,
                        np.float32,
                    ),
                )
            )
            return (
                (board_tensor, pieces_in_hand_tensor),
                (
                    move_label_tensor,
                    result_value_tensor,
                    move_win_rate_tensor,
                ),
            )

        return (
            (board_tensor, pieces_in_hand_tensor),
            (
                move_label_tensor,
                result_value_tensor,
            ),
        )

    @staticmethod
    def _structured_field_to_tensor(
        record: np.ndarray,
        *,
        field_name: str,
        expected_dtype: DTypeLike | tuple[DTypeLike, ...],
    ) -> torch.Tensor:
        try:
            field = record[field_name]
        except (
            ValueError
        ) as exc:  # pragma: no cover - numpy raises ValueError
            msg = f"Preprocessed record lacks field `{field_name}`"
            raise ValueError(msg) from exc

        return KifDataset._numpy_to_tensor(
            field,
            field_name=field_name,
            expected_dtype=expected_dtype,
        )

    @staticmethod
    def _numpy_to_tensor(
        array: np.ndarray,
        *,
        field_name: str,
        expected_dtype: DTypeLike | tuple[DTypeLike, ...],
    ) -> torch.Tensor:
        np_array = np.asarray(array)
        # np.dtype() の再構築はサンプル毎×フィールド毎に走る
        # (worker のホットパス) ため結果をキャッシュする．
        expected_dtypes = _resolve_expected_dtypes(
            expected_dtype
        )
        if np_array.dtype not in expected_dtypes:
            expected_desc = (
                expected_dtypes[0].name
                if len(expected_dtypes) == 1
                else " or ".join(
                    dtype.name for dtype in expected_dtypes
                )
            )
            msg = (
                f"Field `{field_name}` must have dtype {expected_desc}, "
                f"got {np_array.dtype}"
            )
            raise TypeError(msg)
        if not np_array.flags.c_contiguous:
            msg = (
                f"Field `{field_name}` must be C-contiguous to enable zero-copy "
                "conversion"
            )
            raise ValueError(msg)
        if not np_array.flags.writeable:
            msg = (
                f"Field `{field_name}` was loaded as read-only. "
                "Ensure preprocessing files are opened via copy-on-write "
                "memory mapping so tensors can share storage."
            )
            raise ValueError(msg)
        return torch.from_numpy(np_array)


class _StageDataset(Dataset, Sized):
    """Common dataset for the multi-stage pre-training stages.

    Stage 1 と Stage 2 は特徴量 (盤面・持ち駒) が同一で，教師信号の
    フィールド名と，そのフィールドを平坦化するかどうかだけが異なる．
    サブクラスは `_stage_label` / `_target_field` / `_flatten_target`
    を定義する．
    """

    logger: logging.Logger = logging.getLogger(__name__)

    #: ログに出すステージ名 ("Stage 1" など)．
    _stage_label: str
    #: 教師信号を取り出す structured array のフィールド名．
    _target_field: str
    #: 教師信号を 1 次元へ平坦化するかどうか．
    _flatten_target: bool

    def __init__(
        self,
        *,
        datasource: DataSource,
    ):
        """Initialize the stage dataset.

        Args:
            datasource: Data source providing the stage's training data
        """
        self.__datasource = datasource
        self.logger.info(
            f"{self._stage_label} Dataset: "
            f"{len(self.__datasource)} samples"
        )

    def __len__(self) -> int:
        return len(self.__datasource)

    def __getitem__(
        self, idx: int
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],  # features
        torch.Tensor,  # target
    ]:
        """Get a single training sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (features，target):
                - features: (board_tensor，pieces_in_hand_tensor)
                    - board_tensor: (9，9) uint8 tensor
                    - pieces_in_hand_tensor: (14，) uint8 tensor
                - target: 1-D float32 tensor of binary labels
        """
        data = self.__datasource[idx]

        board_tensor = KifDataset._structured_field_to_tensor(
            data,
            field_name="boardIdPositions",
            expected_dtype=np.uint8,
        )
        pieces_in_hand_tensor = (
            KifDataset._structured_field_to_tensor(
                data,
                field_name="piecesInHand",
                expected_dtype=np.uint8,
            )
        )
        target_tensor = KifDataset._structured_field_to_tensor(
            data,
            field_name=self._target_field,
            expected_dtype=np.uint8,
        )
        if self._flatten_target:
            # (9，9) -> (81，)
            target_tensor = target_tensor.flatten()
        # Convert to float for BCE loss
        target_tensor = target_tensor.float()

        return (
            (board_tensor, pieces_in_hand_tensor),
            target_tensor,
        )


class Stage1Dataset(_StageDataset):
    """Dataset for Stage 1 (reachable squares) training.

    This dataset is used for the first stage of multi-stage training，
    where the model learns which board squares pieces can move to.
    The target is a 9×9 binary map indicating reachable squares，
    flattened to (81，).

    The data source is expected to provide the schema defined by
    get_stage1_dtype().
    """

    _stage_label = "Stage 1"
    _target_field = "reachableSquares"
    _flatten_target = True


class Stage2Dataset(_StageDataset):
    """Dataset for Stage 2 (legal moves) training.

    This dataset is used for the second stage of multi-stage training，
    where the model learns which moves are legal in a given position.
    The target is a MOVE_LABELS_NUM-dimensional binary vector indicating legal moves.

    The data source is expected to provide the schema defined by
    get_stage2_dtype().
    """

    _stage_label = "Stage 2"
    _target_field = "legalMovesLabel"
    _flatten_target = False
