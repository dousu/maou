import abc
import logging
from collections.abc import Collection, Sized
from typing import Any, cast

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

    def columnar_record(
        self, idx: int
    ) -> dict[str, np.ndarray] | None:
        """1 レコードを**列ごとの配列**として返す (対応するソースのみ)．

        既定は ``None`` を返す — 「このソースに列指向の速い口は無い」
        の意味で，呼び出し側は :meth:`__getitem__` の structured array
        経路へフォールバックする．内部を SOA で持つソース
        (``FileDataSource``) だけがこれを override する．

        **なぜ ABC に生やすのか．** 消費側が ``hasattr`` で速い口の
        有無を嗅ぎ回ると，契約が型に出ないまま実装ごとに散らばる．
        ここに置けば口の名前と戻り値の形が 1 箇所で決まり，
        BigQuery / ObjectStorage は既定実装のまま従来経路に落ちる．

        **戻り値の契約** (``None`` でない場合):

        - キー集合は :meth:`__getitem__` が返す structured array の
          ``dtype.names`` と**一致する**．消費側は列の有無を
          ``dtype.names`` で判定しているので，ここがずれると
          「``moveWinRate`` 列を持たない古いデータで教師の要素数が
          経路によって変わる」といった黙った挙動差になる．ソースが
          その列を供給できない場合は structured 経路と同じくゼロ埋め
          した配列を入れる．
        - 各値の shape は structured array の当該フィールドと同じ
          (スカラーフィールドは 0 次元配列)．
        - 各値は **C-contiguous かつ writeable** — ``torch.from_numpy``
          がストレージを共有するため (``_numpy_to_tensor`` 参照)．
        - 値は元のバッチの**ビュー**でよい．structured 経路が
          サンプル毎に払っていた ``np.empty`` + フィールド毎の
          memcpy を無くすのがこの口の目的である．

        Args:
            idx: レコードインデックス

        Returns:
            フィールド名 → 配列の辞書．非対応なら ``None``
        """
        return None


def _columnar_capable(datasource: object) -> DataSource | None:
    """列アクセサの契約を実装しているソースだけを返す．

    ``KifDataset`` / ``_StageDataset`` は型としては
    :class:`DataSource` を要求するが，実際には ABC を継承しない
    duck-typed なソースも渡される (``PolarsDataFrameSource`` は
    structured array ではなく ``_PolarsRow`` を返すため，ABC の
    ``__getitem__ -> np.ndarray`` を正直には名乗れない)．
    それらに :meth:`DataSource.columnar_record` は生えていないので，
    **契約を実装しているか**を構築時に一度だけ判定して覚えておき，
    ホットパスでは分岐だけにする．

    ``hasattr`` でメソッドの有無を嗅ぐのではなく ABC への
    ``isinstance`` で見ているのが要点である — 前者だと口の所在が
    実装ごとに散らばるが，後者なら常に ABC 1 箇所に決まる．

    Args:
        datasource: 学習データソース

    Returns:
        :class:`DataSource` を実装していればそれ自身，でなければ ``None``
    """
    return (
        datasource
        if isinstance(datasource, DataSource)
        else None
    )


def _record_fields(
    columnar_source: DataSource | None,
    datasource: Any,
    idx: int,
) -> tuple[Any, Collection[str]]:
    """1 サンプル分のフィールド束と，その名前集合を返す．

    速い口 (:meth:`DataSource.columnar_record`) が使えるならその
    辞書を，使えないなら従来どおり structured array を返す．
    どちらも ``obj[field_name]`` でフィールドを取れるので，
    呼び出し側は**同じコードで両方を読める**．名前集合だけは
    取り方が違う (辞書のキー / ``dtype.names``) ので別に返す．

    Args:
        columnar_source: 列アクセサの契約を実装しているソース (無ければ ``None``)
        datasource: フォールバック用のデータソース
        idx: レコードインデックス

    Returns:
        ``(フィールド束, 名前集合)``

    Raises:
        ValueError: structured array 経路でフィールド名を持たない場合
    """
    if columnar_source is not None:
        record = columnar_source.columnar_record(idx)
        if record is not None:
            return record, record.keys()

    data = datasource[idx]
    names = data.dtype.names
    if names is None:
        raise ValueError(
            "Preprocessed record lacks named fields"
        )
    return data, names


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
        self.__columnar_source = _columnar_capable(datasource)

    def __len__(self) -> int:
        return len(self.__datasource)

    def __getitem__(
        self, idx: int
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, ...],
    ]:
        # 前処理済みのデータを使う．
        # 列アクセサを持つソース (内部が SOA の FileDataSource) では
        # 列ごとのビューを直接受け取り，サンプル毎の
        # ``np.empty(9,081B)`` + フィールド毎 memcpy を回避する．
        # 持たないソースは従来どおり structured array を返す．
        data, names = _record_fields(
            self.__columnar_source, self.__datasource, idx
        )

        # torch.from_numpy()を使用してゼロコピー変換（read-onlyの場合はcopy()で回避）
        # Dataset内ではCUDA操作を避け、DataLoaderのpin_memory機能を活用
        if "boardIdPositions" not in names:
            raise ValueError(
                "Preprocessed record lacks boardIdPositions"
            )

        if "piecesInHand" not in names:
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
            self._has_move_win_rate = "moveWinRate" in names
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
        record: Any,
        *,
        field_name: str,
        expected_dtype: DTypeLike | tuple[DTypeLike, ...],
    ) -> torch.Tensor:
        """1 フィールドをテンソルへ変換する．

        ``record`` は structured array (欠けたフィールドで numpy が
        ``ValueError``) でも，:meth:`DataSource.columnar_record` が
        返す辞書 (欠けたキーで ``KeyError``) でもよい．どちらも同じ
        ``ValueError`` に揃えて投げ直す．
        """
        try:
            field = record[field_name]
        except (
            ValueError,
            KeyError,
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
        self.__columnar_source = _columnar_capable(datasource)
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
        # KifDataset と同じく列アクセサがあればそちらを使う
        # (:func:`_record_fields` 参照)．
        data, _ = _record_fields(
            self.__columnar_source, self.__datasource, idx
        )

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
