"""検索インデックスの抽象インターフェース定義．

infra層の ``SearchIndex`` 実装に依存せず，
interface層・app層が参照できるProtocolを提供する．
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SearchIndexProtocol(Protocol):
    """検索インデックスのProtocol．

    infra層の ``SearchIndex`` が暗黙的に実装する．
    interface層・app層はこのProtocolに依存することで，
    infra → interface → app → domain の依存方向を維持する．
    """

    use_mock_data: bool

    def search_by_id(
        self, record_id: str
    ) -> tuple[int, int] | None:
        """IDでレコード位置を検索する．

        Args:
            record_id: 検索するレコードID

        Returns:
            (file_index, row_number) のタプル，見つからない場合は None
        """
        ...

    def get_eval_by_position(
        self, file_index: int, row_number: int
    ) -> int | None:
        """ファイルインデックスと行番号から評価値を取得する．

        Args:
            file_index: ファイルインデックス
            row_number: 行番号

        Returns:
            評価値，取得できない場合は None
        """
        ...

    def search_id_prefix(
        self, prefix: str, limit: int = 50
    ) -> list[str]:
        """IDプレフィックスで候補を検索する．

        Args:
            prefix: 検索プレフィックス
            limit: 最大取得件数

        Returns:
            マッチするIDのリスト
        """
        ...

    def get_all_ids(
        self, limit: int | None = None
    ) -> list[str]:
        """全IDリストを取得する．

        Args:
            limit: 最大取得件数（Noneで全件）

        Returns:
            IDのリスト
        """
        ...

    def search_by_eval_range(
        self,
        min_eval: int | None = None,
        max_eval: int | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> list[tuple[int, int]]:
        """評価値範囲でレコードを検索する．

        Args:
            min_eval: 最小評価値（Noneで下限なし）
            max_eval: 最大評価値（Noneで上限なし）
            offset: 開始オフセット
            limit: 最大取得件数

        Returns:
            (file_index, row_number) のタプルのリスト
        """
        ...

    def count_eval_range(
        self,
        min_eval: int | None = None,
        max_eval: int | None = None,
    ) -> int:
        """評価値範囲に該当するレコード数を取得する．

        Args:
            min_eval: 最小評価値（Noneで下限なし）
            max_eval: 最大評価値（Noneで上限なし）

        Returns:
            該当レコード数
        """
        ...

    def total_records(self) -> int:
        """総レコード数を取得する．

        Returns:
            総レコード数
        """
        ...
