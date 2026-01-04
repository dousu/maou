"""データ検索ユースケース（アプリケーション層）．

検索インデックスとデータソースを組み合わせてデータを取得する．
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from maou.infra.file_system.file_data_source import (
    FileDataSource,
)
from maou.infra.visualization.search_index import SearchIndex

logger = logging.getLogger(__name__)


class DataRetriever:
    """データ検索オーケストレータ．

    検索インデックスでレコード位置を特定し，データソースから実データを取得する．
    """

    def __init__(
        self,
        search_index: SearchIndex,
        file_paths: List[Path],
        array_type: str,
    ) -> None:
        """データ取得サービスを初期化．

        Args:
            search_index: 検索インデックス
            file_paths: データファイルパスリスト
            array_type: データ型（hcpe, preprocessing, stage1, stage2）
        """
        self.search_index = search_index
        self.file_paths = file_paths
        self.array_type = array_type

        # FileDataSourceは後で必要に応じて初期化
        self._data_source: Optional[FileDataSource] = None

        if search_index.use_mock_data:
            logger.warning(
                "⚠️  DataRetriever in MOCK MODE - will return fake data"
            )
        else:
            logger.info(
                "✅ DataRetriever in REAL MODE - will load actual records"
            )

        logger.info(
            f"DataRetriever initialized: {len(file_paths)} files, "
            f"type={array_type}"
        )

    def _ensure_data_source(self) -> FileDataSource:
        """データソースを遅延初期化．

        Returns:
            FileDataSourceインスタンス
        """
        if self._data_source is None:
            self._data_source = FileDataSource(
                file_paths=self.file_paths,
                array_type=self.array_type,
                cache_mode="mmap",  # 効率的なランダムアクセス
            )
            logger.info(
                "FileDataSource initialized with mmap cache"
            )

        return self._data_source

    def get_by_id(
        self, record_id: str
    ) -> Optional[Dict[str, Any]]:
        """IDでレコードを取得．

        Args:
            record_id: 検索するレコードID

        Returns:
            レコードデータの辞書，見つからない場合はNone
        """
        # インデックスで位置を検索
        location = self.search_index.search_by_id(record_id)

        if location is None:
            logger.info(f"Record not found: {record_id}")
            return None

        file_index, row_number = location

        # Mock mode
        if self.search_index.use_mock_data:
            logger.debug(
                f"⚠️  MOCK: Returning fake record for {record_id}"
            )
            return self._create_mock_record(
                record_id, row_number
            )

        # Real mode
        logger.debug(
            f"✅ REAL: Loading record {record_id} from "
            f"file={self.file_paths[file_index].name}, row={row_number}"
        )

        try:
            record = self._load_record_at_location(
                file_index, row_number
            )
            return record
        except Exception as e:
            logger.exception(f"Failed to load record: {e}")
            return None

    def get_by_eval_range(
        self,
        min_eval: Optional[int],
        max_eval: Optional[int],
        offset: int,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """評価値範囲でレコードを取得．

        Args:
            min_eval: 最小評価値
            max_eval: 最大評価値
            offset: スキップする件数
            limit: 取得する最大件数

        Returns:
            レコードデータのリスト
        """
        # インデックスで位置を検索
        locations = self.search_index.search_by_eval_range(
            min_eval, max_eval, offset, limit
        )

        logger.info(
            f"Found {len(locations)} records in range "
            f"[{min_eval}, {max_eval}], offset={offset}, limit={limit}"
        )

        # Mock mode
        if self.search_index.use_mock_data:
            logger.debug(
                f"⚠️  MOCK: Returning {len(locations)} fake records"
            )
            records = []
            for file_index, row_number in locations:
                record_id = f"mock_id_{row_number}"
                record = self._create_mock_record(
                    record_id, row_number
                )
                records.append(record)
            return records

        # Real mode
        logger.debug(
            f"✅ REAL: Loading {len(locations)} records from files"
        )

        records = []
        for file_index, row_number in locations:
            try:
                record = self._load_record_at_location(
                    file_index, row_number
                )
                records.append(record)
            except Exception as e:
                logger.error(
                    f"Failed to load record at ({file_index}, {row_number}): {e}"
                )
                continue  # Skip failed records

        return records

    def _load_record_at_location(
        self, file_index: int, row_number: int
    ) -> Dict[str, Any]:
        """指定位置から実際のレコードをロード．

        Args:
            file_index: ファイルインデックス
            row_number: 行番号

        Returns:
            レコードデータの辞書

        Raises:
            Exception: ファイル読み込みエラー
        """
        from maou.domain.data.rust_io import (
            load_hcpe_df,
            load_preprocessing_df,
            load_stage1_df,
            load_stage2_df,
        )

        loader_map = {
            "hcpe": load_hcpe_df,
            "preprocessing": load_preprocessing_df,
            "stage1": load_stage1_df,
            "stage2": load_stage2_df,
        }

        load_df = loader_map[self.array_type]
        file_path = self.file_paths[file_index]

        # DataFrameをロード
        df = load_df(file_path)

        # 行を抽出して辞書に変換
        # df[row_number]は1行のDataFrameを返す
        row = df[row_number]
        record = {}
        for col in df.columns:
            value = row[col]
            # Polars Seriesの場合，最初の要素を取得
            if hasattr(value, "to_list"):
                # Seriesから値を取り出す（長さ1のSeriesなので[0]）
                record[col] = value.to_list()[0]
            else:
                record[col] = value.item()

        # HCPEデータの場合，hcpフィールドから盤面情報をデコード
        if self.array_type == "hcpe" and "hcp" in record:
            try:
                record = self._decode_hcp_to_board_info(record)
            except Exception as e:
                logger.warning(
                    f"Failed to decode HCP data: {e}. "
                    f"Visualization may not work properly."
                )

        return record

    def _decode_hcp_to_board_info(
        self, record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """HCPフィールドから盤面情報をデコードする．

        Args:
            record: HCPEレコードデータ（hcpフィールドを含む）

        Returns:
            boardIdPositionsとpiecesInHandを追加したレコード

        Raises:
            Exception: デコード失敗
        """
        import numpy as np
        from maou.domain.board.shogi import Board

        # HCPバイナリデータを取得
        hcp_bytes = record.get("hcp")
        if not hcp_bytes:
            raise ValueError("hcp field is missing or empty")

        # HCPバイナリをuint8配列に変換（32バイト）
        hcp_array = np.frombuffer(hcp_bytes, dtype=np.uint8)

        # Boardインスタンスを作成してHCPをデコード
        board = Board()
        board.set_hcp(hcp_array)

        # 盤面の駒配置を取得
        board_df = board.get_board_id_positions_df()
        board_id_positions = board_df["boardIdPositions"][0]

        # 持ち駒を取得
        black_hand, white_hand = board.get_pieces_in_hand()
        pieces_in_hand = black_hand + white_hand  # 14要素のリスト

        # レコードに追加
        record["boardIdPositions"] = board_id_positions
        record["piecesInHand"] = pieces_in_hand

        return record

    def _create_mock_record(
        self, record_id: str, row_number: int
    ) -> Dict[str, Any]:
        """モックレコードデータを生成．

        Args:
            record_id: レコードID
            row_number: 行番号

        Returns:
            モックレコードデータ
        """
        # 簡単なモック盤面データ
        mock_board = [[0 for _ in range(9)] for _ in range(9)]

        # いくつかの駒を配置（row_numberに応じて変化）
        mock_board[0][4] = 16 + 8  # 後手王
        mock_board[8][4] = 8  # 先手王

        # row_numberに応じて追加の駒
        if row_number % 3 == 0:
            mock_board[0][1] = 16 + 6  # 後手角
            mock_board[8][7] = 6  # 先手角

        if row_number % 2 == 0:
            mock_board[0][7] = 16 + 7  # 後手飛車
            mock_board[8][1] = 7  # 先手飛車

        mock_hand = [0] * 14
        # row_numberに応じて持ち駒
        mock_hand[0] = row_number % 5  # 先手の歩
        mock_hand[7] = (row_number + 1) % 4  # 後手の歩

        return {
            "id": record_id,
            "eval": (row_number % 2000) - 1000,
            "moves": 50 + (row_number % 50),
            "boardIdPositions": mock_board,
            "piecesInHand": mock_hand,
        }
