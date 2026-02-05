"""データ検索ユースケース（アプリケーション層）．

検索インデックスとデータソースを組み合わせてデータを取得する．
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from maou.domain.data.stage1_generator import (
    Stage1DataGenerator,
)
from maou.infra.file_system.file_data_source import (
    FileDataSource,
)
from maou.infra.visualization.search_index import SearchIndex

logger = logging.getLogger(__name__)


class DataRetriever:
    """データ検索オーケストレータ．

    検索インデックスでレコード位置を特定し，データソースから実データを取得する．
    """

    # Stage1パターンのクラスレベルキャッシュ
    _stage1_patterns: Optional[List[Dict[str, Any]]] = None

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
            # インデックス時の評価値を取得
            indexed_eval = (
                self.search_index.get_eval_by_position(
                    file_index, row_number
                )
            )
            return self._create_mock_record(
                record_id, row_number, indexed_eval
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
                # インデックス時の評価値を取得
                indexed_eval = (
                    self.search_index.get_eval_by_position(
                        file_index, row_number
                    )
                )
                record = self._create_mock_record(
                    record_id, row_number, indexed_eval
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

        # 手番を取得（HCPから復元）
        turn = board.get_turn()

        # 盤面の駒配置を取得
        board_df = board.get_board_id_positions_df()
        board_id_positions = board_df["boardIdPositions"][0]

        # 持ち駒を取得
        black_hand, white_hand = board.get_pieces_in_hand()
        pieces_in_hand = (
            black_hand + white_hand
        )  # 14要素のリスト

        # レコードに追加
        record["boardIdPositions"] = board_id_positions
        record["piecesInHand"] = pieces_in_hand
        record["turn"] = (
            turn.value
        )  # Store turn as int (0=BLACK, 1=WHITE)

        return record

    def _create_mock_record(
        self,
        record_id: str,
        row_number: int,
        indexed_eval: Optional[int] = None,
    ) -> Dict[str, Any]:
        """モックレコードデータを生成するディスパッチャ．

        array_typeに応じて適切なモック生成メソッドを呼び出す．

        Args:
            record_id: レコードID
            row_number: 行番号
            indexed_eval: インデックス時の評価値（Noneの場合は計算で生成）

        Returns:
            モックレコードデータ

        Raises:
            ValueError: サポートされていないarray_typeの場合
        """
        if self.array_type == "hcpe":
            return self._create_hcpe_mock(
                record_id, row_number, indexed_eval
            )
        elif self.array_type == "stage1":
            return self._create_stage1_mock(
                record_id, row_number, indexed_eval
            )
        elif self.array_type == "stage2":
            return self._create_stage2_mock(
                record_id, row_number, indexed_eval
            )
        elif self.array_type == "preprocessing":
            return self._create_preprocessing_mock(
                record_id, row_number, indexed_eval
            )
        else:
            raise ValueError(
                f"Unsupported array type: {self.array_type}"
            )

    def _create_hcpe_mock(
        self,
        record_id: str,
        row_number: int,
        indexed_eval: Optional[int] = None,
    ) -> Dict[str, Any]:
        """HCPEフォーマットのモックレコードデータを生成．

        Args:
            record_id: レコードID
            row_number: 行番号
            indexed_eval: インデックス時の評価値（Noneの場合は計算で生成）

        Returns:
            HCPEフォーマットのモックレコードデータ
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

        # 評価値はインデックス時の値を使用（整合性のため）
        eval_value = (
            indexed_eval
            if indexed_eval is not None
            else (row_number % 2000) - 1000
        )

        # モック用の指し手を生成（7g7f = 58 | (67 << 7) = 8634）
        # 様々な指し手を生成
        mock_moves = [
            8634,  # 7g7f
            7609,  # 2g2f
            16482,  # 8h2b+
            7610,  # 3g3f
            8633,  # 6g6f
        ]
        best_move = mock_moves[row_number % len(mock_moves)]

        return {
            "id": record_id,
            "eval": eval_value,
            "moves": 50 + (row_number % 50),
            "boardIdPositions": mock_board,
            "piecesInHand": mock_hand,
            "bestMove16": best_move,
        }

    @classmethod
    def _get_stage1_patterns(cls) -> List[Dict[str, Any]]:
        """Stage1パターンを取得（キャッシュ付き）．

        Stage1DataGeneratorから全パターンを生成し，クラス変数としてキャッシュする．
        2回目以降の呼び出しではキャッシュを返す．

        Returns:
            Stage1パターンのリスト．各要素は以下のキーを持つ辞書:
                - board: 9x9盤面配列
                - hand: 14要素の持ち駒配列
                - reachable: 9x9到達可能マス配列
        """
        if cls._stage1_patterns is not None:
            return cls._stage1_patterns

        logger.info(
            "Generating Stage1 patterns from Stage1DataGenerator..."
        )

        patterns: List[Dict[str, Any]] = []

        # 盤上の駒パターンを生成
        for (
            board_pattern
        ) in Stage1DataGenerator.enumerate_board_patterns():
            record = Stage1DataGenerator._generate_record_from_board_pattern(
                board_pattern
            )
            patterns.append(
                {
                    "board": record["boardIdPositions"],
                    "hand": record["piecesInHand"],
                    "reachable": record["reachableSquares"],
                }
            )

        # 持ち駒パターンを生成
        for (
            hand_pattern
        ) in Stage1DataGenerator.enumerate_hand_patterns():
            record = Stage1DataGenerator._generate_record_from_hand_pattern(
                hand_pattern
            )
            patterns.append(
                {
                    "board": record["boardIdPositions"],
                    "hand": record["piecesInHand"],
                    "reachable": record["reachableSquares"],
                }
            )

        cls._stage1_patterns = patterns
        logger.info(
            f"Generated {len(patterns)} Stage1 patterns"
        )

        return patterns

    def _create_stage1_mock(
        self,
        record_id: str,
        row_number: int,
        indexed_eval: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Stage1フォーマットのモックレコードデータを生成．

        Stage1DataGeneratorから生成したパターンを使用して，
        到達可能マス付きのモックデータを生成する．

        Args:
            record_id: レコードID
            row_number: 行番号（パターン選択に使用）
            indexed_eval: インデックス時の評価値（Stage1では未使用）

        Returns:
            Stage1フォーマットのモックレコードデータ:
                - id: レコードID
                - boardIdPositions: 9x9盤面配列
                - piecesInHand: 14要素の持ち駒配列
                - reachableSquares: 9x9到達可能マス配列（0/1）
        """
        patterns = self._get_stage1_patterns()

        # row_number % パターン数 でパターンを選択
        pattern_index = row_number % len(patterns)
        pattern = patterns[pattern_index]

        return {
            "id": f"mock_stage1_{row_number}",
            "boardIdPositions": pattern["board"],
            "piecesInHand": pattern["hand"],
            "reachableSquares": pattern["reachable"],
        }

    def _create_midgame_board(
        self, row_number: int
    ) -> List[List[int]]:
        """中盤風の盤面を生成．

        駒が適度に交換された状態の盤面を生成する．
        row_numberに応じて盤面が変化する．

        Args:
            row_number: 行番号（盤面のバリエーション決定に使用）

        Returns:
            9x9の盤面配列（駒ID）
        """
        import random

        # シード固定で再現可能
        rng = random.Random(row_number)

        # 空の盤面を初期化
        board: List[List[int]] = [[0] * 9 for _ in range(9)]

        # 駒ID定義（cshogi互換）
        # 先手: 歩=1, 香=2, 桂=3, 銀=4, 金=5, 角=6, 飛=7, 玉=8
        # 成駒: と=9, 成香=10, 成桂=11, 成銀=12, 馬=13, 竜=14
        # 後手: +16

        # 両玉を配置（必須）
        # 後手玉: 上部で横に揺らぐ
        white_king_col = 4 + rng.randint(-2, 2)
        white_king_row = rng.randint(0, 2)
        board[white_king_row][white_king_col] = 8 + 16  # 後手玉

        # 先手玉: 下部で横に揺らぐ
        black_king_col = 4 + rng.randint(-2, 2)
        black_king_row = rng.randint(6, 8)
        board[black_king_row][black_king_col] = 8  # 先手玉

        # 金銀を玉の周りに配置
        # 先手金銀
        for _ in range(rng.randint(1, 3)):
            dr, dc = rng.choice(
                [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1)]
            )
            r, c = black_king_row + dr, black_king_col + dc
            if 0 <= r < 9 and 0 <= c < 9 and board[r][c] == 0:
                board[r][c] = rng.choice([4, 5])  # 銀or金

        # 後手金銀
        for _ in range(rng.randint(1, 3)):
            dr, dc = rng.choice(
                [(1, -1), (1, 0), (1, 1), (0, -1), (0, 1)]
            )
            r, c = white_king_row + dr, white_king_col + dc
            if 0 <= r < 9 and 0 <= c < 9 and board[r][c] == 0:
                board[r][c] = (
                    rng.choice([4, 5]) + 16
                )  # 後手銀or金

        # 大駒（角・飛車）をランダムに配置（成り駒含む）
        if rng.random() < 0.7:  # 70%の確率で先手角
            r, c = rng.randint(0, 8), rng.randint(0, 8)
            if board[r][c] == 0:
                piece = (
                    13 if (r < 3 and rng.random() < 0.5) else 6
                )  # 馬(13)or角(6)
                board[r][c] = piece

        if rng.random() < 0.7:  # 70%の確率で先手飛車
            r, c = rng.randint(0, 8), rng.randint(0, 8)
            if board[r][c] == 0:
                piece = (
                    14 if (r < 3 and rng.random() < 0.5) else 7
                )  # 竜(14)or飛(7)
                board[r][c] = piece

        if rng.random() < 0.6:  # 60%の確率で後手角
            r, c = rng.randint(0, 8), rng.randint(0, 8)
            if board[r][c] == 0:
                piece = (
                    13 if (r > 5 and rng.random() < 0.5) else 6
                ) + 16  # 後手馬(29)or角(22)
                board[r][c] = piece

        if rng.random() < 0.6:  # 60%の確率で後手飛車
            r, c = rng.randint(0, 8), rng.randint(0, 8)
            if board[r][c] == 0:
                piece = (
                    14 if (r > 5 and rng.random() < 0.5) else 7
                ) + 16  # 後手竜(30)or飛(23)
                board[r][c] = piece

        # 歩をいくつか配置
        for col in range(9):
            if rng.random() < 0.5:  # 50%の確率で先手歩
                row = rng.randint(2, 6)
                if board[row][col] == 0:
                    board[row][col] = 1
            if rng.random() < 0.5:  # 50%の確率で後手歩
                row = rng.randint(2, 6)
                if board[row][col] == 0:
                    board[row][col] = 1 + 16

        # と金をいくつか配置
        for _ in range(rng.randint(0, 2)):
            r, c = rng.randint(0, 2), rng.randint(0, 8)
            if board[r][c] == 0:
                board[r][c] = 9  # 先手と金

        for _ in range(rng.randint(0, 2)):
            r, c = rng.randint(6, 8), rng.randint(0, 8)
            if board[r][c] == 0:
                board[r][c] = 9 + 16  # 後手と金

        return board

    def _create_midgame_hand(
        self, row_number: int
    ) -> List[int]:
        """中盤風の持ち駒を生成．

        row_numberに応じて持ち駒が変化する．

        Args:
            row_number: 行番号（持ち駒のバリエーション決定に使用）

        Returns:
            14要素の配列（先手7種 + 後手7種）
            順序: 歩, 香, 桂, 銀, 金, 角, 飛（先手），歩, 香, 桂, 銀, 金, 角, 飛（後手）
        """
        import random

        # シード固定で再現可能
        rng = random.Random(
            row_number + 12345
        )  # boardとは違うシード

        # 持ち駒の最大枚数: 歩18, 香4, 桂4, 銀4, 金4, 角2, 飛2
        max_counts = [18, 4, 4, 4, 4, 2, 2]

        hand: List[int] = []

        # 先手持ち駒
        for max_count in max_counts:
            # 中盤なので0〜最大の半分程度
            count = rng.randint(0, max(1, max_count // 2))
            hand.append(count)

        # 後手持ち駒
        for max_count in max_counts:
            count = rng.randint(0, max(1, max_count // 2))
            hand.append(count)

        return hand

    def _create_stage2_mock(
        self,
        record_id: str,
        row_number: int,
        indexed_eval: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Stage2フォーマットのモックレコードデータを生成．

        中盤風盤面とrow_numberベースの合法手を生成する．

        Args:
            record_id: レコードID
            row_number: 行番号
            indexed_eval: インデックス時の評価値（Noneの場合は計算で生成）

        Returns:
            Stage2フォーマットのモックレコードデータ
        """
        import random

        # 合法手ラベルの総数
        move_labels_num = 2187

        # 中盤風盤面と持ち駒を生成
        board = self._create_midgame_board(row_number)
        hand = self._create_midgame_hand(row_number)

        # 合法手数: 20〜79手（row_numberに応じて変化）
        num_legal_moves = 20 + (row_number % 60)

        # シード固定で再現可能な合法手ラベルを生成
        rng = random.Random(row_number + 99999)

        # 合法手ラベル（2187要素のbinary配列）
        legal_moves_label: List[int] = [0] * move_labels_num

        # ランダムな位置にnum_legal_moves個の1を立てる
        legal_positions = rng.sample(
            range(move_labels_num), num_legal_moves
        )
        for pos in legal_positions:
            legal_moves_label[pos] = 1

        return {
            "id": f"mock_stage2_{row_number}",
            "boardIdPositions": board,
            "piecesInHand": hand,
            "legalMovesLabel": legal_moves_label,
        }

    def _create_endgame_board(
        self, row_number: int
    ) -> List[List[int]]:
        """終盤風の盤面を生成．

        駒が大幅に減った終盤の盤面を生成する．
        row_numberに応じて盤面が変化する．

        Args:
            row_number: 行番号（盤面のバリエーション決定に使用）

        Returns:
            9x9の盤面配列（駒ID）
        """
        import random

        # シード固定で再現可能（midgameと区別するため別オフセット）
        rng = random.Random(row_number + 77777)

        # 空の盤面を初期化
        board: List[List[int]] = [[0] * 9 for _ in range(9)]

        # 駒ID定義（cshogi互換）
        # 先手: 歩=1, 香=2, 桂=3, 銀=4, 金=5, 角=6, 飛=7, 玉=8
        # 成駒: と=9, 成香=10, 成桂=11, 成銀=12, 馬=13, 竜=14
        # 後手: +16

        # 両玉を配置（必須）
        # 終盤は玉が中央付近に逃げていることが多い
        # 後手玉: 上部〜中央で横に揺らぐ
        white_king_col = rng.randint(2, 6)
        white_king_row = rng.randint(0, 4)
        board[white_king_row][white_king_col] = 8 + 16  # 後手玉

        # 先手玉: 下部〜中央で横に揺らぐ
        black_king_col = rng.randint(2, 6)
        black_king_row = rng.randint(4, 8)
        # 先手玉と後手玉が同じ位置にならないようにする
        while (
            black_king_row == white_king_row
            and black_king_col == white_king_col
        ):
            black_king_row = rng.randint(4, 8)
            black_king_col = rng.randint(2, 6)
        board[black_king_row][black_king_col] = 8  # 先手玉

        # 終盤は金銀が少なく，玉の周りに1〜2枚程度
        # 先手金銀
        for _ in range(rng.randint(0, 2)):
            dr, dc = rng.choice(
                [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1)]
            )
            r, c = black_king_row + dr, black_king_col + dc
            if 0 <= r < 9 and 0 <= c < 9 and board[r][c] == 0:
                board[r][c] = rng.choice([4, 5])  # 銀or金

        # 後手金銀
        for _ in range(rng.randint(0, 2)):
            dr, dc = rng.choice(
                [(1, -1), (1, 0), (1, 1), (0, -1), (0, 1)]
            )
            r, c = white_king_row + dr, white_king_col + dc
            if 0 <= r < 9 and 0 <= c < 9 and board[r][c] == 0:
                board[r][c] = (
                    rng.choice([4, 5]) + 16
                )  # 後手銀or金

        # 終盤は大駒が成っていることが多い
        if rng.random() < 0.5:  # 50%の確率で先手馬/竜
            r, c = rng.randint(0, 8), rng.randint(0, 8)
            if board[r][c] == 0:
                board[r][c] = rng.choice(
                    [13, 14]
                )  # 馬(13)or竜(14)

        if rng.random() < 0.4:  # 40%の確率で後手馬/竜
            r, c = rng.randint(0, 8), rng.randint(0, 8)
            if board[r][c] == 0:
                board[r][c] = (
                    rng.choice([13, 14]) + 16
                )  # 後手馬(29)or竜(30)

        # 終盤は歩が少ない
        for col in range(9):
            if rng.random() < 0.25:  # 25%の確率で先手歩
                row = rng.randint(1, 5)
                if board[row][col] == 0:
                    board[row][col] = 1
            if rng.random() < 0.25:  # 25%の確率で後手歩
                row = rng.randint(3, 7)
                if board[row][col] == 0:
                    board[row][col] = 1 + 16

        # と金を多めに配置（終盤らしさ）
        for _ in range(rng.randint(1, 3)):
            r, c = rng.randint(0, 4), rng.randint(0, 8)
            if board[r][c] == 0:
                board[r][c] = 9  # 先手と金

        for _ in range(rng.randint(0, 2)):
            r, c = rng.randint(4, 8), rng.randint(0, 8)
            if board[r][c] == 0:
                board[r][c] = 9 + 16  # 後手と金

        return board

    def _create_endgame_hand(
        self, row_number: int
    ) -> List[int]:
        """終盤風の持ち駒を生成．

        終盤は駒が多く取られているため，持ち駒が多い状態を生成する．
        row_numberに応じて持ち駒が変化する．

        Args:
            row_number: 行番号（持ち駒のバリエーション決定に使用）

        Returns:
            14要素の配列（先手7種 + 後手7種）
            順序: 歩, 香, 桂, 銀, 金, 角, 飛（先手），歩, 香, 桂, 銀, 金, 角, 飛（後手）
        """
        import random

        # シード固定で再現可能（boardやmidgameとは違うシード）
        rng = random.Random(row_number + 88888)

        # 持ち駒の最大枚数: 歩18, 香4, 桂4, 銀4, 金4, 角2, 飛2
        max_counts = [18, 4, 4, 4, 4, 2, 2]

        hand: List[int] = []

        # 先手持ち駒（終盤なので多め）
        for max_count in max_counts:
            # 終盤なので半分〜最大程度
            count = rng.randint(max_count // 3, max_count)
            hand.append(count)

        # 後手持ち駒（終盤なので多め）
        for max_count in max_counts:
            count = rng.randint(max_count // 3, max_count)
            hand.append(count)

        return hand

    def _create_preprocessing_mock(
        self,
        record_id: str,
        row_number: int,
        indexed_eval: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Preprocessingフォーマットのモックレコードデータを生成．

        終盤風盤面とrow_numberベースの確率分布でモックデータを生成する．

        確率分布のパターン:
        - row_number % 3 == 0: 集中型（1手に80%以上）
        - row_number % 3 == 1: 分散型（上位3手に均等，約33%ずつ）
        - row_number % 3 == 2: 混合型（1手が50%，残り分散）

        Args:
            record_id: レコードID
            row_number: 行番号
            indexed_eval: インデックス時の評価値（Noneの場合は計算で生成）

        Returns:
            Preprocessingフォーマットのモックレコードデータ
        """
        import random

        # 合法手ラベルの総数
        move_labels_num = 2187

        # 終盤風盤面と持ち駒を生成
        board = self._create_endgame_board(row_number)
        hand = self._create_endgame_hand(row_number)

        # シード固定で再現可能
        rng = random.Random(row_number + 123456)

        # 確率分布を生成（2187要素，合計1.0）
        move_label: List[float] = [0.0] * move_labels_num

        pattern = row_number % 3

        if pattern == 0:
            # 集中型: 1手に80%以上
            main_pos = rng.randint(0, move_labels_num - 1)
            main_prob = 0.80 + rng.random() * 0.15  # 80%〜95%

            # 残りの確率を数手に分散
            remaining = 1.0 - main_prob
            num_others = rng.randint(3, 8)
            other_positions = rng.sample(
                [
                    i
                    for i in range(move_labels_num)
                    if i != main_pos
                ],
                num_others,
            )

            move_label[main_pos] = main_prob
            for pos in other_positions:
                # 残りを均等に分配
                move_label[pos] = remaining / num_others

        elif pattern == 1:
            # 分散型: 上位3手に均等（約33%ずつ）
            positions = rng.sample(range(move_labels_num), 3)
            base_prob = 1.0 / 3.0

            # 少しだけ揺らぎを加える
            probs = [
                base_prob + (rng.random() - 0.5) * 0.1
                for _ in range(3)
            ]
            # 合計を1.0に正規化
            total = sum(probs)
            probs = [p / total for p in probs]

            for pos, prob in zip(positions, probs):
                move_label[pos] = prob

        else:
            # 混合型: 1手が50%，残り分散
            main_pos = rng.randint(0, move_labels_num - 1)
            main_prob = 0.50

            # 残りの50%を複数手に分散
            remaining = 0.50
            num_others = rng.randint(5, 15)
            other_positions = rng.sample(
                [
                    i
                    for i in range(move_labels_num)
                    if i != main_pos
                ],
                num_others,
            )

            move_label[main_pos] = main_prob

            # 指数的に減衰する分布
            weights = [1.0 / (i + 1) for i in range(num_others)]
            total_weight = sum(weights)
            for pos, weight in zip(other_positions, weights):
                move_label[pos] = remaining * (
                    weight / total_weight
                )

        # resultValue: -1.0〜1.0の範囲
        result_value = (row_number % 201 - 100) / 100.0

        return {
            "id": f"mock_preprocessing_{row_number}",
            "boardIdPositions": board,
            "piecesInHand": hand,
            "moveLabel": move_label,
            "resultValue": result_value,
        }
