"""Stage2データ生成ユースケース.

HCPE featherファイルからStage2(合法手学習)データを生成する．
Phase 1でHCPの重複排除を行い，Phase 2で合法手ラベルを生成する．
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import polars as pl

from maou.domain.data.rust_io import load_hcpe_df

if TYPE_CHECKING:
    from maou.domain.board import shogi

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stage2DataGenerationConfig:
    """Stage2データ生成の設定."""

    input_dir: Path
    output_dir: Path
    output_data_name: str = "stage2"
    chunk_size: int = 100_000
    cache_dir: Optional[Path] = None


class Stage2DataGenerationUseCase:
    """Stage2学習データ生成ユースケース."""

    def execute(
        self, config: Stage2DataGenerationConfig
    ) -> dict[str, int | str | list[str]]:
        from maou.domain.board import shogi  # noqa: F401

        logger.info("Starting Stage 2 data generation...")

        config.output_dir.mkdir(parents=True, exist_ok=True)

        cache_dir = config.cache_dir or Path(tempfile.mkdtemp())
        cache_dir.mkdir(parents=True, exist_ok=True)
        db_path = cache_dir / "stage2_dedup.duckdb"

        # Phase 1: Collect unique HCPs
        total_input, total_unique = self._collect_unique_hcps(
            config.input_dir, db_path
        )

        # Phase 2: Generate legal move labels and output
        output_files = self._generate_legal_moves_labels(
            db_path=db_path,
            output_dir=config.output_dir,
            output_data_name=config.output_data_name,
            chunk_size=config.chunk_size,
        )

        # Cleanup
        if db_path.exists():
            db_path.unlink()

        logger.info("Stage 2 data generation complete")

        return {
            "total_input_positions": total_input,
            "total_unique_positions": total_unique,
            "output_files": [str(f) for f in output_files],
        }

    def _collect_unique_hcps(
        self, input_dir: Path, db_path: Path
    ) -> tuple[int, int]:
        """Phase 1: HCPを収集し重複排除する.

        Args:
            input_dir: HCPE featherファイルのディレクトリ
            db_path: DuckDBデータベースパス

        Returns:
            (入力局面数, ユニーク局面数)のタプル
        """
        import duckdb

        from maou.domain.board import shogi

        logger.info(
            f"Phase 1: Collecting HCPs from {input_dir}"
        )

        # Collect feather files
        feather_files = sorted(input_dir.glob("*.feather"))
        if not feather_files:
            raise FileNotFoundError(
                f"No .feather files found in {input_dir}"
            )

        logger.info(f"Found {len(feather_files)} feather files")

        conn = duckdb.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS unique_hcps (
                hash_id UBIGINT PRIMARY KEY,
                hcp BLOB NOT NULL
            )
            """
        )

        board = shogi.Board()
        total_input = 0

        for file_path in feather_files:
            logger.info(f"Processing: {file_path.name}")
            df = load_hcpe_df(file_path)

            if "hcp" not in df.columns:
                logger.warning(
                    f"Skipping {file_path.name}: no 'hcp' column"
                )
                continue

            hcp_series = df["hcp"]
            batch_size = len(hcp_series)
            total_input += batch_size

            # Compute hash for each HCP
            hash_ids = np.empty(batch_size, dtype=np.uint64)
            hcp_bytes_list = []

            for i in range(batch_size):
                hcp_bytes = hcp_series[i]
                hcp_array = np.frombuffer(
                    hcp_bytes, dtype=np.uint8
                )
                board.set_hcp(hcp_array)
                hash_ids[i] = board.hash()
                hcp_bytes_list.append(hcp_bytes)

            # Insert into DuckDB with dedup (INSERT OR IGNORE)
            batch_df = pl.DataFrame(  # noqa: F841 (used by DuckDB)
                {
                    "hash_id": pl.Series(
                        "hash_id", hash_ids, dtype=pl.UInt64
                    ),
                    "hcp": pl.Series(
                        "hcp", hcp_bytes_list, dtype=pl.Binary
                    ),
                }
            )

            conn.execute(
                """
                INSERT OR IGNORE INTO unique_hcps
                SELECT * FROM batch_df
                """
            )

            logger.info(
                f"  Processed {batch_size} positions from {file_path.name}"
            )

        row = conn.execute(
            "SELECT COUNT(*) FROM unique_hcps"
        ).fetchone()
        total_unique: int = row[0] if row is not None else 0

        conn.close()

        logger.info(
            f"Phase 1 complete: {total_input} input -> {total_unique} unique positions"
        )

        return total_input, total_unique

    def _generate_legal_moves_labels(
        self,
        *,
        db_path: Path,
        output_dir: Path,
        output_data_name: str,
        chunk_size: int,
    ) -> list[Path]:
        """Phase 2: 合法手ラベルを生成し出力する.

        Args:
            db_path: DuckDBデータベースパス
            output_dir: 出力先ディレクトリ
            output_data_name: 出力ファイル名ベース
            chunk_size: チャンクサイズ

        Returns:
            出力ファイルパスのリスト
        """
        import duckdb

        from maou.app.pre_process.feature import (
            make_board_id_positions,
            make_pieces_in_hand,
        )
        from maou.domain.board import shogi
        from maou.domain.data.rust_io import save_stage2_df
        from maou.domain.data.schema import (
            get_stage2_polars_schema,
        )
        from maou.domain.move.label import (
            MOVE_LABELS_NUM,
            make_move_label,
        )

        logger.info("Phase 2: Generating legal move labels")

        conn = duckdb.connect(str(db_path))
        count_row = conn.execute(
            "SELECT COUNT(*) FROM unique_hcps"
        ).fetchone()
        total_count: int = (
            count_row[0] if count_row is not None else 0
        )

        board = shogi.Board()
        output_files: list[Path] = []
        chunk_idx = 0
        offset = 0

        while offset < total_count:
            # Read chunk from DuckDB
            rows = conn.execute(
                f"""
                SELECT hash_id, hcp FROM unique_hcps
                ORDER BY hash_id
                LIMIT {chunk_size} OFFSET {offset}
                """
            ).fetchall()

            if not rows:
                break

            ids: list[int] = []
            board_id_positions_list: list[list[list[int]]] = []
            pieces_in_hand_list: list[list[int]] = []
            legal_moves_labels_list: list[list[int]] = []

            for hash_id, hcp_bytes in rows:
                hcp_array = np.frombuffer(
                    hcp_bytes, dtype=np.uint8
                )
                board.set_hcp(hcp_array)

                # Generate features
                board_positions = make_board_id_positions(board)
                pieces_in_hand = make_pieces_in_hand(board)

                # Generate legal move labels
                # 盤面は先手視点に正規化済みなので，
                # 正規化後の盤面の合法手からラベルを生成する
                legal_labels = np.zeros(
                    MOVE_LABELS_NUM, dtype=np.uint8
                )
                if board.get_turn() == shogi.Turn.BLACK:
                    # 先手番: 正規化なし，元のboardの合法手をそのまま使用
                    for move in board.get_legal_moves():
                        label = make_move_label(
                            shogi.Turn.BLACK, move
                        )
                        legal_labels[label] = 1
                else:
                    # 後手番: 盤面が180度回転されているため，
                    # 正規化後の盤面を再構築して合法手を取得
                    normalized_board = (
                        self._reconstruct_normalized_board(
                            board_positions,
                            pieces_in_hand,
                        )
                    )
                    for (
                        move
                    ) in normalized_board.get_legal_moves():
                        label = make_move_label(
                            shogi.Turn.BLACK, move
                        )
                        legal_labels[label] = 1

                ids.append(hash_id)
                board_id_positions_list.append(
                    board_positions.tolist()
                )
                pieces_in_hand_list.append(
                    pieces_in_hand.tolist()
                )
                legal_moves_labels_list.append(
                    legal_labels.tolist()
                )

            # Create Polars DataFrame with correct schema
            schema = get_stage2_polars_schema()
            chunk_df = pl.DataFrame(
                {
                    "id": pl.Series("id", ids, dtype=pl.UInt64),
                    "boardIdPositions": board_id_positions_list,
                    "piecesInHand": pieces_in_hand_list,
                    "legalMovesLabel": legal_moves_labels_list,
                },
                schema=schema,
            )

            # Determine filename
            if total_count <= chunk_size:
                filename = f"{output_data_name}.feather"
            else:
                filename = f"{output_data_name}_chunk{chunk_idx:04d}.feather"

            output_path = output_dir / filename
            save_stage2_df(chunk_df, output_path)
            output_files.append(output_path)

            logger.info(
                f"  Wrote chunk {chunk_idx}: {len(rows)} positions -> {output_path}"
            )

            offset += chunk_size
            chunk_idx += 1

        conn.close()

        logger.info(
            f"Phase 2 complete: {len(output_files)} files written"
        )

        return output_files

    @staticmethod
    def _reconstruct_normalized_board(
        board_positions: np.ndarray,
        pieces_in_hand: np.ndarray,
    ) -> shogi.Board:
        """正規化済み盤面からBoardを再構築する．

        make_board_id_positions()で先手視点に正規化された盤面と
        make_pieces_in_hand()の持ち駒から，Boardを再構築する．
        再構築されたBoardはturn=BLACKとなる．

        domain PieceIdをSFEN形式に変換し，set_sfenで構築する．

        Args:
            board_positions: 正規化済み9x9盤面配列(domain PieceId)
            pieces_in_hand: 正規化済み持ち駒配列(14要素)

        Returns:
            再構築されたBoardインスタンス(turn=BLACK)
        """
        from maou.domain.board import shogi as shogi_module

        # domain PieceId → SFEN文字マッピング
        _BLACK_PIECE_TO_SFEN = {
            1: "P",
            2: "L",
            3: "N",
            4: "S",
            5: "G",
            6: "B",
            7: "R",
            8: "K",
            9: "+P",
            10: "+L",
            11: "+N",
            12: "+S",
            13: "+B",
            14: "+R",
        }
        _DOMAIN_WHITE_MIN = 15
        _DOMAIN_WHITE_OFFSET = 14

        # 盤面をSFEN形式に変換
        ranks = []
        for row in board_positions:
            # col=0が1筋，SFENは9筋→1筋の順なので反転
            reversed_row = list(reversed(row))
            rank_str = ""
            empty_count = 0
            for piece_id in reversed_row:
                pid = int(piece_id)
                if pid == 0:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        rank_str += str(empty_count)
                        empty_count = 0
                    if pid >= _DOMAIN_WHITE_MIN:
                        # 後手駒: 小文字
                        black_char = _BLACK_PIECE_TO_SFEN.get(
                            pid - _DOMAIN_WHITE_OFFSET, ""
                        )
                        rank_str += black_char.lower()
                    else:
                        # 先手駒: 大文字
                        rank_str += _BLACK_PIECE_TO_SFEN.get(
                            pid, ""
                        )
            if empty_count > 0:
                rank_str += str(empty_count)
            ranks.append(rank_str if rank_str else "9")

        board_sfen = "/".join(ranks)

        # 持ち駒をSFEN形式に変換
        piece_chars = ["P", "L", "N", "S", "G", "B", "R"]
        hand_parts: list[str] = []

        pih = pieces_in_hand.tolist()
        # 先手の持ち駒 (pih[0:7])
        for i, count in enumerate(pih[:7]):
            if count > 0:
                char = piece_chars[i]
                if count > 1:
                    hand_parts.append(f"{count}{char}")
                else:
                    hand_parts.append(char)
        # 後手の持ち駒 (pih[7:14])
        for i, count in enumerate(pih[7:14]):
            if count > 0:
                char = piece_chars[i].lower()
                if count > 1:
                    hand_parts.append(f"{count}{char}")
                else:
                    hand_parts.append(char)

        hand_sfen = "".join(hand_parts) if hand_parts else "-"
        sfen = f"{board_sfen} b {hand_sfen} 1"

        board = shogi_module.Board()
        board.set_sfen(sfen)
        return board
