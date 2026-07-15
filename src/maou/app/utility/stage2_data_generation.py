"""Stage2データ生成ユースケース.

HCPE featherファイルからStage2(合法手学習)データを生成する．
Phase 1でHCPの重複排除を行い，Phase 2で合法手ラベルを生成する．
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from tqdm.auto import tqdm

from maou.domain.data.rust_io import load_hcpe_df

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stage2DataGenerationConfig:
    """Stage2データ生成の設定."""

    input_dir: Path
    output_dir: Path
    output_data_name: str = "stage2"
    chunk_size: int = 1_000_000
    cache_dir: Path | None = None


class Stage2DataGenerationUseCase:
    """Stage2学習データ生成ユースケース."""

    def execute(
        self, config: Stage2DataGenerationConfig
    ) -> dict[str, int | str | list[str]]:
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

        from maou._rust.maou_search import hcp_hashes

        # Collect feather files
        feather_files = sorted(input_dir.rglob("*.feather"))
        if not feather_files:
            raise FileNotFoundError(
                f"No .feather files found in {input_dir}"
            )

        conn = duckdb.connect(str(db_path))
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS unique_hcps (
                    hash_id UBIGINT PRIMARY KEY,
                    hcp BLOB NOT NULL
                )
                """
            )

            total_input = 0

            for file_path in tqdm(
                feather_files,
                desc=f"Phase 1: Collecting HCPs ({len(feather_files)} files)",
            ):
                df = load_hcpe_df(file_path)

                if "hcp" not in df.columns:
                    logger.warning(
                        f"Skipping {file_path.name}: no 'hcp' column"
                    )
                    continue

                hcp_bytes_list = df["hcp"].to_list()
                batch_size = len(hcp_bytes_list)
                total_input += batch_size

                # Rust 一括計算で全 HCP の zobrist hash を取得
                hcps = np.frombuffer(
                    b"".join(hcp_bytes_list), dtype=np.uint8
                ).reshape(-1, 32)
                hash_ids = hcp_hashes(hcps)

                # Insert into DuckDB with dedup (INSERT OR IGNORE)
                batch_df = pl.DataFrame(  # noqa: F841 (used by DuckDB)
                    {
                        "hash_id": pl.Series(
                            "hash_id",
                            hash_ids,
                            dtype=pl.UInt64,
                        ),
                        "hcp": pl.Series(
                            "hcp",
                            hcp_bytes_list,
                            dtype=pl.Binary,
                        ),
                    }
                )

                conn.execute(
                    """
                    INSERT OR IGNORE INTO unique_hcps
                    SELECT * FROM batch_df
                    """
                )

            row = conn.execute(
                "SELECT COUNT(*) FROM unique_hcps"
            ).fetchone()
            total_unique: int = row[0] if row is not None else 0
        finally:
            conn.close()

        tqdm.write(
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

        from maou._rust.maou_search import (
            encode_hcp_features,
            legal_move_masks,
        )
        from maou.domain.data.rust_io import save_stage2_df
        from maou.domain.data.schema import (
            get_stage2_polars_schema,
        )

        schema = get_stage2_polars_schema()

        conn = duckdb.connect(str(db_path))
        try:
            count_row = conn.execute(
                "SELECT COUNT(*) FROM unique_hcps"
            ).fetchone()
            total_count: int = (
                count_row[0] if count_row is not None else 0
            )

            output_files: list[Path] = []
            chunk_idx = 0
            offset = 0
            total_chunks = (
                total_count + chunk_size - 1
            ) // chunk_size

            with tqdm(
                total=total_chunks,
                desc="Phase 2: Generating labels",
            ) as pbar:
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

                    ids: list[int] = [
                        hash_id for hash_id, _ in rows
                    ]

                    # チャンク内の全 HCP を束ねて Rust 一括エンコード
                    # (特徴量 + 手番視点正規化済み合法手ラベルマスク)
                    hcps = np.frombuffer(
                        b"".join(
                            hcp_bytes for _, hcp_bytes in rows
                        ),
                        dtype=np.uint8,
                    ).reshape(-1, 32)
                    board_ids, hands = encode_hcp_features(hcps)
                    legal_masks = legal_move_masks(hcps)

                    board_id_positions_list: list[
                        list[list[int]]
                    ] = board_ids.tolist()
                    pieces_in_hand_list: list[list[int]] = (
                        hands.tolist()
                    )
                    legal_moves_labels_list: list[list[int]] = (
                        legal_masks.tolist()
                    )

                    chunk_df = pl.DataFrame(
                        {
                            "id": pl.Series(
                                "id",
                                ids,
                                dtype=pl.UInt64,
                            ),
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

                    pbar.update(1)

                    offset += chunk_size
                    chunk_idx += 1
        finally:
            conn.close()

        tqdm.write(
            f"Phase 2 complete: {len(output_files)} files written"
        )

        return output_files
