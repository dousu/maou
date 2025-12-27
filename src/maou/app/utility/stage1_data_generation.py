"""Stage 1 data generation use case.

Orchestrates the generation of Stage 1 training data for piece movement learning.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from maou.domain.data.rust_io import save_stage1_df
from maou.domain.data.stage1_generator import (
    Stage1DataGenerator,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stage1DataGenerationConfig:
    """Configuration for Stage 1 data generation."""

    output_dir: Path


class Stage1DataGenerationUseCase:
    """Use case for generating Stage 1 training data."""

    def execute(
        self, config: Stage1DataGenerationConfig
    ) -> dict[str, int | str]:
        """Generate Stage 1 data and save to output directory.

        Args:
            config: Generation configuration

        Returns:
            Result dictionary with:
                - total_patterns: int
                - output_file: str
        """
        logger.info("Starting Stage 1 data generation...")

        # Generate data
        logger.info("Generating patterns...")
        df = Stage1DataGenerator.generate_all_stage1_data()
        total_patterns = len(df)

        logger.info(f"Generated {total_patterns} patterns")

        # Create output directory
        config.output_dir.mkdir(parents=True, exist_ok=True)

        # Save to .feather file
        output_file = config.output_dir / "stage1_data.feather"
        logger.info(f"Saving to {output_file}...")
        save_stage1_df(df, output_file)

        logger.info("Stage 1 data generation complete")

        return {
            "total_patterns": total_patterns,
            "output_file": str(output_file),
        }
