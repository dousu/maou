"""駒移動ラベルからUSI表記への変換サービス（ドメイン層）．

このモジュールは，駒移動ラベル（0-2186の整数）を人間が読める
USI形式の文字列（例: "7g7f", "P*5e"）に変換するドメインロジックを提供する．
"""

import logging
from typing import List, Tuple

from maou.domain.board.shogi import Board
from maou.domain.move.label import (
    IllegalMove,
    make_usi_move_from_label,
)

logger: logging.Logger = logging.getLogger(__name__)


class MoveLabelConverter:
    """駒移動ラベルをUSI表記に変換するサービス．

    このクラスは，ニューラルネットワークの出力である駒移動ラベル（整数）を，
    可視化のための人間が読める形式（USI文字列）に変換する純粋なドメインロジックを提供する．
    """

    def convert_label_to_usi(
        self, board: Board, label: int
    ) -> str:
        """単一の駒移動ラベルをUSI表記に変換する．

        Args:
            board: 現在の局面情報
            label: 駒移動ラベル（0-2186）

        Returns:
            USI形式の指し手文字列（例: "7g7f", "P*5e"）
            無効なラベルの場合は "<invalid:{label}>" 形式の文字列

        Example:
            >>> board = Board()
            >>> board.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
            >>> converter = MoveLabelConverter()
            >>> converter.convert_label_to_usi(board, 1234)
            "7g7f"
        """
        try:
            return make_usi_move_from_label(board, label)
        except (IllegalMove, ValueError) as e:
            logger.warning(
                f"Failed to convert label {label} to USI: {e}"
            )
            return f"<invalid:{label}>"

    def convert_labels_to_usi_list(
        self,
        board: Board,
        labels: List[int],
        limit: int = 10,
    ) -> List[str]:
        """複数の駒移動ラベルをUSI表記のリストに変換する．

        Stage2データ型の合法手表示に使用される．

        Args:
            board: 現在の局面情報
            labels: 駒移動ラベルのリスト
            limit: 変換する最大数（デフォルト: 10）

        Returns:
            USI形式の指し手文字列のリスト

        Example:
            >>> converter = MoveLabelConverter()
            >>> converter.convert_labels_to_usi_list(board, [100, 200, 300], limit=5)
            ["7g7f", "2g2f", "6i7h"]
        """
        limited_labels = labels[:limit]
        return [
            self.convert_label_to_usi(board, label)
            for label in limited_labels
        ]

    def convert_probability_labels_to_usi(
        self,
        board: Board,
        probabilities: List[float],
        threshold: float = 0.01,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """駒移動ラベルの確率分布を上位USI手のリストに変換する．

        Preprocessingデータ型の訓練データ表示に使用される．
        確率の高い手から順にUSI表記と確率のペアを返す．

        Args:
            board: 現在の局面情報
            probabilities: 駒移動ラベルの確率分布（MOVE_LABELS_NUM要素）
            threshold: 含める最小確率（デフォルト: 0.01 = 1%）
            top_k: 返す最大手数（デフォルト: 10）

        Returns:
            (USI文字列, 確率)のタプルのリスト，確率降順

        Example:
            >>> converter = MoveLabelConverter()
            >>> probs = [0.0] * 1496
            >>> probs[100] = 0.45
            >>> probs[200] = 0.30
            >>> probs[300] = 0.15
            >>> converter.convert_probability_labels_to_usi(board, probs, top_k=3)
            [("7g7f", 0.45), ("2g2f", 0.30), ("6i7h", 0.15)]
        """
        # 1. threshold以上の確率を持つラベルを抽出
        label_prob_pairs: List[Tuple[int, float]] = [
            (label, prob)
            for label, prob in enumerate(probabilities)
            if prob >= threshold
        ]

        # 2. 確率でソート（降順）
        label_prob_pairs.sort(key=lambda x: x[1], reverse=True)

        # 3. 上位top_k件を取得
        top_pairs = label_prob_pairs[:top_k]

        # 4. USI表記に変換
        result: List[Tuple[str, float]] = []
        for label, prob in top_pairs:
            usi = self.convert_label_to_usi(board, label)
            # 無効なラベルはスキップ
            if not usi.startswith("<invalid:"):
                result.append((usi, prob))

        return result
