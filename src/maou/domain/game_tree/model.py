"""ゲームツリーのデータモデル定義."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameTreeNode:
    """ゲームツリーのノード(一意の局面)．

    Attributes:
        position_hash: Zobrist hash(局面の一意識別子)
        result_value: 局面の勝率(手番側視点，0.0〜1.0)
        best_move_win_rate: 最善手の勝率
        num_branches: 分岐数．通常ノードでは実際にエッジが生成された数．
            max_depth到達ノードではmin_probability以上の候補手数(未展開)
        depth: 初期局面からの最短距離
    """

    position_hash: int
    result_value: float
    best_move_win_rate: float
    num_branches: int  # UInt16範囲内(0〜65535)
    depth: int  # UInt16範囲内(0〜65535)


@dataclass(frozen=True)
class GameTreeEdge:
    """ゲームツリーのエッジ(局面間の遷移)．

    Attributes:
        parent_hash: 親局面のZobrist hash
        child_hash: 子局面のZobrist hash
        move16: cshogi move16形式の指し手
        move_label: moveLabelのインデックス(0〜1495)
        probability: moveLabel値(親局面からの相対出現確率)
        win_rate: moveWinRate値(この手の勝率)
        is_leaf: child_hashがpreprocessデータに存在しない場合True
    """

    parent_hash: int
    child_hash: int
    move16: int
    move_label: int
    probability: float
    win_rate: float
    is_leaf: bool
