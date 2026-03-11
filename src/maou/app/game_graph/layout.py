"""ゲームグラフのレイアウト事前計算．

全ノードの (x, y) 座標をサーバー側で事前計算する．
depth ベースの Y 座標と確率順の X 座標配置により，
フロントエンドのレイアウトエンジンに依存せず正確な描画を実現する．
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict

import polars as pl


@dataclasses.dataclass
class GraphLayout:
    """事前計算されたグラフレイアウト．

    Attributes:
        node_positions: position_hash → (x, y) 座標のマッピング
        bounds: (min_x, min_y, max_x, max_y) の全体境界
    """

    node_positions: dict[int, tuple[float, float]]
    bounds: tuple[float, float, float, float]


class GameGraphLayoutService:
    """ゲームグラフのレイアウト計算サービス．

    depth ベースの Y 座標配置と，親の X 座標を中心とした
    兄弟ノードの均等配置を行う．

    アルゴリズム:
        1. Y座標 = depth × rank_spacing
        2. X座標: depth 順に処理し，各親の子を確率降順で配置
        3. グループ間の重なりを検出して押し広げる
    """

    def compute_layout(
        self,
        nodes_df: pl.DataFrame,
        edges_df: pl.DataFrame,
        root_hash: int,
        rank_spacing: float = 80.0,
        sibling_spacing: float = 60.0,
        min_node_spacing: float = 40.0,
    ) -> GraphLayout:
        """全ノードの座標を計算する．

        Args:
            nodes_df: ノードデータ
            edges_df: エッジデータ
            root_hash: ルートノードのposition_hash
            rank_spacing: depth 間の Y 方向間隔
            sibling_spacing: 兄弟ノード間の X 方向間隔
            min_node_spacing: 異なる親の子グループ間の最小間隔

        Returns:
            事前計算された GraphLayout
        """
        if len(nodes_df) == 0:
            return GraphLayout(
                node_positions={}, bounds=(0.0, 0.0, 0.0, 0.0)
            )

        # --- データ構造の構築 ---

        # ノードの depth マップ
        node_depths: dict[int, int] = {}
        for row in nodes_df.iter_rows(named=True):
            node_depths[row["position_hash"]] = int(
                row["depth"]
            )

        # depth → ノードリスト
        depth_nodes: dict[int, list[int]] = defaultdict(list)
        for h, d in node_depths.items():
            depth_nodes[d].append(h)

        # 親 → [(child_hash, probability)] (確率降順)
        parent_children: dict[int, list[tuple[int, float]]] = (
            defaultdict(list)
        )
        # 子 → primary parent (depth-1 で確率最大の親)
        child_primary_parent: dict[int, int] = {}
        # 子 → primary parent からの確率
        child_primary_prob: dict[int, float] = {}

        for row in edges_df.iter_rows(named=True):
            p_hash = row["parent_hash"]
            c_hash = row["child_hash"]
            prob = float(row["probability"])

            # ノードが nodes_df に存在しない場合はスキップ
            if (
                p_hash not in node_depths
                or c_hash not in node_depths
            ):
                continue

            # NOTE: バックエッジ(c_depth <= p_depth)も含めて追加する．
            # これにより子リストの X 座標計算に影響するが，
            # primary parent 選択は c_depth == p_depth + 1 の
            # フォワードエッジのみを対象とするため無限ループは発生しない．
            parent_children[p_hash].append((c_hash, prob))

            p_depth = node_depths.get(p_hash, -1)
            c_depth = node_depths.get(c_hash, -1)

            # primary parent: depth が 1 小さい親のうち確率最大
            if c_depth == p_depth + 1:
                current_prob = child_primary_prob.get(
                    c_hash, -1.0
                )
                if prob > current_prob:
                    child_primary_parent[c_hash] = p_hash
                    child_primary_prob[c_hash] = prob

        # 各親の子リストを確率降順でソート
        for p_hash in parent_children:
            parent_children[p_hash].sort(
                key=lambda item: item[1], reverse=True
            )

        # --- 座標計算 ---

        positions: dict[int, tuple[float, float]] = {}
        max_depth = (
            max(node_depths.values()) if node_depths else 0
        )

        # depth=0: ルートを (0, 0) に配置
        positions[root_hash] = (0.0, 0.0)

        # depth 0 の他のノードも配置(通常は root のみ)
        for h in depth_nodes.get(0, []):
            if h not in positions:
                positions[h] = (0.0, 0.0)

        # depth 1, 2, ... と順に処理
        for d in range(1, max_depth + 1):
            y = d * rank_spacing
            nodes_at_depth = depth_nodes.get(d, [])
            if not nodes_at_depth:
                continue

            # primary parent ごとにグループ化
            parent_groups: dict[int, list[int]] = defaultdict(
                list
            )
            orphans: list[int] = []

            for h in nodes_at_depth:
                pp = child_primary_parent.get(h)
                if pp is not None and pp in positions:
                    parent_groups[pp].append(h)
                else:
                    orphans.append(h)

            # 各親グループの子を確率降順でソートして配置
            # (parent_children の順序を使用)
            placed: list[tuple[int, float]] = []  # (hash, x)

            # 親の X 座標順にグループを処理
            sorted_parents = sorted(
                parent_groups.keys(),
                key=lambda p: positions[p][0],
            )

            for pp in sorted_parents:
                children_at_d = parent_groups[pp]
                # parent_children の順序(確率降順)に従ってソート
                child_order = {
                    c: i
                    for i, (c, _) in enumerate(
                        parent_children.get(pp, [])
                    )
                }
                children_at_d.sort(
                    key=lambda c: child_order.get(c, 999999)
                )

                parent_x = positions[pp][0]
                n = len(children_at_d)
                # 親の x を中心に均等配置
                start_x = (
                    parent_x - (n - 1) * sibling_spacing / 2
                )

                for i, c in enumerate(children_at_d):
                    x = start_x + i * sibling_spacing
                    placed.append((c, x))

            # orphans は右端に配置
            if placed:
                max_x = max(x for _, x in placed)
                for i, h in enumerate(orphans):
                    placed.append(
                        (
                            h,
                            max_x + (i + 1) * sibling_spacing,
                        )
                    )
            else:
                for i, h in enumerate(orphans):
                    placed.append((h, i * sibling_spacing))

            # X 座標順にソートして重なり解消
            placed.sort(key=lambda item: item[1])
            resolved = self._resolve_overlaps(
                placed, min_node_spacing
            )

            for h, x in resolved:
                positions[h] = (x, y)

        # positions に含まれていないノードの処理
        # (エッジがないか depth_nodes から漏れたノード)
        for h in node_depths:
            if h not in positions:
                d = node_depths[h]
                y = d * rank_spacing
                # 該当 depth の右端に配置
                existing_at_d = [
                    positions[n][0]
                    for n in depth_nodes.get(d, [])
                    if n in positions
                ]
                if existing_at_d:
                    x = max(existing_at_d) + sibling_spacing
                else:
                    x = 0.0
                positions[h] = (x, y)

        # --- 境界計算 ---
        if positions:
            xs = [p[0] for p in positions.values()]
            ys = [p[1] for p in positions.values()]
            bounds = (min(xs), min(ys), max(xs), max(ys))
        else:
            bounds = (0.0, 0.0, 0.0, 0.0)

        return GraphLayout(
            node_positions=positions, bounds=bounds
        )

    @staticmethod
    def _resolve_overlaps(
        placed: list[tuple[int, float]],
        min_spacing: float,
    ) -> list[tuple[int, float]]:
        """X 座標の重なりを解消する．

        隣接するノード間の間隔が min_spacing 未満の場合，
        右側のノードを押し広げる．

        Args:
            placed: [(hash, x)] のリスト(x 昇順)
            min_spacing: 最小間隔

        Returns:
            重なりを解消した [(hash, x)] のリスト
        """
        if len(placed) <= 1:
            return placed

        result = list(placed)
        for i in range(1, len(result)):
            h_prev, x_prev = result[i - 1]
            h_curr, x_curr = result[i]
            if x_curr - x_prev < min_spacing:
                result[i] = (h_curr, x_prev + min_spacing)

        return result
