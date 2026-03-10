"""ゲームグラフの検索・フィルタリング."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import polars as pl


class GameGraphQuery:
    """グラフデータの検索・フィルタリング．

    構築済みのゲームグラフ(nodes/edges DataFrames)に対して，
    サブグラフ取得・ノード詳細・パス探索などのクエリを提供する．
    """

    def __init__(
        self, nodes_df: pl.DataFrame, edges_df: pl.DataFrame
    ) -> None:
        """初期化．

        Args:
            nodes_df: ノードデータ(nodes.feather相当)
            edges_df: エッジデータ(edges.feather相当)
        """
        self.nodes_df = nodes_df
        self.edges_df = edges_df
        # get_path_to_root のLRUキャッシュ(DataFrameは不変のため安全)
        # _get_detail_outputs で同一ノードに4回呼ばれるのを吸収する
        self._path_cache: OrderedDict[int, list[int]] = (
            OrderedDict()
        )
        self._path_cache_maxsize = 64

    def get_subgraph(
        self,
        root_hash: int,
        max_depth: int = 3,
        min_probability: float = 0.01,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """指定ノードを起点としたサブグラフを取得する．

        UIに送信するデータ量を制限するため，
        表示深さとエッジ確率閾値でフィルタリングする．

        Args:
            root_hash: ルートノードのposition_hash
            max_depth: 取得する最大深さ(ルートからの相対深さ)
            min_probability: エッジの最小確率閾値

        Returns:
            (nodes_df, edges_df) のフィルタ済みタプル
        """
        # BFSでroot_hashからmax_depth以内のノードを収集
        visited: set[int] = {root_hash}
        current_level: set[int] = {root_hash}
        collected_edges: list[pl.DataFrame] = []

        for _ in range(max_depth):
            if not current_level:
                break

            # 現在レベルのノードから出るエッジを取得
            level_edges = self.edges_df.filter(
                pl.col("parent_hash").is_in(list(current_level))
                & (pl.col("probability") >= min_probability)
            )

            if len(level_edges) == 0:
                break

            collected_edges.append(level_edges)

            # 次レベルのノードを収集(nodes_dfに存在するもののみ)
            child_hashes = set(
                level_edges["child_hash"].to_list()
            )
            existing_children = set(
                self.nodes_df.filter(
                    pl.col("position_hash").is_in(
                        list(child_hashes)
                    )
                )["position_hash"].to_list()
            )
            next_level = existing_children - visited
            visited.update(next_level)
            current_level = next_level

        # ノードをフィルタ
        subgraph_nodes = self.nodes_df.filter(
            pl.col("position_hash").is_in(list(visited))
        )

        # エッジを結合し，両端が訪問済みノードに含まれるもののみ保持
        if collected_edges:
            visited_list = list(visited)
            subgraph_edges = pl.concat(collected_edges).filter(
                pl.col("parent_hash").is_in(visited_list)
                & pl.col("child_hash").is_in(visited_list)
            )
        else:
            subgraph_edges = self.edges_df.head(0)

        return subgraph_nodes, subgraph_edges

    def get_node_detail(
        self, position_hash: int
    ) -> dict[str, Any]:
        """ノードの詳細情報を取得する．

        Args:
            position_hash: 対象ノードのZobrist hash

        Returns:
            ノード情報の辞書．見つからない場合は空辞書．
        """
        node = self.nodes_df.filter(
            pl.col("position_hash") == position_hash
        )
        if len(node) == 0:
            return {}

        row = node.row(0, named=True)
        return {
            "position_hash": row["position_hash"],
            "result_value": row["result_value"],
            "best_move_win_rate": row["best_move_win_rate"],
            "num_branches": row["num_branches"],
            "depth": row["depth"],
            "is_depth_cutoff": row["is_depth_cutoff"],
        }

    def get_children(self, position_hash: int) -> pl.DataFrame:
        """指定ノードの子エッジ一覧を確率降順で取得する．

        Args:
            position_hash: 対象ノードのZobrist hash

        Returns:
            子エッジのDataFrame(確率降順)
        """
        return self.edges_df.filter(
            pl.col("parent_hash") == position_hash
        ).sort("probability", descending=True)

    def get_path_to_root(self, position_hash: int) -> list[int]:
        """指定ノードから根までの最短パスを取得する．

        depthフィールドを利用して，depth が1ずつ減少する
        親エッジのみを辿ることで最短パスを効率的に取得する．

        結果はインスタンスレベルでキャッシュされる．
        同一 position_hash に対する2回目以降の呼び出しは
        キャッシュから即座に返す．

        Args:
            position_hash: 対象ノードのZobrist hash

        Returns:
            ルートから対象ノードまでのposition_hashリスト
        """
        if position_hash in self._path_cache:
            self._path_cache.move_to_end(position_hash)
            return self._path_cache[position_hash]

        # ノードが存在しない場合は空リストを返す
        initial_depth = self._get_depth(position_hash)
        if initial_depth is None:
            return []

        path = [position_hash]
        current = position_hash

        while True:
            node_depth = self._get_depth(current)
            if node_depth is None or node_depth == 0:
                break

            # depth が 1 小さい親を選択(複数ある場合は確率最大のエッジを優先)
            parent_edges = self.edges_df.filter(
                pl.col("child_hash") == current
            )
            if len(parent_edges) == 0:
                break

            parent_with_depth = parent_edges.join(
                self.nodes_df.select("position_hash", "depth"),
                left_on="parent_hash",
                right_on="position_hash",
            ).filter(pl.col("depth") == node_depth - 1)

            if len(parent_with_depth) == 0:
                break

            best_parent = parent_with_depth.sort(
                "probability", descending=True
            ).row(0, named=True)
            current = best_parent["parent_hash"]
            path.append(current)

        result = list(reversed(path))
        self._path_cache[position_hash] = result
        if len(self._path_cache) > self._path_cache_maxsize:
            self._path_cache.popitem(last=False)
        return result

    def get_edge_between(
        self, parent_hash: int, child_hash: int
    ) -> dict[str, Any] | None:
        """2ノード間のエッジ情報を取得する．

        Args:
            parent_hash: 親ノードのZobrist hash
            child_hash: 子ノードのZobrist hash

        Returns:
            エッジ情報の辞書．見つからない場合はNone．
        """
        edge = self.edges_df.filter(
            (pl.col("parent_hash") == parent_hash)
            & (pl.col("child_hash") == child_hash)
        )
        if len(edge) == 0:
            return None
        return edge.row(0, named=True)

    def _get_depth(self, position_hash: int) -> int | None:
        """ノードのdepthを取得する．

        Args:
            position_hash: 対象ノードのZobrist hash

        Returns:
            ノードのdepth．見つからない場合はNone．
        """
        node = self.nodes_df.filter(
            pl.col("position_hash") == position_hash
        )
        if len(node) == 0:
            return None
        return node["depth"][0]
