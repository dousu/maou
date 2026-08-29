"""Gradio UIサーバー実装(インフラ層)．

将棋データ可視化のためのGradio Webインターフェースを提供する．
"""

import json
import logging
import os
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

# Fix matplotlib backend for Google Colab compatibility
# matplotlib reads MPLBACKEND during import, so we must fix it before importing
_saved_mplbackend = os.environ.get("MPLBACKEND")
if _saved_mplbackend and "inline" in _saved_mplbackend:
    os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use(
    "Agg", force=True
)  # Ensure non-interactive backend

# Restore environment to avoid affecting other Colab cells
# matplotlib is now cached, so other imports will reuse this instance
if _saved_mplbackend is not None:
    os.environ["MPLBACKEND"] = _saved_mplbackend
elif "MPLBACKEND" in os.environ:
    del os.environ["MPLBACKEND"]

# 以降の import は **すべて上の matplotlib.use("Agg", force=True) より後**に
# 置く必要がある (間接的に matplotlib を import するものがあり，先に読むと
# inline backend が選ばれてヘッドレス環境で描画が壊れる)．
# 整形ツールで先頭へ移動させないこと．
# 2026-08-04 まで各行に E402 の抑止コメントが付いていたが，ruff の既定集合は
# E402 を含まないため意味を失い削除した．制約自体は残っている．
import gradio as gr

from maou.infra.file_system.file_system import (
    FileSystem,
)
from maou.infra.visualization.game_graph_shared import (
    FONT_LINKS,
    WORKBENCH_LANES,
    as_float,
    as_int,
    build_graph_html,
    build_workbench_head,
    make_workbench_bridge,
    workbench_js_on_load,
)
from maou.infra.visualization.indexing_state import (
    IndexingState,
)
from maou.infra.visualization.search_index import (
    SearchIndex,
)
from maou.interface.path_suggestions import (
    PathSuggestionService,
)
from maou.interface.visualization import (
    BoardPosition,
    SVGBoardRenderer,
    VisualizationInterface,
)
from maou.interface.visualize_workbench import (
    GraphData,
    RecordData,
    StatusView,
    WorkbenchState,
    render_workbench,
)

logger = logging.getLogger(__name__)


def _load_custom_css() -> str:
    """カスタムCSSファイルを読み込む．

    Returns:
        str: 結合されたCSS文字列
    """
    # 画面はワークベンチ 1 枚なので，Gradio コンポーネント向けの
    # theme.css / components.css は読み込まない (旧 UI の遺物であり，
    # 残すとワークベンチのトークンと競合する)．ゲームグラフの
    # game_graph.css は _build_head_scripts が head に注入する．
    static_dir = Path(__file__).parent / "static"
    css_files = ["visualize_workbench.css"]

    css_parts = []
    for css_file in css_files:
        css_path = static_dir / css_file
        if css_path.exists():
            css_parts.append(
                css_path.read_text(encoding="utf-8")
            )
        else:
            logger.warning("CSS file not found: %s", css_path)

    return "\n\n".join(css_parts)


class GradioVisualizationServer:
    """Gradio可視化サーバークラス．

    将棋データの検索と視覚化のためのWebインターフェースを提供する．
    """

    def __init__(
        self,
        file_paths: list[Path],
        array_type: str,
        use_mock_data: bool = False,
    ) -> None:
        """サーバーを初期化．

        Args:
            file_paths: データファイルのパスリスト
            array_type: データ型（hcpe, preprocessing, stage1, stage2）
            use_mock_data: Trueの場合はモックデータを使用
        """
        self.file_paths = file_paths
        self.array_type = (
            array_type  # This can now be changed dynamically
        )
        self.use_mock_data = use_mock_data
        self.renderer = SVGBoardRenderer()

        # Check if data is available
        self.has_data = len(file_paths) > 0 or use_mock_data

        # ゲームグラフ状態(game-graph モード時に使用)
        self._game_graph_viz: Any = None
        self._game_graph_root_hash: int = 0
        self._game_graph_layout: Any = None
        self._game_graph_spatial_buckets: dict[
            tuple[int, int], list[int]
        ] = {}
        self._game_graph_bucket_size: float = 500.0

        # 評価値検索をサポートするかどうかを判定
        self.supports_eval_search = self._supports_eval_search()

        # ワークベンチ再描画の連番．JS 側 (MutationObserver) が
        # data-render の変化で「差し替わった」ことを検出する．
        self._render_seq = 0
        # データセット統計は「更新」を押したときだけ取りに行き，
        # 読み込み・再構築でクリアする (毎描画で数え直さない)．
        self._stats_cache: dict[str, Any] = {}

        # Initialize path suggestion service
        self.path_suggester = PathSuggestionService(
            cache_ttl=60
        )

        # Initialize threading infrastructure
        self.indexing_state = IndexingState()
        self._index_lock = threading.Lock()
        self._indexing_thread: threading.Thread | None = None

        if self.has_data and array_type == "game-graph":
            # ゲームグラフ: 直接読み込み(インデックス不要)
            self.search_index = None  # type: ignore[assignment]
            self.viz_interface = None  # type: ignore[assignment]
            try:
                self._load_game_graph_data(file_paths[0])
                logger.info(
                    f"✅ Game graph loaded: root={self._game_graph_root_hash:#018x}"
                )
            except Exception:
                logger.exception(
                    "Failed to load game graph data"
                )
        elif self.has_data:
            # Start background indexing instead of blocking
            logger.info(
                f"🎯 Starting background indexing: "
                f"{len(file_paths)} files, type={array_type}"
            )

            # Initialize with None - will be set by background thread
            self.search_index = None  # type: ignore[assignment]
            self.viz_interface = None  # type: ignore[assignment]

            # Start background indexing
            self.indexing_state.set_indexing(
                total_files=len(file_paths),
                initial_message="開始中...",
            )
            self._indexing_thread = threading.Thread(
                target=self._build_index_background,
                args=(file_paths, array_type, use_mock_data),
                daemon=True,
            )
            self._indexing_thread.start()

            mode_msg = (
                "MOCK MODE (fake data)"
                if use_mock_data
                else "REAL MODE (actual data)"
            )
            logger.info(
                f"⚡ Background indexing started: {mode_msg}"
            )
        else:
            # Empty state - will be initialized when user loads data
            self.search_index = None  # type: ignore[assignment]
            self.viz_interface = None  # type: ignore[assignment]
            logger.warning(
                "⚠️  No data loaded - UI will show empty state"
            )

    def _build_index_background(
        self,
        file_paths: list[Path],
        array_type: str,
        use_mock_data: bool,
    ) -> None:
        """バックグラウンドでインデックスを構築．

        Args:
            file_paths: データファイルのパスリスト
            array_type: データ型
            use_mock_data: Trueの場合はモックデータを使用
        """
        try:
            logger.info("🔄 Background indexing started")

            # Progress callback to update IndexingState
            def progress_callback(
                files_done: int, records: int, message: str
            ) -> None:
                # Check for cancellation
                if self.indexing_state.is_cancelled():
                    raise InterruptedError(
                        "Indexing cancelled by user"
                    )

                self.indexing_state.update_progress(
                    files_done, records, message
                )

            # Build search index with progress tracking
            new_index = SearchIndex.build(
                file_paths=file_paths,
                array_type=array_type,
                use_mock_data=use_mock_data,
                num_mock_records=1000,
                progress_callback=progress_callback,
            )

            # Create visualization interface
            new_viz_interface = VisualizationInterface(
                search_index=new_index,
                file_paths=file_paths,
                array_type=array_type,
            )

            # Atomically update state with lock
            with self._index_lock:
                if not self.indexing_state.is_cancelled():
                    self.search_index = new_index
                    self.viz_interface = new_viz_interface
                    self.indexing_state.set_ready(
                        new_index.total_records()
                    )

                    logger.info(
                        f"✅ Background indexing completed: "
                        f"{new_index.total_records():,} records"
                    )
                else:
                    logger.info(
                        "🚫 Background indexing cancelled"
                    )

        except InterruptedError as e:
            logger.info("Indexing interrupted: %s", e)
            self.indexing_state.set_failed(
                "インデックス作成がキャンセルされました"
            )
        except Exception as e:
            logger.exception("❌ Background indexing failed")
            self.indexing_state.set_failed(str(e))

    def _check_indexing_status(
        self,
    ) -> tuple[str, gr.Button, gr.Button, str]:
        """インデックス作成状態をポーリングしてUI更新を返す．

        ローディングスピナーと推定残り時間を含むステータスメッセージ，
        ボタンの有効/無効状態，モードバッジを返す．

        Returns:
            (status_message, load_btn, rebuild_btn, mode_badge)のタプル
        """
        status = self.indexing_state.get_status()

        if status == "indexing":
            progress = self.indexing_state.get_progress()

            # 推定残り時間を計算
            remaining_seconds = (
                self.indexing_state.estimate_remaining_time()
            )
            time_str = ""
            if remaining_seconds is not None:
                if remaining_seconds < 60:
                    time_str = f" - 約{remaining_seconds}秒残り"
                else:
                    minutes = remaining_seconds // 60
                    seconds = remaining_seconds % 60
                    time_str = (
                        f" - 約{minutes}分{seconds}秒残り"
                    )

            # Loading spinner HTML (inline CSS animation)
            spinner_html = """
<div style="display: inline-block; vertical-align: middle; margin-right: 8px;">
    <div style="display: inline-block; width: 16px; height: 16px;
                border: 2px solid #f3f3f3; border-top: 2px solid #ff9800;
                border-radius: 50%; animation: spin-anim 1s linear infinite;"></div>
</div>
<style>
@keyframes spin-anim {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>
"""

            status_msg = (
                f"{spinner_html}🟡 **Indexing:** {progress['message']} "
                f"({progress['files']}/{progress['total_files']} files, "
                f"{progress['records']:,} records){time_str}"
            )

            return (
                status_msg,
                gr.Button(interactive=False),  # Load button
                gr.Button(interactive=False),  # Rebuild button
                '<span class="mode-badge-text">🟡 INDEXING</span>',
            )
        elif status == "ready":
            # Thread-safe access to search_index
            with self._index_lock:
                if self.search_index is not None:
                    total = self.search_index.total_records()
                else:
                    total = 0

            # Build path info string
            if len(self.file_paths) == 1:
                path_info = str(self.file_paths[0])
            elif len(self.file_paths) > 1:
                path_info = (
                    f"{self.file_paths[0].parent}/ "
                    f"({len(self.file_paths)} files)"
                )
            else:
                path_info = "N/A"

            status_msg = (
                f"🟢 **Ready:** {total:,} records loaded\n"
                f"- **Type:** {self.array_type}\n"
                f"- **Path:** {path_info}"
            )

            # モックモード時は MOCK MODE バッジを表示
            if self.use_mock_data:
                badge = '<span class="mode-badge-text">🔴 MOCK MODE</span>'
            else:
                badge = '<span class="mode-badge-text">🟢 REAL MODE</span>'

            return (
                status_msg,
                gr.Button(interactive=True),
                gr.Button(interactive=True),
                badge,
            )
        elif status == "failed":
            error = self.indexing_state.get_error()
            return (
                f"❌ **Error:** {error}",
                gr.Button(interactive=True),
                gr.Button(interactive=False),
                '<span class="mode-badge-text">⚪ ERROR</span>',
            )
        else:  # idle
            return (
                "⚪ **No data loaded**",
                gr.Button(interactive=True),
                gr.Button(interactive=False),
                '<span class="mode-badge-text">⚪ NO DATA</span>',
            )

    def _supports_eval_search(self) -> bool:
        """評価値範囲検索をサポートするデータ型かどうかを判定．

        Returns:
            bool: hcpeの場合はTrue，それ以外はFalse
        """
        return self.array_type == "hcpe"

    def _resolve_directory(self, dir_path: str) -> list[Path]:
        """Resolve directory to list of .feather files．

        Args:
            dir_path: Directory path string from UI input

        Returns:
            list of .feather file paths sorted by name

        Raises:
            ValueError: If directory not found, empty, or not a directory
        """
        if not dir_path or not dir_path.strip():
            raise ValueError("Directory path is required")

        path = Path(dir_path.strip()).expanduser()

        if not path.exists():
            raise ValueError(f"Directory not found: {path}")

        if not path.is_dir():
            raise ValueError(f"Not a directory: {path}")

        feather_files = FileSystem.collect_files(
            path, ext=".feather"
        )

        if not feather_files:
            raise ValueError(
                f"No .feather files found in {path}"
            )

        logger.info(
            "Found %d .feather files in %s",
            len(feather_files),
            path,
        )
        return sorted(feather_files)

    def _resolve_file_list(self, files_str: str) -> list[Path]:
        """Resolve comma-separated file paths．

        Args:
            files_str: Comma-separated file paths from UI input

        Returns:
            list of validated .feather file paths

        Raises:
            ValueError: If files not found or not .feather format
        """
        if not files_str or not files_str.strip():
            raise ValueError("File paths are required")

        # Split by comma and clean up whitespace
        path_strs = [
            f.strip() for f in files_str.split(",") if f.strip()
        ]
        paths = [Path(p).expanduser() for p in path_strs]

        # Check for missing files
        missing = [p for p in paths if not p.exists()]
        if missing:
            missing_str = ", ".join(str(p) for p in missing)
            raise ValueError(f"Files not found: {missing_str}")

        # Check for non-.feather files
        invalid = [p for p in paths if p.suffix != ".feather"]
        if invalid:
            invalid_str = ", ".join(str(p) for p in invalid)
            raise ValueError(
                f"Not .feather files: {invalid_str}"
            )

        logger.info(
            f"Resolved {len(paths)} .feather files from file list"
        )
        return paths

    def _load_new_data_source(
        self,
        source_mode: str,
        dir_path: str,
        files_path: str,
        array_type: str,
    ) -> tuple[str, bool, str, Any, Any, Any]:
        """Load new data source and rebuild index in background．

        Args:
            source_mode: "Directory" or "File list"
            dir_path: Directory path (used if source_mode == "Directory")
            files_path: Comma-separated files (used if source_mode == "File list")
            array_type: Data array type

        Returns:
            tuple of (status_message, rebuild_btn_enabled, mode_badge,
                       timer_update, record_panel_visible, game_graph_panel_visible)
        """
        is_game_graph = array_type == "game-graph"

        # Step 1: Validate and resolve paths
        try:
            if is_game_graph:
                # game-graph: ディレクトリパスをそのまま使用
                graph_dir = (
                    Path(dir_path)
                    if source_mode == "Directory"
                    else Path(files_path)
                )
                if not graph_dir.is_dir():
                    raise ValueError(
                        f"ディレクトリが存在しません: {graph_dir}"
                    )
                file_paths = [graph_dir]
            elif source_mode == "Directory":
                file_paths = self._resolve_directory(dir_path)
            else:  # "File list"
                file_paths = self._resolve_file_list(files_path)
        except ValueError as e:
            logger.error("Path resolution failed: %s", e)
            return (
                f"❌ **Error:** {e}",
                False,
                '<span class="mode-badge-text">⚪ NO DATA</span>',
                gr.update(),
                gr.update(),
                gr.update(),
            )

        # Step 2: Cancel any ongoing indexing
        if self.indexing_state.is_indexing():
            logger.info(
                "Cancelling ongoing indexing before loading new data source"
            )
            self.indexing_state.cancel()
            if (
                self._indexing_thread is not None
                and self._indexing_thread.is_alive()
            ):
                self._indexing_thread.join(timeout=5.0)
                if self._indexing_thread.is_alive():
                    logger.warning(
                        "Previous indexing thread did not terminate in time"
                    )

        # Step 3: Update file paths and array type
        self.file_paths = file_paths
        self.array_type = array_type
        self.has_data = True
        self.supports_eval_search = self._supports_eval_search()

        # Panel visibility
        record_visible = gr.update(visible=not is_game_graph)
        graph_visible = gr.update(visible=is_game_graph)

        # game-graph: グラフデータを直接読み込む(インデックス不要)
        if is_game_graph:
            try:
                self._load_game_graph_data(file_paths[0])
                return (
                    f"✅ **Game Graph loaded:** "
                    f"{self._game_graph_root_hash:#018x}",
                    False,
                    '<span class="mode-badge-text">🟢 GAME TREE</span>',
                    gr.update(),
                    record_visible,
                    graph_visible,
                )
            except Exception as e:
                logger.exception("Failed to load game graph")
                return (
                    f"❌ **Error:** {e}",
                    False,
                    '<span class="mode-badge-text">⚪ NO DATA</span>',
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )

        # Step 4: Start new background indexing (record types)
        logger.info(
            f"Starting background indexing for {len(file_paths)} files..."
        )

        self.indexing_state.set_indexing(
            total_files=len(file_paths),
            initial_message="開始中...",
        )

        self._indexing_thread = threading.Thread(
            target=self._build_index_background,
            args=(file_paths, array_type, False),
            daemon=True,
        )
        self._indexing_thread.start()

        # Step 5: Return immediate response (indexing continues in background)
        return (
            f"🟡 **Indexing:** Started for {len(file_paths)} file(s)",
            False,
            '<span class="mode-badge-text">🟡 INDEXING</span>',
            gr.Timer(value=2.0, active=True),
            record_visible,
            graph_visible,
        )

    def _load_game_graph_data(self, graph_dir: Path) -> None:
        """ゲームグラフデータを読み込む．

        Args:
            graph_dir: グラフデータディレクトリ
        """
        from maou.interface.game_graph_io import GameGraphIO
        from maou.interface.game_graph_visualization import (
            GameGraphVisualizationInterface,
        )

        io = GameGraphIO()
        nodes_df, edges_df = io.load(graph_dir)
        metadata = io.load_metadata(graph_dir)
        logger.info(
            "Loaded game graph: %d nodes, %d edges",
            len(nodes_df),
            len(edges_df),
        )

        self._game_graph_viz = GameGraphVisualizationInterface(
            nodes_df,
            edges_df,
            initial_sfen=metadata.get("initial_sfen"),
        )
        self._game_graph_root_hash = (
            self._game_graph_viz.get_root_hash()
        )

        # レイアウト事前計算
        self._game_graph_layout = (
            self._game_graph_viz.compute_layout()
        )

        # ビューポートクエリ用の空間インデックス
        bucket_size = self._game_graph_bucket_size
        buckets: dict[tuple[int, int], list[int]] = defaultdict(
            list
        )
        for h, (
            x,
            y,
        ) in self._game_graph_layout.node_positions.items():
            bx = int(x // bucket_size)
            by = int(y // bucket_size)
            buckets[(bx, by)].append(h)
        self._game_graph_spatial_buckets = buckets

    def _rebuild_index(self) -> tuple[str, bool, str, Any]:
        """Rebuild search index from current file paths in background．

        Returns:
            tuple of (status_message, rebuild_btn_enabled, mode_badge, timer_update)
        """
        if not self.has_data or not self.file_paths:
            logger.warning(
                "Rebuild requested but no data source is loaded"
            )
            return (
                "❌ **Error:** No data source loaded",
                False,
                '<span class="mode-badge-text">⚪ NO DATA</span>',
                gr.update(),
            )

        # Cancel any ongoing indexing
        if self.indexing_state.is_indexing():
            logger.info(
                "Cancelling ongoing indexing before rebuilding"
            )
            self.indexing_state.cancel()
            if (
                self._indexing_thread is not None
                and self._indexing_thread.is_alive()
            ):
                self._indexing_thread.join(timeout=5.0)

        # Build path info string for status message
        if len(self.file_paths) == 1:
            path_info = str(self.file_paths[0])
        else:
            path_info = (
                f"{self.file_paths[0].parent}/ "
                f"({len(self.file_paths)} files)"
            )

        # Start background indexing
        logger.info(
            f"Starting background rebuild for {len(self.file_paths)} files..."
        )

        self.indexing_state.set_indexing(
            total_files=len(self.file_paths),
            initial_message="再構築中...",
        )

        self._indexing_thread = threading.Thread(
            target=self._build_index_background,
            args=(self.file_paths, self.array_type, False),
            daemon=True,
        )
        self._indexing_thread.start()

        status_msg = (
            f"🟡 **Rebuilding Index**\n"
            f"- **Type:** {self.array_type}\n"
            f"- **Path:** {path_info}"
        )

        return (
            status_msg,
            False,  # Rebuild button disabled during indexing
            '<span class="mode-badge-text">🟡 INDEXING</span>',
            gr.Timer(value=2.0, active=True),
        )

    def _current_table_headers(self) -> list[str]:
        """結果一覧の列見出しを返す (未ロード時は空)．

        Returns:
            列見出しのリスト
        """
        if self.viz_interface is None:
            return []
        try:
            return self.viz_interface.get_table_columns()
        except Exception:
            logger.warning(
                "Failed to get table columns", exc_info=True
            )
            return []

    def _render_empty_board_placeholder(self) -> str:
        """Render placeholder SVG when no data is loaded．

        Returns:
            SVG string with placeholder message
        """
        return """
    <svg width="450" height="450" xmlns="http://www.w3.org/2000/svg">
        <rect width="450" height="450" fill="#f5f5f5"/>
        <text x="225" y="200" text-anchor="middle"
              font-size="20" fill="#666">
            No Data Loaded
        </text>
        <text x="225" y="240" text-anchor="middle"
              font-size="14" fill="#999">
            Use Data Source Management section
        </text>
        <text x="225" y="265" text-anchor="middle"
              font-size="14" fill="#999">
            to load .feather files
        </text>
    </svg>
    """

    # ================================================================
    # ワークベンチ (gr.HTML 1 枚) のデータ収集と描画
    # ================================================================

    def _status_view(self) -> StatusView:
        """トップバーに出すサーバー状態を組み立てる．

        Returns:
            StatusView
        """
        status = self.indexing_state.get_status()
        path_label = self._path_label()

        if status == "indexing":
            progress = self.indexing_state.get_progress()
            remaining = (
                self.indexing_state.estimate_remaining_time()
            )
            tail = ""
            if remaining is not None:
                tail = (
                    f" — 残り約{remaining}秒"
                    if remaining < 60
                    else f" — 残り約{remaining // 60}分{remaining % 60}秒"
                )
            return StatusView(
                badge="INDEXING",
                tone="busy",
                count_main=f"{progress['records']:,}",
                count_unit=(
                    f"records / {progress['files']}"
                    f"/{progress['total_files']} files"
                ),
                path_label=path_label,
                message=f"{progress['message']}{tail}",
            )

        if status == "failed":
            return StatusView(
                badge="ERROR",
                tone="error",
                count_main="0",
                count_unit="records",
                path_label=path_label,
                message=self.indexing_state.get_error() or "",
            )

        if not self.has_data:
            return StatusView(
                badge="NO DATA",
                tone="none",
                count_main="0",
                count_unit="records",
                path_label=path_label,
                message="左のデータソースから読み込んでください",
            )

        if self.array_type == "game-graph":
            viz = self._game_graph_viz
            nodes, edges = (
                viz.get_counts() if viz is not None else (0, 0)
            )
            return StatusView(
                badge="GRAPH",
                tone="ok",
                count_main=f"{nodes:,}",
                count_unit=f"nodes / {edges:,} edges",
                path_label=path_label,
            )

        with self._index_lock:
            total = (
                self.search_index.total_records()
                if self.search_index is not None
                else 0
            )
        return StatusView(
            badge="MOCK" if self.use_mock_data else "REAL",
            tone="mock" if self.use_mock_data else "ok",
            count_main=f"{total:,}",
            count_unit="records",
            path_label=path_label,
        )

    def _path_label(self) -> str:
        """読み込み中のパスを 1 行で表す．

        Returns:
            表示用のパス文字列
        """
        if not self.file_paths:
            return "—"
        if len(self.file_paths) == 1:
            return str(self.file_paths[0])
        return (
            f"{self.file_paths[0].parent}/ "
            f"({len(self.file_paths)} files)"
        )

    @staticmethod
    def _parse_eval(text: str) -> int | None:
        """評価値入力を整数に直す (空欄・不正は無制限)．

        Args:
            text: 入力文字列

        Returns:
            整数．無制限なら None．
        """
        try:
            return int(str(text).strip())
        except (TypeError, ValueError):
            return None

    def _page_records(
        self, state: WorkbenchState
    ) -> tuple[list[dict[str, Any]], int]:
        """状態に対応するページのレコードと総ページ数を返す．

        Args:
            state: 操作状態

        Returns:
            (レコード列, 総ページ数)
        """
        if self.viz_interface is None:
            return ([], 1)
        min_eval = self._parse_eval(state.min_eval)
        max_eval = self._parse_eval(state.max_eval)
        try:
            (_, _, _, _, records) = (
                self.viz_interface.search_by_eval_range(
                    min_eval=min_eval,
                    max_eval=max_eval,
                    page=state.page,
                    page_size=state.page_size,
                )
            )
        except Exception:
            logger.exception("Failed to fetch page records")
            return ([], 1)
        total_pages = self._calculate_total_pages(
            min_eval, max_eval, state.page_size
        )
        return (list(records), max(total_pages, 1))

    def _search_record(
        self, state: WorkbenchState
    ) -> dict[str, Any] | None:
        """ID / SFEN 検索の 1 件を取り出す．

        Args:
            state: 操作状態

        Returns:
            レコード．見つからなければ None．
        """
        if self.viz_interface is None:
            return None
        retriever = self.viz_interface.data_retriever
        try:
            if state.id_query.strip():
                return retriever.get_by_id(
                    state.id_query.strip()
                )
            if state.sfen_query.strip():
                return retriever.get_by_sfen(
                    state.sfen_query.strip()
                )
        except Exception:
            logger.exception("Search failed")
        return None

    def _collect_record(
        self, state: WorkbenchState
    ) -> RecordData:
        """レコードブラウザの表示データを組み立てる．

        Args:
            state: 操作状態

        Returns:
            RecordData
        """
        supports_eval = state.array_type == "hcpe"
        if self.viz_interface is None or not self.has_data:
            return RecordData(
                supports_eval_search=supports_eval,
                highlight_label=self._highlight_label(),
            )

        if state.mode == "search":
            hit = self._search_record(state)
            records = [hit] if hit is not None else []
            total_pages = 1
        else:
            records, total_pages = self._page_records(state)

        renderer = self.viz_interface.renderer
        headers = self.viz_interface.get_table_columns()
        offset = (
            0
            if state.mode == "search"
            else (state.page - 1) * state.page_size
        )
        rows = [
            renderer.format_table_row(i + offset + 1, record)
            for i, record in enumerate(records)
        ]

        index = min(max(state.selected, 0), len(records) - 1)
        current = records[index] if records else None
        if current is None:
            board_svg = self._render_empty_board_placeholder()
            details: dict[str, Any] = {
                "message": "該当するレコードがありません"
            }
        else:
            board_svg = renderer.render_board(current)
            details = renderer.extract_display_fields(current)

        distribution = (
            self.viz_interface.get_distribution(records)
            if records
            else None
        )
        current_value = (
            self.viz_interface.get_record_value(current)
            if current is not None
            else None
        )
        result_value = (
            current.get("resultValue")
            if current is not None
            and state.array_type == "preprocessing"
            else None
        )

        return RecordData(
            headers=headers,
            rows=rows,
            board_svg=board_svg,
            sfen=str(details.get("sfen", "")),
            record_id=str(details.get("id", "")),
            details=details,
            stats=self._stats_cache,
            distribution=distribution,
            current_value=current_value,
            total_pages=total_pages,
            total_records=len(records),
            result_value=(
                float(result_value)
                if result_value is not None
                else None
            ),
            supports_eval_search=supports_eval,
            highlight_label=self._highlight_label(),
        )

    def _highlight_label(self) -> str:
        """盤面ハイライトの凡例文言を返す．

        Returns:
            凡例文言
        """
        if self.array_type == "stage1":
            return "到達可能マス (stage1)"
        if self.array_type == "stage2":
            return "合法手の着手先 (stage2)"
        return "直前の指し手"

    def _collect_graph(
        self, state: WorkbenchState
    ) -> GraphData:
        """ゲームグラフの表示データを組み立てる．

        Args:
            state: 操作状態

        Returns:
            GraphData
        """
        viz = self._game_graph_viz
        if viz is None:
            return GraphData()

        try:
            root = (
                int(state.node)
                if state.node
                else self._game_graph_root_hash
            )
        except (TypeError, ValueError):
            root = self._game_graph_root_hash

        canvas = viz.get_canvas_data(
            root,
            int(state.depth),
            float(state.min_prob),
            self._game_graph_layout,
        )
        graph_html = build_graph_html(
            json.dumps(canvas, ensure_ascii=False)
        )
        moves = [
            [r.japanese, r.probability, r.win_rate]
            for r in viz.get_move_table(root)
        ]
        crumbs = [
            (str(c.get("label", "")), str(c.get("hash", "")))
            for c in viz.get_breadcrumb_data(root)
        ]
        nodes, edges = viz.get_counts()
        return GraphData(
            graph_html=graph_html,
            breadcrumb=crumbs,
            board_svg=viz.get_board_svg(root),
            node_stats=viz.get_node_stats(root),
            moves=moves,
            usi_line=viz.export_sfen_path(root),
            node_count=len(canvas.get("nodes", [])),
            edge_count=edges,
            total_nodes=nodes,
        )

    def _render(self, state: WorkbenchState) -> str:
        """状態からワークベンチ全体の HTML を組み立てる．

        Args:
            state: 操作状態

        Returns:
            HTML 文字列
        """
        self._render_seq += 1
        status = self._status_view()
        if state.array_type == "game-graph":
            return render_workbench(
                state,
                status,
                graph=self._collect_graph(state),
                render_stamp=str(self._render_seq),
            )
        return render_workbench(
            state,
            status,
            record=self._collect_record(state),
            render_stamp=str(self._render_seq),
        )

    # ================================================================
    # アクション解釈
    # ================================================================

    def _on_action(
        self, action: str, state: WorkbenchState
    ) -> tuple[WorkbenchState, str]:
        """data-action 文字列を状態に適用して再描画する．

        Args:
            action: JS から届いたアクション文字列
            state: 現在の操作状態

        Returns:
            (新しい状態, HTML)
        """
        verb, _, rest = str(action).partition(":")

        if verb == "type":
            state = replace(
                state,
                array_type=rest,
                page=1,
                selected=0,
                mode="page",
                node="",
            )
            self._switch_array_type(rest)
        elif verb == "srcmode":
            state = replace(state, source_mode=rest)
        elif verb == "path":
            state = replace(state, path_text=rest)
        elif verb == "id":
            state = replace(state, id_query=rest, sfen_query="")
        elif verb == "sfen":
            state = replace(state, sfen_query=rest, id_query="")
        elif verb == "mineval":
            state = replace(state, min_eval=rest, page=1)
        elif verb == "maxeval":
            state = replace(state, max_eval=rest, page=1)
        elif verb == "search":
            state = replace(state, mode="search", selected=0)
        elif verb == "clear":
            state = replace(
                state,
                mode="page",
                id_query="",
                sfen_query="",
                min_eval="",
                max_eval="",
                page=1,
                selected=0,
            )
        elif verb == "row":
            state = replace(state, selected=as_int(rest, 0))
        elif verb == "page":
            state = self._step_page(state, rest)
        elif verb == "rec":
            state = self._step_record(state, rest)
        elif verb == "stats":
            self._stats_cache = self._get_current_stats()
        elif verb == "refresh":
            # 表示を今のサーバー状態で描き直すだけ (再描画は常に走る)
            self._stats_cache = {}
        elif verb == "load":
            state = self._do_load(state)
        elif verb == "rebuild":
            self._rebuild_index()
            self._stats_cache = {}
        elif verb == "depth":
            state = replace(
                state, depth=as_int(rest, state.depth)
            )
        elif verb == "minprob":
            state = replace(
                state,
                min_prob=as_float(rest, state.min_prob),
            )
        elif verb in ("redraw", "csv", "setroot"):
            state = self._graph_action(state, verb)
        elif verb == "node":
            state = replace(
                state,
                node="" if rest == "root" else rest,
            )
        else:
            logger.debug("Unknown action: %s", action)

        return (state, self._render(state))

    def _switch_array_type(self, array_type: str) -> None:
        """セグメンテッドコントロールでデータ型を切り替える．

        レンダラーもインデックスも array_type に紐づくので，今のデータ
        ソースを新しい型で読み直す (旧 UI ではデータ型 Dropdown を変えて
        「読み込み」を押す 2 段階だった操作を 1 つにまとめている)．

        Args:
            array_type: 新しいデータ型
        """
        if array_type == self.array_type:
            return
        self.array_type = array_type
        self.supports_eval_search = self._supports_eval_search()
        self._stats_cache = {}

        if array_type == "game-graph":
            # グラフはノード/エッジの feather を要求するので，
            # パスを指定して読み込み直してもらう
            self.has_data = self._game_graph_viz is not None
            return

        if not self.file_paths:
            self.has_data = False
            return

        # モックはその場で作れる．実データはバックグラウンドで貼り直す．
        self.indexing_state.set_indexing(len(self.file_paths))
        self._indexing_thread = threading.Thread(
            target=self._build_index_background,
            args=(
                list(self.file_paths),
                array_type,
                self.use_mock_data,
            ),
            daemon=True,
        )
        self._indexing_thread.start()
        if self.use_mock_data:
            # モック構築は一瞬で終わるので待ち合わせて，
            # 切り替え直後の描画に間に合わせる
            self._indexing_thread.join(timeout=5.0)

    def _graph_action(
        self, state: WorkbenchState, verb: str
    ) -> WorkbenchState:
        """ゲームグラフ固有のアクションを適用する．

        Args:
            state: 操作状態
            verb: アクション名

        Returns:
            新しい状態
        """
        if verb == "setroot":
            # 選択中のノードをそのままルートとして扱う (再描画で反映)
            return state
        if verb == "csv":
            logger.info(
                "CSV export is available in the standalone "
                "game graph server (maou visualize "
                "--array-type game-graph)"
            )
        return state

    def _step_page(
        self, state: WorkbenchState, direction: str
    ) -> WorkbenchState:
        """ページを 1 つ送る．

        Args:
            state: 操作状態
            direction: "prev" / "next"

        Returns:
            新しい状態
        """
        total = self._calculate_total_pages(
            self._parse_eval(state.min_eval),
            self._parse_eval(state.max_eval),
            state.page_size,
        )
        page = state.page + (1 if direction == "next" else -1)
        page = max(1, min(page, max(total, 1)))
        return replace(
            state, page=page, selected=0, mode="page"
        )

    def _step_record(
        self, state: WorkbenchState, direction: str
    ) -> WorkbenchState:
        """レコードを 1 つ送る (ページ境界を跨ぐ)．

        Args:
            state: 操作状態
            direction: "prev" / "next"

        Returns:
            新しい状態
        """
        if state.mode == "search":
            return state
        records, total_pages = self._page_records(state)
        count = len(records)
        if direction == "next":
            if state.selected + 1 < count:
                return replace(
                    state, selected=state.selected + 1
                )
            if state.page < total_pages:
                return replace(
                    state, page=state.page + 1, selected=0
                )
            return state
        if state.selected > 0:
            return replace(state, selected=state.selected - 1)
        if state.page > 1:
            previous = replace(
                state, page=state.page - 1, selected=0
            )
            earlier, _ = self._page_records(previous)
            return replace(
                previous, selected=max(len(earlier) - 1, 0)
            )
        return state

    def _do_load(self, state: WorkbenchState) -> WorkbenchState:
        """データソースを読み込む．

        Args:
            state: 操作状態

        Returns:
            新しい状態
        """
        self._load_new_data_source(
            state.source_mode,
            state.path_text,
            state.path_text,
            state.array_type,
        )
        self._stats_cache = {}
        return replace(
            state, page=1, selected=0, mode="page", node=""
        )

    # ================================================================
    # Gradio の組み立て
    # ================================================================

    def create_demo(self) -> gr.Blocks:
        """gr.Blocks を構築して返す．

        画面はワークベンチ 1 枚 (gr.HTML)．操作は data-action 文字列と
        して JS から届き，レーンごとのブリッジ経由で _on_action に入る．

        Returns:
            設定済みの Gradio Blocks インスタンス
        """
        initial = WorkbenchState(
            array_type=self.array_type,
            path_text=self._path_label()
            if self.file_paths
            else "",
        )

        with gr.Blocks(
            title="Maou Shogi Data Visualizer"
        ) as demo:
            state = gr.State(initial)
            workbench = gr.HTML(
                self._render(initial),
                elem_id="viz-workbench-slot",
            )

            # インデックス構築中だけ回すポーリング．構築が終わったら
            # 自分で止める (常時 2 秒間隔で再描画し続けない)．
            timer = gr.Timer(
                value=2.0,
                active=self.indexing_state.is_indexing(),
            )

            buffers: dict[str, dict[str, str]] = {
                lane: {} for lane in WORKBENCH_LANES
            }
            bridges = {
                lane: gr.HTML(
                    value="",
                    elem_id=f"vz-bridge-{lane}",
                    elem_classes=["maou-hidden"],
                    server_functions=[
                        make_workbench_bridge(buffers[lane])
                    ],
                    js_on_load=workbench_js_on_load(lane),
                )
                for lane in WORKBENCH_LANES
            }

            def _lane_handler(
                lane_name: str,
            ) -> Callable[[WorkbenchState], tuple[Any, ...]]:
                """レーン名を束縛したハンドラを返す．

                ループ変数を直接掴むと全レーンが最後の名前を参照して
                しまうので，引数で束縛する．

                Args:
                    lane_name: レーン名

                Returns:
                    .change() に渡すハンドラ
                """

                def run(
                    current: WorkbenchState,
                ) -> tuple[Any, ...]:
                    """控えたアクションを適用して再描画する．"""
                    return self._on_action(
                        buffers[lane_name].pop("value", ""),
                        current,
                    )

                return run

            for lane in WORKBENCH_LANES:
                bridges[lane].change(
                    _lane_handler(lane),
                    inputs=[state],
                    outputs=[state, workbench],
                )

            timer.tick(
                self._on_tick,
                inputs=[state],
                outputs=[state, workbench, timer],
            )

        return demo

    def _on_tick(
        self, state: WorkbenchState
    ) -> tuple[WorkbenchState, str, Any]:
        """インデックス構築中のポーリング．

        Args:
            state: 操作状態

        Returns:
            (状態, HTML, タイマー更新)
        """
        busy = self.indexing_state.is_indexing()
        if not busy and not self._stats_cache and self.has_data:
            self._stats_cache = self._get_current_stats()
        return (
            state,
            self._render(state),
            gr.Timer(value=2.0, active=busy),
        )

    def _get_default_board_svg(self) -> str:
        """デフォルトの盤面SVGを生成（平手初期配置）．"""
        # 平手初期配置のモック
        # 実際の実装では，標準的な初期配置を設定
        mock_board = [[0 for _ in range(9)] for _ in range(9)]
        mock_hand = [0 for _ in range(14)]

        # 簡易的な初期配置（いくつかの駒を配置）
        # 後手（白）の飛車と角
        mock_board[0][1] = 16 + 6  # 後手角（22）
        mock_board[0][7] = 16 + 7  # 後手飛車（23）
        mock_board[0][4] = 16 + 8  # 後手王（24）

        # 先手（黒）の飛車と角
        mock_board[8][7] = 6  # 先手角
        mock_board[8][1] = 7  # 先手飛車
        mock_board[8][4] = 8  # 先手王

        position = BoardPosition(
            board_id_positions=mock_board,
            pieces_in_hand=mock_hand,
        )

        return self.renderer.render(position)

    def _get_current_stats(self) -> dict[str, Any]:
        """Get current dataset statistics (thread-safe).

        Returns:
            Statistics dict, or empty dict if not ready.
        """
        with self._index_lock:
            if self.viz_interface is not None:
                return self.viz_interface.get_dataset_stats()
            return {}

    def _get_mock_stats(self) -> dict[str, Any]:
        """インデックス統計情報を返す．"""
        total_records = (
            self.search_index.total_records()
            if self.search_index is not None
            else 0
        )
        return {
            "total_records": total_records,
            "array_type": self.array_type,
            "num_files": len(self.file_paths),
        }

    def _calculate_total_pages(
        self,
        min_eval: int | None,
        max_eval: int | None,
        page_size: int,
    ) -> int:
        """総ページ数を計算する．

        Args:
            min_eval: 最小評価値（HCPEのみ）
            max_eval: 最大評価値（HCPEのみ）
            page_size: ページサイズ

        Returns:
            総ページ数
        """
        if self.search_index is None:
            return 1

        if self.supports_eval_search:
            # HCPEの場合は評価値範囲でカウント
            total_records = self.search_index.count_eval_range(
                min_eval, max_eval
            )
        else:
            # その他のデータ型は全レコード
            total_records = self.search_index.total_records()

        if total_records == 0:
            return 1

        return (total_records + page_size - 1) // page_size


def launch_server(
    file_paths: list[Path],
    array_type: str,
    port: int | None,
    share: bool,
    server_name: str,
    debug: bool,
    use_mock_data: bool = False,
) -> None:
    """Gradioサーバーを起動．

    array_typeに応じて異なるUIを提供する:
    - hcpe/preprocessing/stage1/stage2: レコードブラウザUI
    - game-graph: ゲームグラフ可視化UI

    Args:
        file_paths: データファイルのパスリスト
        array_type: データ型
        port: サーバーポート
        share: 公開リンク作成フラグ
        server_name: サーバーバインドアドレス
        debug: デバッグモード
        use_mock_data: Trueの場合はモックデータを使用
    """
    # game-graph はゲームグラフ専用UIにディスパッチ
    if array_type == "game-graph":
        from maou.infra.visualization.game_graph_server import (
            launch_game_graph_server,
        )

        graph_path = file_paths[0] if file_paths else None
        if graph_path is None:
            raise ValueError(
                "game-graph requires a graph data directory path"
            )
        launch_game_graph_server(
            graph_path=graph_path,
            port=port,
            share=share,
            server_name=server_name,
        )
        return

    server = GradioVisualizationServer(
        file_paths=file_paths,
        array_type=array_type,
        use_mock_data=use_mock_data,
    )

    demo = server.create_demo()

    # カスタムCSSを読み込み（Gradio 6ではlaunch()に渡す必要がある）
    custom_css = _load_custom_css()

    port_desc = (
        str(port) if port is not None else "auto (7860-7959)"
    )
    logger.info(
        f"Launching Gradio server on {server_name}:{port_desc} "
        f"(share={share}, debug={debug})"
    )

    # ゲームグラフJS(Canvas 2D レンダラー + イベントハンドラ)をhead要素に注入．
    # gradio_server.py ではデータソース動的切替でゲームグラフが使われるため必要．
    from maou.infra.visualization.game_graph_server import (
        _build_head_scripts,
    )

    head_scripts = (
        FONT_LINKS
        + _build_head_scripts()
        + build_workbench_head()
    )

    launch_kwargs: dict[str, Any] = {
        "server_name": server_name,
        "share": share,
        "debug": debug,
        "show_error": True,
        "css": custom_css,
        "head": head_scripts,
    }
    if port is not None:
        launch_kwargs["server_port"] = port

    demo.launch(**launch_kwargs)
