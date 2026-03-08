"""Gradio UIサーバー実装（インフラ層）．

将棋データ可視化のためのGradio Webインターフェースを提供する．
"""

import logging
import os
import threading
from pathlib import Path
from typing import Any

# Fix matplotlib backend for Google Colab compatibility
# matplotlib reads MPLBACKEND during import, so we must fix it before importing
_saved_mplbackend = os.environ.get("MPLBACKEND")
if _saved_mplbackend and "inline" in _saved_mplbackend:
    os.environ["MPLBACKEND"] = "Agg"

import matplotlib  # noqa: E402

matplotlib.use(
    "Agg", force=True
)  # Ensure non-interactive backend

# Restore environment to avoid affecting other Colab cells
# matplotlib is now cached, so other imports will reuse this instance
if _saved_mplbackend is not None:
    os.environ["MPLBACKEND"] = _saved_mplbackend
elif "MPLBACKEND" in os.environ:
    del os.environ["MPLBACKEND"]

import gradio as gr  # noqa: E402

from maou.infra.file_system.file_system import (  # noqa: E402
    FileSystem,
)
from maou.infra.visualization.indexing_state import (  # noqa: E402
    IndexingState,
)
from maou.infra.visualization.search_index import (  # noqa: E402
    SearchIndex,
)
from maou.interface.path_suggestions import (  # noqa: E402
    PathSuggestionService,
)
from maou.interface.visualization import (  # noqa: E402
    BoardPosition,
    SVGBoardRenderer,
    VisualizationInterface,
)

logger = logging.getLogger(__name__)


def _load_custom_css() -> str:
    """カスタムCSSファイルを読み込む．

    Returns:
        str: 結合されたCSS文字列
    """
    static_dir = Path(__file__).parent / "static"
    css_files = ["theme.css", "components.css"]

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


def create_loading_spinner(
    message: str = "データ読み込み中...",
) -> str:
    """ローディングスピナーHTMLを生成．

    Args:
        message: 表示するメッセージ

    Returns:
        str: ローディングスピナーのHTML文字列
    """
    return f"""
    <div class="loading">
        <div class="spinner"></div>
        <p>{message}</p>
    </div>
    """


def create_toast_notification_script() -> str:
    """トースト通知用JavaScriptを生成．

    Returns:
        str: JavaScriptコード文字列
    """
    return """
    <script>
    (function() {
        // Toast notification system
        let toastContainer = null;

        function initToastContainer() {
            if (!toastContainer) {
                toastContainer = document.createElement('div');
                toastContainer.className = 'toast-container';
                document.body.appendChild(toastContainer);
            }
        }

        function showToast(title, message, type = 'info', duration = 5000) {
            initToastContainer();

            const toast = document.createElement('div');
            toast.className = `toast toast-${type}`;

            const icons = {
                success: '✓',
                error: '✕',
                warning: '⚠',
                info: 'ℹ'
            };

            toast.innerHTML = `
                <div class="toast-icon">${icons[type] || icons.info}</div>
                <div class="toast-content">
                    <div class="toast-title">${title}</div>
                    ${message ? `<div class="toast-message">${message}</div>` : ''}
                </div>
                <button class="toast-close" onclick="this.parentElement.remove()">×</button>
            `;

            toastContainer.appendChild(toast);

            if (duration > 0) {
                setTimeout(() => {
                    toast.style.animation = 'toast-slide-in 0.3s ease-out reverse';
                    setTimeout(() => toast.remove(), 300);
                }, duration);
            }
        }

        // Expose toast function globally
        window.showToast = showToast;

        console.log('🔔 Toast notification system initialized');
    })();
    </script>
    """


def create_keyboard_shortcuts_script() -> str:
    """キーボードショートカット用JavaScriptを生成．

    Returns:
        str: JavaScriptコード文字列
    """
    return """
    <script>
    (function() {
        // Help modal state
        let helpModalVisible = false;

        // Create help modal element
        const helpModal = document.createElement('div');
        helpModal.id = 'keyboard-help-modal';
        helpModal.style.cssText = `
            display: none;
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 32px;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            z-index: 10000;
            max-width: 500px;
            width: 90%;
        `;
        helpModal.innerHTML = `
            <h2 style="margin: 0 0 24px 0; font-size: 24px; font-weight: 600; color: #1a1a1a;">
                ⌨️ キーボードショートカット
            </h2>
            <div style="display: grid; gap: 12px;">
                <div style="display: flex; justify-content: space-between; padding: 8px; border-bottom: 1px solid #e5e5e5;">
                    <span style="font-weight: 600; color: #666;">/</span>
                    <span style="color: #1a1a1a;">検索にフォーカス</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 8px; border-bottom: 1px solid #e5e5e5;">
                    <span style="font-weight: 600; color: #666;">Esc</span>
                    <span style="color: #1a1a1a;">検索クリア/閉じる</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 8px; border-bottom: 1px solid #e5e5e5;">
                    <span style="font-weight: 600; color: #666;">J / ↓</span>
                    <span style="color: #1a1a1a;">次のレコード</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 8px; border-bottom: 1px solid #e5e5e5;">
                    <span style="font-weight: 600; color: #666;">K / ↑</span>
                    <span style="color: #1a1a1a;">前のレコード</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 8px; border-bottom: 1px solid #e5e5e5;">
                    <span style="font-weight: 600; color: #666;">Ctrl + →</span>
                    <span style="color: #1a1a1a;">次のページ</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 8px; border-bottom: 1px solid #e5e5e5;">
                    <span style="font-weight: 600; color: #666;">Ctrl + ←</span>
                    <span style="color: #1a1a1a;">前のページ</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 8px;">
                    <span style="font-weight: 600; color: #666;">?</span>
                    <span style="color: #1a1a1a;">ヘルプ表示</span>
                </div>
            </div>
            <button id="close-help-modal" style="
                margin-top: 24px;
                width: 100%;
                padding: 12px;
                background: #0070f3;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: 500;
                cursor: pointer;
                transition: background 0.2s ease;
            ">閉じる</button>
        `;

        // Create backdrop
        const backdrop = document.createElement('div');
        backdrop.id = 'keyboard-help-backdrop';
        backdrop.style.cssText = `
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            z-index: 9999;
        `;

        // Add to DOM
        document.body.appendChild(backdrop);
        document.body.appendChild(helpModal);

        // Toggle help modal
        function toggleHelpModal() {
            helpModalVisible = !helpModalVisible;
            helpModal.style.display = helpModalVisible ? 'block' : 'none';
            backdrop.style.display = helpModalVisible ? 'block' : 'none';
        }

        // Close modal button
        document.getElementById('close-help-modal').addEventListener('click', toggleHelpModal);
        backdrop.addEventListener('click', toggleHelpModal);

        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            // Don't trigger shortcuts when typing in input fields
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                if (e.key === 'Escape') {
                    e.target.value = '';
                    e.target.blur();
                }
                return;
            }

            // Close help modal with Escape
            if (e.key === 'Escape' && helpModalVisible) {
                toggleHelpModal();
                return;
            }

            switch(e.key.toLowerCase()) {
                case 'j':
                case 'arrowdown':
                    e.preventDefault();
                    document.getElementById('next-record')?.click();
                    break;
                case 'k':
                case 'arrowup':
                    e.preventDefault();
                    document.getElementById('prev-record')?.click();
                    break;
                case '/':
                    e.preventDefault();
                    // Dropdownのinput要素にフォーカス
                    const searchDropdown = document.getElementById('id-search-input')?.querySelector('input');
                    if (searchDropdown) {
                        searchDropdown.focus();
                        searchDropdown.click();  // ドロップダウンを開く
                    }
                    break;
                case '?':
                    e.preventDefault();
                    toggleHelpModal();
                    break;
                case 'arrowright':
                    if (e.ctrlKey) {
                        e.preventDefault();
                        document.getElementById('next-page')?.click();
                    }
                    break;
                case 'arrowleft':
                    if (e.ctrlKey) {
                        e.preventDefault();
                        document.getElementById('prev-page')?.click();
                    }
                    break;
            }
        });

        console.log('⌨️ Keyboard shortcuts initialized');
    })();
    </script>
    """


class GradioVisualizationServer:
    """Gradio可視化サーバークラス．

    将棋データの検索と視覚化のためのWebインターフェースを提供する．
    """

    def __init__(
        self,
        file_paths: list[Path],
        array_type: str,
        model_path: Path | None = None,
        use_mock_data: bool = False,
    ) -> None:
        """サーバーを初期化．

        Args:
            file_paths: データファイルのパスリスト
            array_type: データ型（hcpe, preprocessing, stage1, stage2）
            model_path: オプショナルなモデルファイルパス
            use_mock_data: Trueの場合はモックデータを使用
        """
        self.file_paths = file_paths
        self.array_type = (
            array_type  # This can now be changed dynamically
        )
        self.model_path = model_path
        self.use_mock_data = use_mock_data
        self.renderer = SVGBoardRenderer()

        # Check if data is available
        self.has_data = len(file_paths) > 0 or use_mock_data

        # ゲームツリー状態(game-tree モード時に使用)
        self._game_tree_viz: Any = None
        self._game_tree_root_hash: int = 0

        # 評価値検索をサポートするかどうかを判定
        self.supports_eval_search = self._supports_eval_search()

        # Initialize path suggestion service
        self.path_suggester = PathSuggestionService(
            cache_ttl=60
        )

        # Initialize threading infrastructure
        self.indexing_state = IndexingState()
        self._index_lock = threading.Lock()
        self._indexing_thread: threading.Thread | None = None

        if self.has_data and array_type == "game-tree":
            # ゲームツリー: 直接読み込み(インデックス不要)
            self.search_index = None  # type: ignore[assignment]
            self.viz_interface = None  # type: ignore[assignment]
            try:
                self._load_game_tree_data(file_paths[0])
                logger.info(
                    f"✅ Game tree loaded: root={self._game_tree_root_hash:#018x}"
                )
            except Exception:
                logger.exception(
                    "Failed to load game tree data"
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

    def _check_indexing_status_with_transition(
        self,
        prev_status: str,
        page_size: int,
    ) -> tuple[
        Any,  # status_markdown
        Any,  # load_btn
        Any,  # rebuild_btn
        Any,  # refresh_btn
        Any,  # mode_badge
        str,  # current_status (for previous_indexing_status State)
        Any,  # accordion_update
        Any,  # timer_update
        Any,  # results_table
        Any,  # page_info
        Any,  # board_display
        Any,  # record_details
        Any,  # current_page_records
        Any,  # current_record_index
        Any,  # record_indicator
        Any,  # analytics_chart
        Any,  # prev_btn
        Any,  # next_btn
        Any,  # prev_record_btn
        Any,  # next_record_btn
        Any,  # selected_record_id
        Any,  # stats_json
    ]:
        """インデックス作成状態をポーリングし，状態遷移を検出する．

        タイマーから呼び出され，前回の状態と比較して状態遷移を検出．
        インデックス作成中で状態変化がない場合は，status_markdownのみを更新し，
        他のコンポーネントはgr.update()で更新をスキップする（ちらつき防止）．

        indexing→ready遷移時はデータコンポーネントも一括更新する（Event 7を統合）．

        Args:
            prev_status: 前回のポーリング時の状態
            page_size: ページサイズ（ready遷移時のデータ読み込みに使用）

        Returns:
            22個の出力値のタプル:
            (status_message, load_btn, rebuild_btn, refresh_btn, mode_badge,
             current_status, accordion_update, timer_update,
             results_table, page_info, board_display, record_details,
             current_page_records, current_record_index, record_indicator,
             analytics_chart, prev_btn, next_btn, prev_record_btn,
             next_record_btn, selected_record_id, stats_json)
        """
        current_status = self.indexing_state.get_status()

        # 状態遷移を検出
        is_state_transition = prev_status != current_status
        is_ready_transition = (
            prev_status == "indexing"
            and current_status == "ready"
        )

        # gr.update()の14個のデータコンポーネント（更新しない場合）
        no_data_updates: tuple[Any, ...] = (
            gr.update(),  # results_table
            gr.update(),  # page_info
            gr.update(),  # board_display
            gr.update(),  # record_details
            gr.update(),  # current_page_records
            gr.update(),  # current_record_index
            gr.update(),  # record_indicator
            gr.update(),  # analytics_chart
            gr.update(),  # prev_btn
            gr.update(),  # next_btn
            gr.update(),  # prev_record_btn
            gr.update(),  # next_record_btn
            gr.update(),  # selected_record_id
            gr.update(),  # stats_json
        )

        # 安定状態でもバッジは常に正しい状態を返す
        _, _, _, stable_mode_badge = (
            self._check_indexing_status()
        )

        # 安定状態（状態変化なし，かつ indexing 以外）では再描画をスキップ
        if (
            not is_state_transition
            and current_status != "indexing"
        ):
            return (
                gr.update(),  # status_msg
                gr.update(),  # load_btn
                gr.update(),  # rebuild_btn
                gr.update(),  # refresh_btn
                stable_mode_badge,  # バッジは常にHTMLを返す
                current_status,
                gr.update(),  # accordion_update
                gr.update(),  # timer_update
                *no_data_updates,
            )

        # indexing 中で状態変化なしの場合: status_markdown のみ更新
        if (
            current_status == "indexing"
            and not is_state_transition
        ):
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

            # Loading spinner HTML
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

            # INDEXING バッジ HTML
            indexing_badge = '<span class="mode-badge-text">🟡 INDEXING</span>'

            return (
                status_msg,
                gr.update(),  # load_btn - no change
                gr.update(),  # rebuild_btn - no change
                gr.update(),  # refresh_btn - no change
                indexing_badge,  # バッジは常にHTMLを返す
                current_status,
                gr.update(),  # accordion_update - no change
                gr.update(),  # timer_update - no change
                *no_data_updates,
            )

        # 状態遷移がある場合: ステータスコンポーネントを更新
        status_msg, load_btn, rebuild_btn, mode_badge = (
            self._check_indexing_status()
        )
        refresh_btn = rebuild_btn

        # アコーディオン状態を決定
        if current_status == "indexing":
            accordion_update = gr.update(open=True)
        elif is_ready_transition:
            accordion_update = gr.update(open=False)
        else:
            accordion_update = gr.update()

        # タイマー状態を決定
        timer_update: Any
        if is_ready_transition:
            timer_update = gr.Timer(value=2.0, active=False)
        else:
            timer_update = gr.update()

        # indexing→ready遷移時はデータを読み込み
        if is_ready_transition:
            paginate_result = self._paginate_all_data(
                min_eval=-9999,
                max_eval=9999,
                page=1,
                page_size=page_size,
            )
            stats = self._get_current_stats()
            data_outputs: tuple[Any, ...] = (
                *paginate_result,
                stats,
            )
        else:
            data_outputs = no_data_updates

        return (
            status_msg,
            load_btn,
            rebuild_btn,
            refresh_btn,
            mode_badge,
            current_status,
            accordion_update,
            timer_update,
            *data_outputs,
        )

    def _get_id_suggestions_handler(self, prefix: str) -> Any:
        """ID入力に応じて候補を動的更新．

        Args:
            prefix: ユーザーが入力したプレフィックス

        Returns:
            Dropdownの選択肢更新
        """
        # Thread-safe access to viz_interface
        with self._index_lock:
            # Check for empty state
            if not self.has_data or self.viz_interface is None:
                return gr.update(choices=[])

            if not prefix or len(prefix) < 2:
                # 2文字未満の場合は初期候補（最初の1000件）を表示
                initial_ids = self.viz_interface.get_all_ids(
                    limit=1000
                )
                return gr.update(choices=initial_ids)

            # プレフィックスに基づく候補を取得
            suggestions = self.viz_interface.get_id_suggestions(
                prefix, limit=50
            )
            return gr.update(choices=suggestions)

    def _get_directory_suggestions_handler(
        self, prefix: str
    ) -> Any:
        """Get directory path suggestions based on user input．

        Args:
            prefix: User-typed prefix (minimum 2 characters)

        Returns:
            Dropdown update with suggestions
        """
        if not prefix or len(prefix) < 2:
            return gr.update(choices=[])

        try:
            suggestions = (
                self.path_suggester.get_directory_suggestions(
                    prefix=prefix,
                    limit=50,
                )
            )
            logger.debug(
                f"Directory suggestions for '{prefix}': {len(suggestions)} results"
            )
            return gr.update(choices=suggestions)
        except Exception as e:
            logger.error("Directory suggestion failed: %s", e)
            return gr.update(choices=[])

    def _get_file_suggestions_handler(self, prefix: str) -> Any:
        """Get .feather file path suggestions based on user input．

        Args:
            prefix: User-typed prefix (minimum 2 characters)

        Returns:
            Dropdown update with suggestions
        """
        if not prefix or len(prefix) < 2:
            return gr.update(choices=[])

        try:
            suggestions = (
                self.path_suggester.get_file_suggestions(
                    prefix=prefix,
                    limit=100,  # Higher limit for file mode
                )
            )
            logger.debug(
                f"File suggestions for '{prefix}': {len(suggestions)} results"
            )
            return gr.update(choices=suggestions)
        except Exception as e:
            logger.error("File suggestion failed: %s", e)
            return gr.update(choices=[])

    def _supports_eval_search(self) -> bool:
        """評価値範囲検索をサポートするデータ型かどうかを判定．

        Returns:
            bool: hcpeの場合はTrue，それ以外はFalse
        """
        return self.array_type == "hcpe"

    def _get_initial_status_message(self) -> str:
        """Generate initial status message based on current state．

        Returns:
            str: ステータスメッセージ
        """
        if self.use_mock_data:
            return "**Status:** 🟡 Using mock data for testing"
        elif self.has_data and self.search_index is not None:
            total = self.search_index.total_records()
            file_count = len(self.file_paths)
            return (
                f"**Status:** 🟢 Loaded {total:,} records "
                f"from {file_count} file(s)"
            )
        elif self.has_data:
            # Indexing in progress
            return "**Status:** 🟡 Indexing in progress..."
        else:
            return "**Status:** ⚪ No data loaded - select a data source to begin"

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
                       timer_update, record_panel_visible, game_tree_panel_visible)
        """
        is_game_tree = array_type == "game-tree"

        # Step 1: Validate and resolve paths
        try:
            if is_game_tree:
                # game-tree: ディレクトリパスをそのまま使用
                tree_dir = (
                    Path(dir_path)
                    if source_mode == "Directory"
                    else Path(files_path)
                )
                if not tree_dir.is_dir():
                    raise ValueError(
                        f"ディレクトリが存在しません: {tree_dir}"
                    )
                file_paths = [tree_dir]
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
        record_visible = gr.update(visible=not is_game_tree)
        tree_visible = gr.update(visible=is_game_tree)

        # game-tree: ツリーデータを直接読み込む(インデックス不要)
        if is_game_tree:
            try:
                self._load_game_tree_data(file_paths[0])
                return (
                    f"✅ **Game Tree loaded:** "
                    f"{self._game_tree_root_hash:#018x}",
                    False,
                    '<span class="mode-badge-text">🟢 GAME TREE</span>',
                    gr.update(),
                    record_visible,
                    tree_visible,
                )
            except Exception as e:
                logger.exception("Failed to load game tree")
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
            tree_visible,
        )

    def _load_game_tree_data(self, tree_dir: Path) -> None:
        """ゲームツリーデータを読み込む．

        Args:
            tree_dir: ツリーデータディレクトリ
        """
        from maou.interface.game_tree_io import GameTreeIO
        from maou.interface.game_tree_visualization import (
            GameTreeVisualizationInterface,
        )

        io = GameTreeIO()
        nodes_df, edges_df = io.load(tree_dir)
        metadata = io.load_metadata(tree_dir)
        logger.info(
            "Loaded game tree: %d nodes, %d edges",
            len(nodes_df),
            len(edges_df),
        )

        self._game_tree_viz = GameTreeVisualizationInterface(
            nodes_df,
            edges_df,
            initial_sfen=metadata.get("initial_sfen"),
        )
        self._game_tree_root_hash = (
            self._game_tree_viz.get_root_hash()
        )

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

    def _get_empty_state_outputs(
        self,
    ) -> tuple[
        list[list[Any]],  # table_data
        str,  # page_info
        str,  # board_display
        dict[str, Any],  # record_details
        list[dict[str, Any]],  # cached_records
        int,  # record_index
        str,  # record_indicator
        Any | None,  # analytics_figure (Plotly Figure or None)
        gr.Button,  # prev_btn
        gr.Button,  # next_btn
        gr.Button,  # prev_record_btn
        gr.Button,  # next_record_btn
        str,  # selected_record_id
    ]:
        """Generate output values for empty state (no data loaded)．

        Returns:
            tuple matching outputs for pagination methods (13 values).
        """
        empty_table: list[
            list[Any]
        ] = []  # Empty list for results_table

        page_info = "No data loaded"

        board_display = self._render_empty_board_placeholder()

        record_details = {
            "message": "No data loaded",
            "instruction": "Use 'Data Source Management' section to load data",
        }

        cached_records: list[dict[str, Any]] = []
        record_index = 0
        record_indicator = "Record 0 / 0"
        analytics_figure = (
            None  # gr.Plot accepts None for empty state
        )

        return (
            empty_table,
            page_info,
            board_display,
            record_details,
            cached_records,
            record_index,
            record_indicator,
            analytics_figure,
            gr.Button(interactive=False),  # prev_btn
            gr.Button(interactive=False),  # next_btn
            gr.Button(interactive=False),  # prev_record_btn
            gr.Button(interactive=False),  # next_record_btn
            "",  # selected_record_id
        )

    def _get_empty_state_navigation(
        self,
    ) -> tuple[
        int,  # current_page
        int,  # current_record_index
        list[list[Any]],  # table_data
        str,  # page_info
        str,  # board_svg
        dict[str, Any],  # record_details
        list[dict[str, Any]],  # current_page_records
        str,  # record_indicator
        Any | None,  # analytics_figure (Plotly Figure or None)
        gr.Button,  # prev_btn
        gr.Button,  # next_btn
        str,  # selected_record_id
    ]:
        """Generate output values for empty state navigation (no viz_interface)．

        Returns:
            tuple matching outputs for navigation methods (12 values).
        """
        return (
            1,  # current_page
            0,  # current_record_index
            [],  # empty table
            "No data loaded",  # page_info
            self._render_empty_board_placeholder(),  # board_svg
            {"message": "No data loaded"},  # record_details
            [],  # current_page_records
            "Record 0 / 0",  # record_indicator
            None,  # analytics_figure
            gr.Button(interactive=False),  # prev button
            gr.Button(interactive=False),  # next button
            "",  # selected_record_id
        )

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

    def create_demo(self) -> gr.Blocks:
        """Gradio UIデモを作成．

        Returns:
            設定済みのGradio Blocksインスタンス
        """
        with gr.Blocks(
            title="Maou Shogi Data Visualizer"
        ) as demo:
            # Header with mode badge
            with gr.Row():
                gr.Markdown("# ⚡ Maou将棋データ可視化ツール")

            # Mode indicator with badge (referenceable for updates)
            if self.use_mock_data:
                badge_content = '<span class="mode-badge-text">🔴 MOCK MODE</span>'
            elif self.has_data:
                badge_content = '<span class="mode-badge-text">🟢 REAL MODE</span>'
            else:
                badge_content = '<span class="mode-badge-text">⚪ NO DATA</span>'

            mode_badge = gr.HTML(
                value=badge_content,
                elem_id="mode-badge",
            )

            # Toast notifications
            gr.HTML(create_toast_notification_script())

            # Keyboard shortcuts
            gr.HTML(create_keyboard_shortcuts_script())

            # --- レコードブラウザ UI (hcpe/preprocessing/stage1/stage2) ---
            is_record_mode = self.array_type != "game-tree"
            record_browser_panel = gr.Row(
                visible=is_record_mode
            )
            with record_browser_panel:
                # 左パネル: ナビゲーションと検索コントロール
                with gr.Column(scale=1):
                    # データソース管理セクション
                    with gr.Accordion(
                        "📂 Data Source Management",
                        open=True,  # Always expanded by default
                    ) as data_source_accordion:
                        with gr.Row():
                            source_mode = gr.Radio(
                                choices=[
                                    "Directory",
                                    "File list",
                                ],
                                value="Directory",
                                label="Source Type",
                                scale=1,
                            )

                        initial_dirs = self.path_suggester.preload_directories(
                            base_path=Path.cwd(),
                            max_depth=2,
                            limit=100,
                        )
                        dir_input = gr.Dropdown(
                            label="📁 Directory Path",
                            choices=initial_dirs,
                            value=None,
                            allow_custom_value=True,
                            filterable=True,
                            info="Select from list or type to search",
                            visible=True,
                            scale=3,
                        )

                        files_input = gr.Dropdown(
                            label="📄 File Paths",
                            choices=[],
                            value=None,
                            allow_custom_value=True,
                            filterable=True,
                            info="Type to search .feather files (2+ characters)",
                            visible=False,
                            scale=3,
                        )

                        array_type_dropdown = gr.Dropdown(
                            choices=[
                                "hcpe",
                                "preprocessing",
                                "stage1",
                                "stage2",
                                "game-tree",
                            ],
                            value=self.array_type,
                            label="Array Type",
                            interactive=True,
                            elem_id="array-type-dropdown",
                        )

                        with gr.Row():
                            load_btn = gr.Button(
                                "Load Data Source",
                                variant="primary",
                                scale=2,
                            )
                            rebuild_btn = gr.Button(
                                "Rebuild Index",
                                variant="secondary",
                                scale=1,
                                interactive=self.has_data,  # Only enabled when data is loaded
                            )
                            refresh_btn = gr.Button(
                                "🔄 Refresh",
                                variant="secondary",
                                scale=1,
                                interactive=self.has_data,  # Only enabled when data is loaded
                            )

                        status_markdown = gr.Markdown(
                            value=self._get_initial_status_message(),
                            elem_classes=["status-message"],
                        )

                        # Status polling timer (polls every 2 seconds)
                        # 起動時にインデックス構築中の場合はアクティブ化
                        status_timer = gr.Timer(
                            value=2.0,
                            active=self.indexing_state.is_indexing(),
                        )

                    # ページ内レコードナビゲーション
                    with gr.Group():
                        gr.Markdown(
                            "### 🎯 レコードナビゲーション"
                        )
                        with gr.Row():
                            prev_record_btn = gr.Button(
                                "← 前のレコード",
                                size="sm",
                                elem_id="prev-record",
                            )
                            record_indicator = gr.Markdown(
                                "Record 0 / 0",
                                elem_id="record-indicator",
                            )
                            next_record_btn = gr.Button(
                                "次のレコード →",
                                size="sm",
                                elem_id="next-record",
                            )

                    # ページネーション
                    with gr.Group():
                        gr.Markdown("### 📄 ページネーション")
                        page_size = gr.Slider(
                            label="📊 1ページあたりの件数",
                            info="一度に表示するレコード数を設定（10〜100件）",
                            minimum=10,
                            maximum=100,
                            value=20,
                            step=10,
                        )
                        with gr.Row():
                            prev_btn = gr.Button(
                                "← 前へ", elem_id="prev-page"
                            )
                            next_btn = gr.Button(
                                "次へ →", elem_id="next-page"
                            )
                        page_info = gr.Markdown(
                            "ページ 1", elem_id="page-info"
                        )

                    # 検索機能
                    gr.Markdown("## 🔍 検索機能")

                    # ID検索
                    with gr.Group():
                        gr.Markdown("### ID検索")

                        # 初期化時にID候補リストを取得（最大1000件）
                        initial_ids: list[str] = []
                        if (
                            self.has_data
                            and self.viz_interface is not None
                        ):
                            try:
                                initial_ids = self.viz_interface.get_all_ids(
                                    limit=1000
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to load initial ID list: {e}"
                                )

                        id_input = gr.Dropdown(
                            label="🔍 レコードID",
                            choices=initial_ids,
                            value=None,
                            allow_custom_value=True,
                            filterable=True,
                            info="IDを入力すると候補が絞り込まれます（2文字以上で動的更新）",
                            elem_id="id-search-input",
                        )
                        id_search_btn = gr.Button(
                            "ID検索",
                            variant="primary",
                            elem_id="id-search-btn",
                        )

                    # 評価値範囲検索（HCPEデータのみ）
                    if self.supports_eval_search:
                        with gr.Group():
                            gr.Markdown("### 評価値範囲検索")
                            min_eval = gr.Number(
                                label="📉 最小評価値",
                                info="評価値の下限（空欄で無制限）",
                                value=lambda: None,
                                precision=0,
                                placeholder="制限なし",
                            )
                            max_eval = gr.Number(
                                label="📈 最大評価値",
                                info="評価値の上限（空欄で無制限）",
                                value=lambda: None,
                                precision=0,
                                placeholder="制限なし",
                            )
                            eval_search_btn = gr.Button(
                                "範囲検索", variant="secondary"
                            )
                    else:
                        # 評価値検索非対応の場合はダミーコンポーネント
                        min_eval = gr.Number(visible=False)
                        max_eval = gr.Number(visible=False)
                        eval_search_btn = gr.Button(
                            visible=False
                        )

                    # データセット情報
                    with gr.Accordion(
                        "📊 データセット情報", open=True
                    ):
                        with gr.Row():
                            stats_refresh_btn = gr.Button(
                                "🔄 更新",
                                size="sm",
                                scale=0,
                            )
                        stats_json = gr.JSON(
                            value={},
                            label="統計情報",
                        )

                        # Refresh button click handler
                        stats_refresh_btn.click(
                            fn=self._get_current_stats,
                            inputs=[],
                            outputs=[stats_json],
                        )

                # 右パネル: 視覚化
                with gr.Column(scale=2):
                    gr.Markdown("## 🎴 盤面表示")

                    # ボード表示（SVG）
                    board_display = gr.HTML(
                        value=self._get_default_board_svg(),
                        label="盤面",
                        elem_id="board-display",
                    )

                    # 選択中のレコードID（コピー用）
                    selected_record_id = gr.Textbox(
                        value="",
                        label="選択中のID（部分選択してコピー可能）",
                        interactive=False,
                        elem_id="selected-record-id",
                    )

                    # タブ式レコード詳細表示
                    with gr.Tabs():
                        with gr.Tab("📋 概要"):
                            record_details = gr.JSON(
                                label="レコード詳細",
                            )

                        with gr.Tab("📊 検索結果"):
                            # Rendererから動的にヘッダーを取得
                            table_headers: list[str] = (
                                self.viz_interface.get_table_columns()
                                if self.viz_interface
                                is not None
                                else []
                            )

                            results_table = gr.Dataframe(
                                headers=table_headers
                                if table_headers
                                else None,
                                label="結果一覧",
                                interactive=False,
                                elem_id="search-results-table",
                            )

                        with gr.Tab("📈 データ分析"):
                            analytics_chart = gr.Plot(
                                value=None,
                                label="データ分析チャート",
                            )

            # --- ゲームツリー UI (game-tree) ---
            is_tree_mode = self.array_type == "game-tree"
            game_tree_panel = gr.Column(visible=is_tree_mode)
            with game_tree_panel:
                gt_info = gr.Markdown(
                    value="ゲームツリーデータを読み込んでください",
                )
                with gr.Row():
                    gt_depth_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="表示深さ",
                        scale=1,
                    )
                    gt_min_prob_slider = gr.Slider(
                        minimum=0.001,
                        maximum=0.3,
                        value=0.01,
                        step=0.001,
                        label="最小確率",
                        scale=1,
                    )
                    gt_refresh_btn = gr.Button(
                        "更新",
                        variant="primary",
                        scale=0,
                    )
                    gt_back_btn = gr.Button(
                        "ルートに戻る",
                        variant="secondary",
                        scale=0,
                    )
                # パンくずリスト
                gt_breadcrumb_html = gr.HTML(
                    value='<div class="breadcrumb-nav"></div>',
                    label="パンくずリスト",
                    show_label=False,
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        gt_tree_html = gr.HTML(
                            label="ツリー表示",
                            elem_id="tree-view",
                        )
                    with gr.Column(scale=2):
                        gt_board_html = gr.HTML(
                            label="盤面",
                        )
                        gt_stats_json = gr.JSON(
                            label="局面統計",
                        )
                        gt_move_table = gr.Dataframe(
                            headers=["指し手", "確率", "勝率"],
                            label="指し手一覧",
                            interactive=False,
                        )
                        gt_analytics_plot = gr.Plot(
                            label="分岐分析",
                        )
                        with gr.Accordion(
                            "エクスポート", open=False
                        ):
                            gt_sfen_text = gr.Textbox(
                                label="USI position文字列",
                                interactive=False,
                                lines=2,
                            )

                # Hidden state for game tree
                # NOTE: visible="hidden" は Gradio 5.36+ / 6.x で
                # 追加されたオプションで，コンポーネントをDOMに残しつつ
                # 視覚的に非表示にする．visible=False は Svelte の条件
                # レンダリング({#if visible})でDOM要素を生成しないため
                # 使用不可．
                gt_selected_node = gr.Textbox(
                    label="",
                    elem_id="selected-node-id",
                    visible="hidden",  # type: ignore[arg-type]
                )
                gt_expand_node = gr.Textbox(
                    label="",
                    elem_id="expand-node-id",
                    visible="hidden",  # type: ignore[arg-type]
                )
                gt_current_root = gr.Textbox(
                    label="",
                    value="",
                    elem_id="current-root",
                    visible="hidden",  # type: ignore[arg-type]
                )
                # Hidden buttons (JSからクリックしてGradioコールバックを発火)
                gt_select_trigger = gr.Button(
                    value="",
                    elem_id="node-select-trigger",
                    visible="hidden",  # type: ignore[arg-type]
                )
                gt_expand_trigger = gr.Button(
                    value="",
                    elem_id="node-expand-trigger",
                    visible="hidden",  # type: ignore[arg-type]
                )

            # イベントハンドラとState変数
            current_page = gr.State(value=1)
            # ページ内ナビゲーション用のState
            current_page_records = gr.State(value=[])
            current_record_index = gr.State(value=0)
            # インデックス状態遷移検出用のState
            previous_indexing_status = gr.State(value="idle")

            # 初回表示時にページ1をロード（全データ型で実行）
            # バッジとステータスメッセージも更新してサーバー状態と同期する
            demo.load(
                fn=self._initial_page_load,
                inputs=[
                    min_eval,
                    max_eval,
                    current_page,
                    page_size,
                ],
                outputs=[
                    results_table,
                    page_info,
                    board_display,
                    record_details,
                    current_page_records,  # キャッシュ
                    current_record_index,  # インデックス
                    record_indicator,  # インジケーター
                    analytics_chart,  # 分析チャート
                    prev_btn,  # ページ前へボタン状態
                    next_btn,  # ページ次へボタン状態
                    prev_record_btn,  # レコード前へボタン状態
                    next_record_btn,  # レコード次へボタン状態
                    selected_record_id,  # 選択中のレコードID
                    mode_badge,  # バッジをサーバー状態と同期
                    status_markdown,  # ステータスメッセージをサーバー状態と同期
                ],
            )

            # テーブル行選択イベント
            results_table.select(
                fn=self._on_table_row_select,
                inputs=[current_page_records],
                outputs=[
                    board_display,
                    record_details,
                    selected_record_id,
                    current_record_index,
                    record_indicator,
                    prev_record_btn,
                    next_record_btn,
                ],
            )

            id_search_btn.click(
                fn=self._search_by_id,
                inputs=[id_input],
                outputs=[
                    board_display,
                    record_details,
                    selected_record_id,
                    record_indicator,
                    prev_record_btn,
                    next_record_btn,
                ],
            )

            eval_search_btn.click(
                fn=self._search_and_cache,
                inputs=[
                    min_eval,
                    max_eval,
                    current_page,
                    page_size,
                ],
                outputs=[
                    results_table,
                    page_info,
                    board_display,
                    record_details,
                    current_page_records,  # キャッシュ
                    current_record_index,  # インデックス
                    record_indicator,  # インジケーター
                    analytics_chart,  # 分析チャート
                    prev_btn,  # ページ前へボタン状態
                    next_btn,  # ページ次へボタン状態
                    prev_record_btn,  # レコード前へボタン状態
                    next_record_btn,  # レコード次へボタン状態
                    selected_record_id,  # 選択中のレコードID
                ],
            )

            # ページネーション（常に_search_and_cacheを使用）
            paginate_fn = (
                self._search_and_cache
                if self.supports_eval_search
                else self._paginate_all_data
            )

            next_btn.click(
                fn=lambda page, min_eval, max_eval, page_size: (
                    min(
                        page + 1,
                        self._calculate_total_pages(
                            min_eval, max_eval, page_size
                        ),
                    )
                ),
                inputs=[
                    current_page,
                    min_eval,
                    max_eval,
                    page_size,
                ],
                outputs=[current_page],
            ).then(
                fn=paginate_fn,
                inputs=[
                    min_eval,
                    max_eval,
                    current_page,
                    page_size,
                ],
                outputs=[
                    results_table,
                    page_info,
                    board_display,
                    record_details,
                    current_page_records,  # キャッシュ
                    current_record_index,  # インデックス
                    record_indicator,  # インジケーター
                    analytics_chart,  # 分析チャート
                    prev_btn,  # ページ前へボタン状態
                    next_btn,  # ページ次へボタン状態
                    prev_record_btn,  # レコード前へボタン状態
                    next_record_btn,  # レコード次へボタン状態
                    selected_record_id,  # 選択中のレコードID
                ],
            )

            prev_btn.click(
                fn=lambda page: max(1, page - 1),
                inputs=[current_page],
                outputs=[current_page],
            ).then(
                fn=paginate_fn,
                inputs=[
                    min_eval,
                    max_eval,
                    current_page,
                    page_size,
                ],
                outputs=[
                    results_table,
                    page_info,
                    board_display,
                    record_details,
                    current_page_records,  # キャッシュ
                    current_record_index,  # インデックス
                    record_indicator,  # インジケーター
                    analytics_chart,  # 分析チャート
                    prev_btn,  # ページ前へボタン状態
                    next_btn,  # ページ次へボタン状態
                    prev_record_btn,  # レコード前へボタン状態
                    next_record_btn,  # レコード次へボタン状態
                    selected_record_id,  # 選択中のレコードID
                ],
            )

            # ページ内レコードナビゲーション（ページ境界を跨ぐ）
            next_record_btn.click(
                fn=self._navigate_next_record,
                inputs=[
                    current_page,
                    current_record_index,
                    current_page_records,
                    page_size,
                    min_eval,
                    max_eval,
                ],
                outputs=[
                    current_page,
                    current_record_index,
                    results_table,
                    page_info,
                    board_display,
                    record_details,
                    current_page_records,
                    record_indicator,
                    analytics_chart,
                    prev_record_btn,  # レコード前へボタン状態
                    next_record_btn,  # レコード次へボタン状態
                    selected_record_id,  # 選択中のレコードID
                ],
            )

            prev_record_btn.click(
                fn=self._navigate_prev_record,
                inputs=[
                    current_page,
                    current_record_index,
                    current_page_records,
                    page_size,
                    min_eval,
                    max_eval,
                ],
                outputs=[
                    current_page,
                    current_record_index,
                    results_table,
                    page_info,
                    board_display,
                    record_details,
                    current_page_records,
                    record_indicator,
                    analytics_chart,
                    prev_record_btn,  # レコード前へボタン状態
                    next_record_btn,  # レコード次へボタン状態
                    selected_record_id,  # 選択中のレコードID
                ],
            )

            # ID入力時の候補動的更新
            id_input.change(
                fn=self._get_id_suggestions_handler,
                inputs=[id_input],
                outputs=[id_input],
            )

            # パス候補イベントハンドラ
            dir_input.change(
                fn=self._get_directory_suggestions_handler,
                inputs=[dir_input],
                outputs=[dir_input],
            )

            files_input.change(
                fn=self._get_file_suggestions_handler,
                inputs=[files_input],
                outputs=[files_input],
            )

            # データソース管理イベントハンドラ

            # Event 1: Toggle between directory and file list inputs
            source_mode.change(
                fn=lambda mode: (
                    gr.update(visible=(mode == "Directory")),
                    gr.update(visible=(mode == "File list")),
                ),
                inputs=[source_mode],
                outputs=[dir_input, files_input],
            )

            # Event 2: Load new data source
            load_result = load_btn.click(
                fn=self._load_new_data_source,
                inputs=[
                    source_mode,
                    dir_input,
                    files_input,
                    array_type_dropdown,
                ],
                outputs=[
                    status_markdown,
                    rebuild_btn,
                    mode_badge,
                    status_timer,
                    record_browser_panel,
                    game_tree_panel,
                ],
            )

            # Event 3: After successful load, reload first page
            # current_pageもリセットして状態を一貫させる
            load_result.then(
                fn=lambda: (
                    (
                        1,
                        *self._paginate_all_data(
                            min_eval=-9999,
                            max_eval=9999,
                            page=1,
                            page_size=20,
                        ),
                    )
                    if self.has_data
                    else (1, *self._get_empty_state_outputs())
                ),
                inputs=[],
                outputs=[
                    current_page,  # ページ番号を1にリセット
                    results_table,
                    page_info,
                    board_display,
                    record_details,
                    current_page_records,  # キャッシュ
                    current_record_index,  # インデックス
                    record_indicator,  # インジケーター
                    analytics_chart,  # 分析チャート
                    prev_btn,  # ページ前へボタン状態
                    next_btn,  # ページ次へボタン状態
                    prev_record_btn,  # レコード前へボタン状態
                    next_record_btn,  # レコード次へボタン状態
                    selected_record_id,  # 選択中のレコードID
                ],
            )

            # Event 4: Rebuild index (background processing)
            # Auto-refresh is handled by Event 6 (timer polls status transitions)
            rebuild_btn.click(
                fn=self._rebuild_index,
                inputs=[],
                outputs=[
                    status_markdown,
                    rebuild_btn,
                    mode_badge,
                    status_timer,
                ],
            )

            # Event 5.5: Manual refresh button
            refresh_btn.click(
                fn=lambda sz: self._paginate_all_data(
                    min_eval=-9999,
                    max_eval=9999,
                    page=1,
                    page_size=sz,
                ),
                inputs=[page_size],
                outputs=[
                    results_table,
                    page_info,
                    board_display,
                    record_details,
                    current_page_records,
                    current_record_index,
                    record_indicator,
                    analytics_chart,
                    prev_btn,
                    next_btn,
                    prev_record_btn,
                    next_record_btn,
                    selected_record_id,  # 選択中のレコードID
                ],
            )

            # Event 6: Status polling timer with auto-refresh on completion
            # (Event 7 merged: data components updated directly on indexing→ready)
            status_timer.tick(
                fn=self._check_indexing_status_with_transition,
                inputs=[previous_indexing_status, page_size],
                outputs=[
                    status_markdown,
                    load_btn,
                    rebuild_btn,
                    refresh_btn,  # リフレッシュボタン状態
                    mode_badge,
                    previous_indexing_status,  # 現在の状態を保存
                    data_source_accordion,  # アコーディオン展開/閉じ制御
                    status_timer,  # タイマー動的制御
                    # Data components (updated on indexing→ready transition)
                    results_table,
                    page_info,
                    board_display,
                    record_details,
                    current_page_records,
                    current_record_index,
                    record_indicator,
                    analytics_chart,
                    prev_btn,
                    next_btn,
                    prev_record_btn,
                    next_record_btn,
                    selected_record_id,  # 選択中のレコードID
                    stats_json,
                ],
            )

            # --- Game Tree Event Handlers ---

            def _gt_update_tree(
                display_depth: int,
                min_prob: float,
                current_root: str,
            ) -> tuple[
                str, str, str, dict, list, Any, str, str
            ]:
                """ゲームツリーの更新コールバック．

                Returns:
                    (tree_html, board_svg, info, stats, moves, plot,
                     breadcrumb_html, sfen_text)
                """
                from maou.infra.visualization.game_tree_server import (
                    _build_breadcrumb_html,
                    _build_tree_html,
                    _create_analytics_plot,
                    _create_empty_plot,
                )

                viz = self._game_tree_viz
                if viz is None:
                    return (
                        "",
                        "",
                        "",
                        {},
                        [],
                        _create_empty_plot(),
                        "",
                        "",
                    )

                rh = (
                    int(current_root)
                    if current_root
                    else self._game_tree_root_hash
                )
                import json as _json

                elements = viz.get_cytoscape_elements(
                    rh, int(display_depth), min_prob
                )
                tree_html = _build_tree_html(
                    _json.dumps(elements, ensure_ascii=False)
                )
                board_svg = viz.get_board_svg(rh)
                stats = viz.get_node_stats(rh)
                moves = viz.get_move_table(rh)
                analytics = viz.get_analytics_data(rh)
                plot = _create_analytics_plot(analytics)
                if plot is None:
                    plot = _create_empty_plot()

                breadcrumb = viz.get_breadcrumb_data(rh)
                breadcrumb_html = _build_breadcrumb_html(
                    breadcrumb
                )
                sfen_text = viz.export_sfen_path(rh)

                info = (
                    f"Nodes: **{len(viz._query.nodes_df):,}** / "
                    f"Edges: **{len(viz._query.edges_df):,}** / "
                    f"Root: `0x{rh:016X}`"
                )
                return (
                    tree_html,
                    board_svg,
                    info,
                    stats,
                    moves,
                    plot,
                    breadcrumb_html,
                    sfen_text,
                )

            def _gt_on_node_selected(
                node_id: str,
            ) -> tuple[str, dict, list, Any, str, str]:
                """ノードクリック時のコールバック．

                Returns:
                    (board_svg, stats, moves, plot,
                     breadcrumb_html, sfen_text)
                """
                from maou.infra.visualization.game_tree_server import (
                    _build_breadcrumb_html,
                    _create_analytics_plot,
                    _create_empty_plot,
                )

                viz = self._game_tree_viz
                if not node_id or viz is None:
                    return (
                        "",
                        {},
                        [],
                        _create_empty_plot(),
                        "",
                        "",
                    )

                pos_hash = int(node_id)
                board_svg = viz.get_board_svg(pos_hash)
                stats = viz.get_node_stats(pos_hash)
                moves = viz.get_move_table(pos_hash)
                analytics = viz.get_analytics_data(pos_hash)
                plot = _create_analytics_plot(analytics)
                if plot is None:
                    plot = _create_empty_plot()

                breadcrumb = viz.get_breadcrumb_data(pos_hash)
                breadcrumb_html = _build_breadcrumb_html(
                    breadcrumb
                )
                sfen_text = viz.export_sfen_path(pos_hash)

                return (
                    board_svg,
                    stats,
                    moves,
                    plot,
                    breadcrumb_html,
                    sfen_text,
                )

            def _gt_on_node_expanded(
                node_id: str,
                display_depth: int,
                min_prob: float,
            ) -> tuple[
                str, str, str, str, dict, list, Any, str, str
            ]:
                """ノード展開時のコールバック．

                Returns:
                    (tree_html, board_svg, current_root, info,
                     stats, moves, plot, breadcrumb_html, sfen_text)
                """
                from maou.infra.visualization.game_tree_server import (
                    _create_empty_plot,
                )

                viz = self._game_tree_viz
                if not node_id or viz is None:
                    return (
                        "",
                        "",
                        str(self._game_tree_root_hash),
                        "",
                        {},
                        [],
                        _create_empty_plot(),
                        "",
                        "",
                    )
                result = _gt_update_tree(
                    display_depth, min_prob, node_id
                )
                # _gt_update_tree は
                # (tree, board, info, stats, moves, plot, bc, sfen)
                # を返す．expand では current_root も更新するため
                # info の前に node_id を挿入する．
                return (
                    result[0],  # tree_html
                    result[1],  # board_svg
                    node_id,  # current_root (新しいルートhash)
                    result[2],  # info
                    result[3],  # stats
                    result[4],  # moves
                    result[5],  # plot
                    result[6],  # breadcrumb_html
                    result[7],  # sfen_text
                )

            def _gt_on_back_to_root(
                display_depth: int,
                min_prob: float,
            ) -> tuple[
                str, str, str, str, dict, list, Any, str, str
            ]:
                """ルートに戻るボタンのコールバック．"""
                root = str(self._game_tree_root_hash)
                result = _gt_update_tree(
                    display_depth, min_prob, root
                )
                return (
                    result[0],  # tree_html
                    result[1],  # board_svg
                    root,  # current_root
                    result[2],  # info
                    result[3],  # stats
                    result[4],  # moves
                    result[5],  # plot
                    result[6],  # breadcrumb_html
                    result[7],  # sfen_text
                )

            # _gt_update_tree が返す 8 要素の出力先
            # (tree_html, board_svg, info, stats, moves, plot,
            #  breadcrumb_html, sfen_text)
            _gt_tree_outputs = [
                gt_tree_html,
                gt_board_html,
                gt_info,
                gt_stats_json,
                gt_move_table,
                gt_analytics_plot,
                gt_breadcrumb_html,
                gt_sfen_text,
            ]

            # Load後にゲームツリーを初期表示
            load_result.then(
                fn=lambda depth, prob: (
                    _gt_update_tree(
                        depth,
                        prob,
                        str(self._game_tree_root_hash),
                    )
                    if self._game_tree_viz is not None
                    else (
                        "",
                        "",
                        "",
                        {},
                        [],
                        None,
                        "",
                        "",
                    )
                ),
                inputs=[gt_depth_slider, gt_min_prob_slider],
                outputs=_gt_tree_outputs,
            )

            gt_refresh_btn.click(
                fn=_gt_update_tree,
                inputs=[
                    gt_depth_slider,
                    gt_min_prob_slider,
                    gt_current_root,
                ],
                outputs=_gt_tree_outputs,
            )

            # js パラメータで DOM から直接値を読み取り，Svelte の
            # 内部状態同期の不整合を回避する．
            _js_read_selected = (
                "(nodeId) => "
                "document.querySelector("
                "'#selected-node-id textarea, "
                "#selected-node-id input')?.value || nodeId"
            )
            _js_read_expand = (
                "(nodeId, depth, prob) => ["
                "document.querySelector("
                "'#expand-node-id textarea, "
                "#expand-node-id input')?.value || nodeId, "
                "depth, prob]"
            )

            # ノード選択(シングルクリック) - hidden buttonクリックで発火
            gt_select_trigger.click(
                fn=_gt_on_node_selected,
                inputs=[gt_selected_node],
                outputs=[
                    gt_board_html,
                    gt_stats_json,
                    gt_move_table,
                    gt_analytics_plot,
                    gt_breadcrumb_html,
                    gt_sfen_text,
                ],
                js=_js_read_selected,
            )

            # expand / back_to_root 共通の出力先
            # (select と異なり tree_html, current_root, info を含む)
            _gt_expand_outputs = [
                gt_tree_html,
                gt_board_html,
                gt_current_root,
                gt_info,
                gt_stats_json,
                gt_move_table,
                gt_analytics_plot,
                gt_breadcrumb_html,
                gt_sfen_text,
            ]

            # ノード展開(ダブルクリック/パンくず) - hidden buttonで発火
            gt_expand_trigger.click(
                fn=_gt_on_node_expanded,
                inputs=[
                    gt_expand_node,
                    gt_depth_slider,
                    gt_min_prob_slider,
                ],
                outputs=_gt_expand_outputs,
                js=_js_read_expand,
            )

            gt_back_btn.click(
                fn=_gt_on_back_to_root,
                inputs=[
                    gt_depth_slider,
                    gt_min_prob_slider,
                ],
                outputs=_gt_expand_outputs,
            )

            # 初回ロード: game-treeモードの場合はツリーを表示
            if (
                self.array_type == "game-tree"
                and self._game_tree_viz is not None
            ):
                demo.load(
                    fn=lambda depth, prob: _gt_update_tree(
                        depth,
                        prob,
                        str(self._game_tree_root_hash),
                    ),
                    inputs=[
                        gt_depth_slider,
                        gt_min_prob_slider,
                    ],
                    outputs=[
                        gt_tree_html,
                        gt_board_html,
                        gt_info,
                        gt_stats_json,
                        gt_move_table,
                        gt_analytics_plot,
                    ],
                )

        return demo

    def _search_and_cache(
        self,
        min_eval: int | None,
        max_eval: int | None,
        page: int,
        page_size: int,
    ) -> tuple[
        list[list[Any]],
        str,
        str,
        dict[str, Any],
        list[dict[str, Any]],
        int,
        str,
        Any | None,  # analytics_figure (Plotly Figure or None)
        gr.Button,
        gr.Button,
        gr.Button,
        gr.Button,
        str,  # selected_record_id
    ]:
        """検索を実行し，レコードをキャッシュするラッパー関数．

        ページ内ナビゲーション用にレコードをキャッシュし，
        レコードインジケーターを初期化する．

        Args:
            min_eval: 最小評価値
            max_eval: 最大評価値
            page: ページ番号
            page_size: ページサイズ

        Returns:
            (table_data, page_info, board_svg, details,
             cached_records, record_index, record_indicator, analytics_figure,
             prev_btn_state, next_btn_state,
             prev_record_btn_state, next_record_btn_state, selected_record_id)
        """
        # Thread-safe access to viz_interface
        with self._index_lock:
            # Check for empty state
            if not self.has_data or self.viz_interface is None:
                return self._get_empty_state_outputs()

            (
                table_data,
                page_info,
                board_svg,
                details,
                cached_records,
            ) = self.viz_interface.search_by_eval_range(
                min_eval=min_eval,
                max_eval=max_eval,
                page=page,
                page_size=page_size,
            )

            # レコードインジケーター初期化
            num_records = len(cached_records)
            if num_records > 0:
                record_indicator = f"Record 1 / {num_records}"
            else:
                record_indicator = "Record 0 / 0"

            # 分析チャート生成（Plotly Figure）
            analytics_figure = (
                self.viz_interface.generate_analytics(
                    cached_records
                )
            )

            # ページボタン状態を計算
            prev_page_interactive, next_page_interactive = (
                self._get_button_states(
                    page, min_eval, max_eval, page_size
                )
            )

            # レコードナビゲーションボタン状態を計算
            prev_record_interactive, next_record_interactive = (
                self._get_record_nav_button_states(
                    page,
                    0,  # 初期はインデックス0
                    num_records,
                    min_eval,
                    max_eval,
                    page_size,
                )
            )

            # 最初のレコードのIDを取得
            first_record_id = (
                str(cached_records[0].get("id", ""))
                if cached_records
                else ""
            )

            # テーブルヘッダーを取得
            table_headers = (
                self.viz_interface.get_table_columns()
            )

            # table_dataの代わりにgr.update()で返す
            table_update = gr.update(
                value=table_data,
                headers=table_headers,
            )

        return (
            table_update,
            page_info,
            board_svg,
            details,
            cached_records,  # キャッシュ
            0,  # record_indexをリセット
            record_indicator,  # インジケーター
            analytics_figure,  # 分析チャート（Plotly Figure）
            gr.Button(
                interactive=prev_page_interactive
            ),  # prev_btn状態
            gr.Button(
                interactive=next_page_interactive
            ),  # next_btn状態
            gr.Button(
                interactive=prev_record_interactive
            ),  # prev_record_btn状態
            gr.Button(
                interactive=next_record_interactive
            ),  # next_record_btn状態
            first_record_id,  # selected_record_id
        )

    def _paginate_all_data(
        self,
        min_eval: int | None,
        max_eval: int | None,
        page: int,
        page_size: int,
    ) -> tuple[
        list[list[Any]],
        str,
        str,
        dict[str, Any],
        list[dict[str, Any]],
        int,
        str,
        Any | None,  # analytics_figure (Plotly Figure or None)
        gr.Button,
        gr.Button,
        gr.Button,
        gr.Button,
        str,  # selected_record_id
    ]:
        """全データをページネーション（評価値フィルタなし）．

        stage1, stage2, preprocessingなどの非HCPEデータ用．
        min_eval, max_evalパラメータは無視される（Gradio UIの互換性のため）．

        Args:
            min_eval: 無視される（互換性のため，常にNoneとして扱う）
            max_eval: 無視される（互換性のため，常にNoneとして扱う）
            page: ページ番号（1始まり）
            page_size: ページサイズ

        Returns:
            (table_data, page_info, board_svg, details,
             cached_records, record_index, record_indicator, analytics_figure,
             prev_btn_state, next_btn_state,
             prev_record_btn_state, next_record_btn_state, selected_record_id)
        """
        # 評価値パラメータを明示的にNoneにして全データを取得
        # （引数のmin_eval, max_evalは無視）
        return self._search_and_cache(
            min_eval=None,  # 評価値フィルタなし
            max_eval=None,  # 評価値フィルタなし
            page=page,
            page_size=page_size,
        )

    def _initial_page_load(
        self,
        min_eval: int | None,
        max_eval: int | None,
        page: int,
        page_size: int,
    ) -> tuple[
        list[list[Any]],
        str,
        str,
        dict[str, Any],
        list[dict[str, Any]],
        int,
        str,
        Any | None,
        gr.Button,
        gr.Button,
        gr.Button,
        gr.Button,
        str,
        str,  # mode_badge
        str,  # status_message
    ]:
        """初回ページロード時にデータとステータスを更新．

        demo.loadイベントで呼び出され，データのページネーションと
        モードバッジ，ステータスメッセージの更新を行う．

        Returns:
            _paginate_all_dataの戻り値 + mode_badge HTML + status_message
        """
        # データをページネーション
        paginate_result = self._paginate_all_data(
            min_eval=min_eval,
            max_eval=max_eval,
            page=page,
            page_size=page_size,
        )

        # 現在のステータスに基づいてバッジとステータスメッセージを取得
        status_msg, _, _, mode_badge = (
            self._check_indexing_status()
        )

        return (*paginate_result, mode_badge, status_msg)

    def _on_table_row_select(
        self,
        evt: gr.SelectData,
        current_page_records: list[dict[str, Any]],
    ) -> tuple[
        str, dict[str, Any], str, int, str, gr.Button, gr.Button
    ]:
        """テーブル行選択時のハンドラ．

        Args:
            evt: Gradio SelectDataイベント（行インデックスを含む）
            current_page_records: 現在のページのレコードキャッシュ

        Returns:
            (board_svg, record_details, selected_id, record_index,
             record_indicator, prev_record_btn, next_record_btn)
        """
        if (
            self.viz_interface is None
            or not current_page_records
        ):
            return (
                self._render_empty_board_placeholder(),
                {"message": "No record selected"},
                "",
                0,
                gr.skip(),
                gr.skip(),
                gr.skip(),
            )

        # evt.index[0]が行インデックス
        # Gradio 6.0+ではevt.indexがtuple, list, intのいずれかで返される
        row_index = (
            evt.index[0]
            if isinstance(evt.index, (tuple, list))
            else evt.index
        )

        if row_index < 0 or row_index >= len(
            current_page_records
        ):
            return (
                self._render_empty_board_placeholder(),
                {"message": "Invalid row index"},
                "",
                0,
                gr.skip(),
                gr.skip(),
                gr.skip(),
            )

        record = current_page_records[row_index]
        board_svg = self.viz_interface.renderer.render_board(
            record
        )
        details = (
            self.viz_interface.renderer.extract_display_fields(
                record
            )
        )
        record_id = str(record.get("id", ""))
        num_records = len(current_page_records)
        record_indicator = (
            f"Record {row_index + 1} / {num_records}"
        )

        return (
            board_svg,
            details,
            record_id,
            row_index,
            record_indicator,
            gr.Button(interactive=True),
            gr.Button(interactive=True),
        )

    def _search_by_id(
        self, record_id: str
    ) -> tuple[
        str, dict[str, Any], str, str, gr.Button, gr.Button
    ]:
        """ID検索のラッパー関数（viz_interfaceがNoneの場合をハンドリング）．

        Args:
            record_id: 検索するレコードID

        Returns:
            (board_svg, record_details, selected_record_id,
             record_indicator, prev_record_btn, next_record_btn)のタプル．
            検索失敗時はboard_svg, record_details以外はgr.skip()を返す．
        """
        # Thread-safe access to viz_interface
        with self._index_lock:
            if not self.has_data or self.viz_interface is None:
                board_svg, details = self._search_by_id_mock(
                    record_id
                )
                return (
                    board_svg,
                    details,
                    record_id,
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                )

            board_svg, details = (
                self.viz_interface.search_by_id(record_id)
            )

            # 検索失敗時（"error"キーの存在で判定）は状態を変更しない
            if "error" in details:
                return (
                    board_svg,
                    details,
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                )

            # 検索成功: ボタン無効化 + インジケータ更新
            return (
                board_svg,
                details,
                record_id,
                "ID検索: 1/1",
                gr.Button(interactive=False),
                gr.Button(interactive=False),
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

    def _search_by_id_mock(
        self, record_id: str
    ) -> tuple[str, dict[str, Any]]:
        """ID検索のモック実装．

        Args:
            record_id: 検索するレコードID

        Returns:
            (board_svg, record_details)のタプル
        """
        logger.info("Mock ID search: %s", record_id)

        # モックレスポンス
        board_svg = self._get_default_board_svg()
        record_details = {
            "message": "ID検索機能は実装中です",
            "searched_id": record_id,
            "status": "mock",
        }

        return (board_svg, record_details)

    def _search_by_eval_range_mock(
        self,
        min_eval: int,
        max_eval: int,
        page: int,
        page_size: int,
    ) -> tuple[
        list[list[Any]],
        str,
        str,
        dict[str, Any],
        list[dict[str, Any]],
    ]:
        """評価値範囲検索のモック実装．

        Args:
            min_eval: 最小評価値
            max_eval: 最大評価値
            page: ページ番号
            page_size: ページサイズ

        Returns:
            (results_table_data, page_info, board_svg, record_details, cached_records)
        """
        logger.info(
            f"Mock eval range search: [{min_eval}, {max_eval}], page={page}"
        )

        # モックテーブルデータ
        mock_results = [
            [i, f"mock_id_{i}", min_eval + i * 10, 50 + i]
            for i in range(page_size)
        ]

        # モックレコードデータ（ナビゲーション用）
        mock_records = []
        for i in range(page_size):
            mock_board = [
                [0 for _ in range(9)] for _ in range(9)
            ]
            mock_hand = [0 for _ in range(14)]

            # 簡易的な盤面（各レコードで少し異なる配置）
            mock_board[0][4] = 16 + 8  # 後手王
            mock_board[8][4] = 8  # 先手王

            # レコードごとに駒配置を変える
            if i % 3 == 0:
                mock_board[0][1] = 16 + 6  # 後手角
                mock_board[8][7] = 6  # 先手角
            elif i % 3 == 1:
                mock_board[0][7] = 16 + 7  # 後手飛車
                mock_board[8][1] = 7  # 先手飛車
            else:
                mock_board[0][1] = 16 + 6  # 後手角
                mock_board[0][7] = 16 + 7  # 後手飛車
                mock_board[8][7] = 6  # 先手角
                mock_board[8][1] = 7  # 先手飛車

            mock_record = {
                "id": f"mock_id_{i}",
                "eval": min_eval + i * 10,
                "moves": 50 + i,
                "boardIdPositions": mock_board,
                "piecesInHand": mock_hand,
            }
            mock_records.append(mock_record)

        page_info = f"ページ {page} （モックデータ）"
        board_svg = self._get_default_board_svg()
        record_details = {
            "message": "範囲検索機能は実装中です",
            "min_eval": min_eval,
            "max_eval": max_eval,
            "status": "mock",
        }

        return (
            mock_results,
            page_info,
            board_svg,
            record_details,
            mock_records,
        )

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

    def _get_button_states(
        self,
        current_page: int,
        min_eval: int | None,
        max_eval: int | None,
        page_size: int,
    ) -> tuple[bool, bool]:
        """ページネーションボタンの有効/無効状態を計算．

        Args:
            current_page: 現在のページ番号
            min_eval: 最小評価値（HCPEのみ）
            max_eval: 最大評価値（HCPEのみ）
            page_size: ページサイズ

        Returns:
            (prev_interactive, next_interactive)のタプル．
            Trueは有効，Falseは無効を表す．
        """
        total_pages = self._calculate_total_pages(
            min_eval, max_eval, page_size
        )

        prev_interactive = current_page > 1
        next_interactive = current_page < total_pages

        return (prev_interactive, next_interactive)

    def _get_record_nav_button_states(
        self,
        current_page: int,
        current_record_index: int,
        num_records_on_page: int,
        min_eval: int | None,
        max_eval: int | None,
        page_size: int,
    ) -> tuple[bool, bool]:
        """レコードナビゲーションボタンの有効/無効状態を計算．

        Args:
            current_page: 現在のページ番号
            current_record_index: 現在のレコードインデックス
            num_records_on_page: 現在のページのレコード数
            min_eval: 最小評価値（HCPEのみ）
            max_eval: 最大評価値（HCPEのみ）
            page_size: ページサイズ

        Returns:
            (prev_interactive, next_interactive)のタプル．
            Trueは有効，Falseは無効を表す．
        """
        total_pages = self._calculate_total_pages(
            min_eval, max_eval, page_size
        )

        # 最初のページの最初のレコードなら前へボタンを無効化
        is_first_record = (
            current_page == 1 and current_record_index == 0
        )
        prev_interactive = not is_first_record

        # 最後のページの最後のレコードなら次へボタンを無効化
        is_last_record = (
            current_page == total_pages
            and current_record_index == num_records_on_page - 1
        )
        next_interactive = not is_last_record

        return (prev_interactive, next_interactive)

    def _navigate_next_record(
        self,
        current_page: int,
        current_record_index: int,
        current_page_records: list[dict[str, Any]],
        page_size: int,
        min_eval: int | None,
        max_eval: int | None,
    ) -> tuple[
        int,
        int,
        list[list[Any]],
        str,
        str,
        dict[str, Any],
        list[dict[str, Any]],
        str,
        Any | None,  # analytics_figure (Plotly Figure or None)
        gr.Button,
        gr.Button,
        str,  # selected_record_id
    ]:
        """次のレコードへナビゲート（ページ境界を跨ぐ）．

        Args:
            current_page: 現在のページ番号
            current_record_index: 現在のレコードインデックス
            current_page_records: 現在のページのレコードキャッシュ
            page_size: ページサイズ
            min_eval: 最小評価値（HCPEのみ）
            max_eval: 最大評価値（HCPEのみ）

        Returns:
            (new_page, new_index, table_data, page_info,
             board_svg, details, cached_records, record_indicator, analytics_figure,
             prev_record_btn_state, next_record_btn_state, selected_record_id)
        """
        if self.viz_interface is None:
            return self._get_empty_state_navigation()

        num_records = len(current_page_records)
        total_pages = self._calculate_total_pages(
            min_eval, max_eval, page_size
        )

        # ページ内で次のレコードがある場合
        if current_record_index < num_records - 1:
            new_index = current_record_index + 1
            board_svg, details = (
                self.viz_interface.navigate_within_page(
                    current_page_records, new_index
                )
            )
            record_indicator = (
                f"Record {new_index + 1} / {num_records}"
            )

            # レコードナビゲーションボタン状態を計算
            prev_interactive, next_interactive = (
                self._get_record_nav_button_states(
                    current_page,
                    new_index,
                    num_records,
                    min_eval,
                    max_eval,
                    page_size,
                )
            )

            # 新しいインデックスのレコードIDを取得
            record_id = str(
                current_page_records[new_index].get("id", "")
            )

            # ページ内ナビゲーション — テーブル・チャート再計算を排除
            # NOTE: gr.State (current_page, current_page_records) は既存値を返す
            # NOTE: UIコンポーネント (table, page_info, analytics) は gr.skip()
            return (
                current_page,
                new_index,
                gr.skip(),  # results_table — ページ内では変化なし
                gr.skip(),  # page_info — 変化なし
                board_svg,
                details,
                current_page_records,
                record_indicator,
                gr.skip(),  # analytics_chart — ページ単位統計，変化なし
                gr.Button(interactive=prev_interactive),
                gr.Button(interactive=next_interactive),
                record_id,  # selected_record_id
            )

        # ページ境界チェック：最後のページの最後のレコードなら停止
        if current_page >= total_pages:
            # 最後のページの最後のレコード：何もしない（境界で止める）
            board_svg, details = (
                self.viz_interface.navigate_within_page(
                    current_page_records, current_record_index
                )
            )
            record_indicator = f"Record {current_record_index + 1} / {num_records}"

            # 現在のレコードIDを取得
            record_id = (
                str(
                    current_page_records[
                        current_record_index
                    ].get("id", "")
                )
                if current_page_records
                else ""
            )

            # 境界条件 — テーブル・チャート再計算を排除
            # NOTE: gr.State は既存値を返す，UIコンポーネントは gr.skip()
            return (
                current_page,
                current_record_index,
                gr.skip(),  # results_table
                gr.skip(),  # page_info
                board_svg,
                details,
                current_page_records,
                record_indicator,
                gr.skip(),  # analytics_chart
                gr.Button(
                    interactive=True
                ),  # prev_record_btn有効
                gr.Button(
                    interactive=False
                ),  # next_record_btn無効
                record_id,  # selected_record_id
            )

        # ページ境界：次のページへ移動
        next_page = current_page + 1

        # 新しいページのデータを取得
        paginate_fn = (
            self._search_and_cache
            if self.supports_eval_search
            else self._paginate_all_data
        )

        (
            table_data,
            page_info_str,
            board_svg,
            details,
            cached_records,
            _,  # record_indexは0にリセットされる
            record_indicator,
            analytics_figure,
            _,  # prev_btn state（ページナビゲーション用）
            _,  # next_btn state（ページナビゲーション用）
            _,  # prev_record_btn state（レコードナビゲーション用）
            _,  # next_record_btn state（レコードナビゲーション用）
            first_record_id,  # selected_record_id
        ) = paginate_fn(
            min_eval, max_eval, next_page, page_size
        )

        # レコードナビゲーションボタン状態を計算
        # 新しいページの最初のレコードに移動
        new_num_records = len(cached_records)
        prev_interactive, next_interactive = (
            self._get_record_nav_button_states(
                next_page,
                0,  # 新しいページの最初のレコード
                new_num_records,
                min_eval,
                max_eval,
                page_size,
            )
        )

        return (
            next_page,
            0,  # 新しいページの最初のレコード
            table_data,
            page_info_str,
            board_svg,
            details,
            cached_records,
            record_indicator,
            analytics_figure,
            gr.Button(interactive=prev_interactive),
            gr.Button(interactive=next_interactive),
            first_record_id,  # selected_record_id
        )

    def _navigate_prev_record(
        self,
        current_page: int,
        current_record_index: int,
        current_page_records: list[dict[str, Any]],
        page_size: int,
        min_eval: int | None,
        max_eval: int | None,
    ) -> tuple[
        int,
        int,
        list[list[Any]],
        str,
        str,
        dict[str, Any],
        list[dict[str, Any]],
        str,
        Any | None,  # analytics_figure (Plotly Figure or None)
        gr.Button,
        gr.Button,
        str,  # selected_record_id
    ]:
        """前のレコードへナビゲート（ページ境界を跨ぐ）．

        Args:
            current_page: 現在のページ番号
            current_record_index: 現在のレコードインデックス
            current_page_records: 現在のページのレコードキャッシュ
            page_size: ページサイズ
            min_eval: 最小評価値（HCPEのみ）
            max_eval: 最大評価値（HCPEのみ）

        Returns:
            (new_page, new_index, table_data, page_info,
             board_svg, details, cached_records, record_indicator, analytics_figure,
             prev_record_btn_state, next_record_btn_state, selected_record_id)
        """
        if self.viz_interface is None:
            return self._get_empty_state_navigation()

        num_records = len(current_page_records)

        # ページ内で前のレコードがある場合
        if current_record_index > 0:
            new_index = current_record_index - 1
            board_svg, details = (
                self.viz_interface.navigate_within_page(
                    current_page_records, new_index
                )
            )
            record_indicator = (
                f"Record {new_index + 1} / {num_records}"
            )

            # レコードナビゲーションボタン状態を計算
            prev_interactive, next_interactive = (
                self._get_record_nav_button_states(
                    current_page,
                    new_index,
                    num_records,
                    min_eval,
                    max_eval,
                    page_size,
                )
            )

            # 新しいインデックスのレコードIDを取得
            record_id = str(
                current_page_records[new_index].get("id", "")
            )

            # ページ内ナビゲーション — テーブル・チャート再計算を排除
            # NOTE: gr.State (current_page, current_page_records) は既存値を返す
            # NOTE: UIコンポーネント (table, page_info, analytics) は gr.skip()
            return (
                current_page,
                new_index,
                gr.skip(),  # results_table — ページ内では変化なし
                gr.skip(),  # page_info — 変化なし
                board_svg,
                details,
                current_page_records,
                record_indicator,
                gr.skip(),  # analytics_chart — ページ単位統計，変化なし
                gr.Button(interactive=prev_interactive),
                gr.Button(interactive=next_interactive),
                record_id,  # selected_record_id
            )

        # ページ境界チェック：最初のページの最初のレコードなら停止
        if current_page <= 1:
            # 最初のページの最初のレコード：何もしない（境界で止める）
            board_svg, details = (
                self.viz_interface.navigate_within_page(
                    current_page_records, current_record_index
                )
            )
            record_indicator = f"Record {current_record_index + 1} / {num_records}"

            # 現在のレコードIDを取得
            record_id = (
                str(
                    current_page_records[
                        current_record_index
                    ].get("id", "")
                )
                if current_page_records
                else ""
            )

            # 境界条件 — テーブル・チャート再計算を排除
            # NOTE: gr.State は既存値を返す，UIコンポーネントは gr.skip()
            return (
                current_page,
                current_record_index,
                gr.skip(),  # results_table
                gr.skip(),  # page_info
                board_svg,
                details,
                current_page_records,
                record_indicator,
                gr.skip(),  # analytics_chart
                gr.Button(
                    interactive=False
                ),  # prev_record_btn無効
                gr.Button(
                    interactive=True
                ),  # next_record_btn有効
                record_id,  # selected_record_id
            )

        # ページ境界：前のページへ移動
        prev_page = current_page - 1

        # 新しいページのデータを取得
        paginate_fn = (
            self._search_and_cache
            if self.supports_eval_search
            else self._paginate_all_data
        )

        (
            table_data,
            page_info_str,
            board_svg,
            details,
            cached_records,
            _,  # record_indexは最後に設定される
            _,  # record_indicatorは後で更新
            analytics_figure,
            _,  # prev_btn state（ページナビゲーション用）
            _,  # next_btn state（ページナビゲーション用）
            _,  # prev_record_btn state（レコードナビゲーション用）
            _,  # next_record_btn state（レコードナビゲーション用）
            _,  # selected_record_id（後で再計算）
        ) = paginate_fn(
            min_eval, max_eval, prev_page, page_size
        )

        # 新しいページの最後のレコードを表示
        new_num_records = len(cached_records)
        if new_num_records > 0:
            new_index = new_num_records - 1
            board_svg, details = (
                self.viz_interface.navigate_within_page(
                    cached_records, new_index
                )
            )
            record_indicator = (
                f"Record {new_index + 1} / {new_num_records}"
            )
            # 最後のレコードのIDを取得
            record_id = str(
                cached_records[new_index].get("id", "")
            )
        else:
            new_index = 0
            record_indicator = "Record 0 / 0"
            record_id = ""

        # レコードナビゲーションボタン状態を計算
        # 新しいページの最後のレコードに移動
        prev_interactive, next_interactive = (
            self._get_record_nav_button_states(
                prev_page,
                new_index,
                new_num_records,
                min_eval,
                max_eval,
                page_size,
            )
        )

        return (
            prev_page,
            new_index,
            table_data,
            page_info_str,
            board_svg,
            details,
            cached_records,
            record_indicator,
            analytics_figure,
            gr.Button(interactive=prev_interactive),
            gr.Button(interactive=next_interactive),
            record_id,  # selected_record_id
        )


def launch_server(
    file_paths: list[Path],
    array_type: str,
    port: int | None,
    share: bool,
    server_name: str,
    model_path: Path | None,
    debug: bool,
    use_mock_data: bool = False,
) -> None:
    """Gradioサーバーを起動．

    array_typeに応じて異なるUIを提供する:
    - hcpe/preprocessing/stage1/stage2: レコードブラウザUI
    - game-tree: ゲームツリー可視化UI

    Args:
        file_paths: データファイルのパスリスト
        array_type: データ型
        port: サーバーポート
        share: 公開リンク作成フラグ
        server_name: サーバーバインドアドレス
        model_path: モデルファイルパス
        debug: デバッグモード
        use_mock_data: Trueの場合はモックデータを使用
    """
    # game-tree はゲームツリー専用UIにディスパッチ
    if array_type == "game-tree":
        from maou.infra.visualization.game_tree_server import (
            launch_game_tree_server,
        )

        tree_path = file_paths[0] if file_paths else None
        if tree_path is None:
            raise ValueError(
                "game-tree requires a tree data directory path"
            )
        launch_game_tree_server(
            tree_path=tree_path,
            port=port,
            share=share,
            server_name=server_name,
        )
        return

    server = GradioVisualizationServer(
        file_paths=file_paths,
        array_type=array_type,
        model_path=model_path,
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

    # ゲームツリーJS(Cytoscape.js + イベントハンドラ)をhead要素に注入．
    # gradio_server.py ではデータソース動的切替でゲームツリーが使われるため必要．
    from maou.infra.visualization.game_tree_server import (
        _build_head_scripts,
    )

    head_scripts = _build_head_scripts()

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
