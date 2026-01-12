"""Gradio UIサーバー実装（インフラ層）．

将棋データ可視化のためのGradio Webインターフェースを提供する．
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

from maou.domain.visualization.board_renderer import (  # noqa: E402
    BoardPosition,
    SVGBoardRenderer,
)
from maou.infra.visualization.search_index import (  # noqa: E402
    SearchIndex,
)
from maou.interface.path_suggestions import (  # noqa: E402
    PathSuggestionService,
)
from maou.interface.visualization import (  # noqa: E402
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
            logger.warning(f"CSS file not found: {css_path}")

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
        file_paths: List[Path],
        array_type: str,
        model_path: Optional[Path] = None,
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

        # 評価値検索をサポートするかどうかを判定
        self.supports_eval_search = self._supports_eval_search()

        # Initialize path suggestion service
        self.path_suggester = PathSuggestionService(
            cache_ttl=60
        )

        if self.has_data:
            # Build index and interface
            # SearchIndexを初期化
            self.search_index = SearchIndex.build(
                file_paths=file_paths,
                array_type=array_type,
                use_mock_data=use_mock_data,
                num_mock_records=1000,
            )

            # VisualizationInterfaceを初期化
            self.viz_interface = VisualizationInterface(
                search_index=self.search_index,
                file_paths=file_paths,
                array_type=array_type,
            )

            mode_msg = (
                "MOCK MODE (fake data)"
                if use_mock_data
                else "REAL MODE (actual data)"
            )
            logger.info(
                f"🎯 Visualization server initialized: {mode_msg}, "
                f"{len(file_paths)} files, type={array_type}, "
                f"{self.search_index.total_records()} records indexed"
            )
        else:
            # Empty state - will be initialized when user loads data
            self.search_index = None  # type: ignore[assignment]
            self.viz_interface = None  # type: ignore[assignment]
            logger.warning(
                "⚠️  No data loaded - UI will show empty state"
            )

    def _get_id_suggestions_handler(self, prefix: str) -> Any:
        """ID入力に応じて候補を動的更新．

        Args:
            prefix: ユーザーが入力したプレフィックス

        Returns:
            Dropdownの選択肢更新
        """
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
            logger.error(f"Directory suggestion failed: {e}")
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
            logger.error(f"File suggestion failed: {e}")
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
        elif self.has_data:
            total = self.search_index.total_records()
            file_count = len(self.file_paths)
            return (
                f"**Status:** 🟢 Loaded {total:,} records "
                f"from {file_count} file(s)"
            )
        else:
            return "**Status:** ⚪ No data loaded - select a data source to begin"

    def _resolve_directory(self, dir_path: str) -> List[Path]:
        """Resolve directory to list of .feather files．

        Args:
            dir_path: Directory path string from UI input

        Returns:
            List of .feather file paths sorted by name

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

        feather_files = sorted(path.glob("*.feather"))

        if not feather_files:
            raise ValueError(
                f"No .feather files found in {path}"
            )

        logger.info(
            f"Found {len(feather_files)} .feather files in {path}"
        )
        return feather_files

    def _resolve_file_list(self, files_str: str) -> List[Path]:
        """Resolve comma-separated file paths．

        Args:
            files_str: Comma-separated file paths from UI input

        Returns:
            List of validated .feather file paths

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
    ) -> Tuple[str, bool, str]:
        """Load new data source and rebuild index．

        Args:
            source_mode: "Directory" or "File List"
            dir_path: Directory path (used if source_mode == "Directory")
            files_path: Comma-separated files (used if source_mode == "File List")
            array_type: Data array type (hcpe, preprocessing, stage1, stage2)

        Returns:
            Tuple of (status_message, rebuild_btn_enabled, mode_badge)
        """
        # Step 1: Validate and resolve paths
        try:
            if source_mode == "Directory":
                file_paths = self._resolve_directory(dir_path)
            else:  # "File List"
                file_paths = self._resolve_file_list(files_path)
        except ValueError as e:
            logger.error(f"Path resolution failed: {e}")
            return (
                f"❌ **Error:** {e}",
                False,  # Keep rebuild button disabled
                '<span class="mode-badge-text">⚪ NO DATA</span>',
            )

        # Step 2: Build new SearchIndex
        try:
            logger.info(
                f"Building search index for {len(file_paths)} files..."
            )
            new_index = SearchIndex.build(
                file_paths=file_paths,
                array_type=array_type,
                use_mock_data=False,
            )
            logger.info(
                f"Index built: {new_index.total_records():,} records"
            )
        except Exception as e:
            logger.exception("Index build failed")
            return (
                f"❌ **Error:** Index build failed - {e}",
                False,
                '<span class="mode-badge-text">⚪ NO DATA</span>',
            )

        # Step 3: Create new VisualizationInterface
        try:
            new_viz_interface = VisualizationInterface(
                search_index=new_index,
                file_paths=file_paths,
                array_type=array_type,
            )
        except Exception as e:
            logger.exception(
                "VisualizationInterface creation failed"
            )
            return (
                f"❌ **Error:** Failed to create interface - {e}",
                False,
                '<span class="mode-badge-text">⚪ NO DATA</span>',
            )

        # Step 4: Update instance state
        self.file_paths = file_paths
        self.array_type = array_type
        self.search_index = new_index
        self.viz_interface = new_viz_interface
        self.has_data = True

        # Step 5: Update eval search support
        self.supports_eval_search = self._supports_eval_search()

        # Step 6: Return success status
        total = new_index.total_records()
        file_count = len(file_paths)
        success_msg = (
            f"✓ **Success:** Loaded {total:,} records "
            f"from {file_count} file(s) (type: {array_type})"
        )

        logger.info(success_msg)
        return (
            success_msg,
            True,  # Enable rebuild button
            '<span class="mode-badge-text">🟢 REAL MODE</span>',
        )

    def _rebuild_index(self) -> str:
        """Rebuild search index from current file paths．

        Returns:
            Status message string
        """
        if not self.has_data or not self.file_paths:
            logger.warning(
                "Rebuild requested but no data source is loaded"
            )
            return "❌ **Error:** No data source loaded"

        try:
            logger.info(
                f"Rebuilding index for {len(self.file_paths)} files..."
            )

            # Build new index
            new_index = SearchIndex.build(
                file_paths=self.file_paths,
                array_type=self.array_type,
                use_mock_data=False,
            )

            # Update search index
            self.search_index = new_index

            # Update viz_interface's search_index reference
            self.viz_interface.search_index = new_index

            total = new_index.total_records()
            success_msg = f"✓ **Success:** Index rebuilt - {total:,} records"

            logger.info(success_msg)
            return success_msg

        except Exception as e:
            logger.exception("Index rebuild failed")
            return f"❌ **Error:** Rebuild failed - {e}"

    def _get_empty_state_outputs(self) -> Tuple:
        """Generate output values for empty state (no data loaded)．

        Returns:
            Tuple matching outputs for pagination methods
        """
        empty_table: List[
            List[Any]
        ] = []  # Empty list for results_table

        page_info = "No data loaded"

        board_display = self._render_empty_board_placeholder()

        record_details = {
            "message": "No data loaded",
            "instruction": "Use 'Data Source Management' section to load data",
        }

        current_page = 1
        current_page_records = gr.State([])
        current_record_index = gr.State(0)

        return (
            empty_table,
            page_info,
            board_display,
            record_details,
            current_page,
            current_page_records,
            current_record_index,
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

            with gr.Row():
                # 左パネル: ナビゲーションと検索コントロール
                with gr.Column(scale=1):
                    # データソース管理セクション
                    with gr.Accordion(
                        "📂 Data Source Management",
                        open=not self.has_data,  # Expanded when no data
                    ):
                        with gr.Row():
                            source_mode = gr.Radio(
                                choices=[
                                    "Directory",
                                    "File List",
                                ],
                                value="Directory",
                                label="Source Type",
                                scale=1,
                            )

                        dir_input = gr.Dropdown(
                            label="📁 Directory Path",
                            choices=[],  # Initially empty
                            value=None,
                            allow_custom_value=True,  # Allow manual path entry
                            filterable=True,  # Enable incremental search
                            info="Type to search directories (2+ characters for suggestions)",
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
                            ],
                            value=self.array_type,
                            label="Array Type",
                            interactive=True,
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

                        status_markdown = gr.Markdown(
                            value=self._get_initial_status_message(),
                            elem_classes=["status-message"],
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
                        page_info = gr.Markdown("ページ 1")

                    # 検索機能
                    gr.Markdown("## 🔍 検索機能")

                    # ID検索
                    with gr.Group():
                        gr.Markdown("### ID検索")

                        # 初期化時にID候補リストを取得（最大1000件）
                        initial_ids = []
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
                                info="評価値の下限（例: -1000）",
                                value=-1000,
                                precision=0,
                            )
                            max_eval = gr.Number(
                                label="📈 最大評価値",
                                info="評価値の上限（例: 1000）",
                                value=1000,
                                precision=0,
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
                    with gr.Group():
                        gr.Markdown("### 📈 データセット情報")
                        gr.JSON(
                            value=self.viz_interface.get_dataset_stats(),
                            label="統計情報",
                        )

                # 右パネル: 視覚化
                with gr.Column(scale=2):
                    gr.Markdown("## 🎴 盤面表示")

                    # ボード表示（SVG）
                    board_display = gr.HTML(
                        value=self._get_default_board_svg(),
                        label="盤面",
                    )

                    # タブ式レコード詳細表示
                    with gr.Tabs():
                        with gr.Tab("📋 概要"):
                            record_details = gr.JSON(
                                label="レコード詳細",
                            )

                        with gr.Tab("📊 検索結果"):
                            # Rendererから動的にヘッダーを取得
                            table_headers = self.viz_interface.get_table_columns()

                            results_table = gr.Dataframe(
                                headers=table_headers,
                                label="結果一覧",
                                interactive=False,
                            )

                        with gr.Tab("📈 データ分析"):
                            analytics_chart = gr.HTML(
                                value="<p style='text-align: center; color: #666;'>検索を実行すると分析チャートが表示されます．</p>",
                                label="データ分析チャート",
                            )

            # イベントハンドラとState変数
            current_page = gr.State(value=1)
            # ページ内ナビゲーション用のState
            current_page_records = gr.State(value=[])
            current_record_index = gr.State(value=0)

            # 初回表示時にページ1をロード（全データ型で実行）
            demo.load(
                fn=self._paginate_all_data,
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
                ],
            )

            id_search_btn.click(
                fn=self.viz_interface.search_by_id,
                inputs=[id_input],
                outputs=[board_display, record_details],
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
                ],
            )

            # ページネーション（常に_search_and_cacheを使用）
            paginate_fn = (
                self._search_and_cache
                if self.supports_eval_search
                else self._paginate_all_data
            )

            next_btn.click(
                fn=lambda page,
                min_eval,
                max_eval,
                page_size: min(
                    page + 1,
                    self._calculate_total_pages(
                        min_eval, max_eval, page_size
                    ),
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
                    gr.update(visible=(mode == "File List")),
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
                ],
            )

            # Event 3: After successful load, reload first page
            if self.supports_eval_search:
                load_result.then(
                    fn=lambda: (
                        self._paginate_all_data(
                            min_eval=-9999,
                            max_eval=9999,
                            page=1,
                            page_size=20,
                        )
                        if self.has_data
                        else self._get_empty_state_outputs()
                    ),
                    inputs=[],
                    outputs=[
                        results_table,
                        page_info,
                        board_display,
                        record_details,
                        current_page,
                        current_page_records,
                        current_record_index,
                    ],
                )
            else:
                load_result.then(
                    fn=lambda: (
                        self._paginate_all_data(
                            min_eval=-9999,
                            max_eval=9999,
                            page=1,
                            page_size=20,
                        )
                        if self.has_data
                        else self._get_empty_state_outputs()
                    ),
                    inputs=[],
                    outputs=[
                        results_table,
                        page_info,
                        board_display,
                        record_details,
                        current_page,
                        current_page_records,
                        current_record_index,
                    ],
                )

            # Event 4: Rebuild index
            rebuild_result = rebuild_btn.click(
                fn=self._rebuild_index,
                inputs=[],
                outputs=[status_markdown],
            )

            # Event 5: After successful rebuild, reload current page
            rebuild_result.then(
                fn=lambda pg, sz: self._paginate_all_data(
                    min_eval=-9999,
                    max_eval=9999,
                    page=pg,
                    page_size=sz,
                ),
                inputs=[current_page, page_size],
                outputs=[
                    results_table,
                    page_info,
                    board_display,
                    record_details,
                    current_page,
                    current_page_records,
                    current_record_index,
                ],
            )

        return demo

    def _search_and_cache(
        self,
        min_eval: Optional[int],
        max_eval: Optional[int],
        page: int,
        page_size: int,
    ) -> Tuple[
        List[List[Any]],
        str,
        str,
        Dict[str, Any],
        List[Dict[str, Any]],
        int,
        str,
        str,
        gr.Button,
        gr.Button,
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
             cached_records, record_index, record_indicator, analytics_html,
             prev_btn_state, next_btn_state)
        """
        # Check for empty state
        if not self.has_data or self.viz_interface is None:
            return self._get_empty_state_outputs() + (
                gr.Button(interactive=False),
                gr.Button(interactive=False),
            )

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

        # 分析チャート生成
        analytics_html = self.viz_interface.generate_analytics(
            cached_records
        )

        # ボタン状態を計算
        prev_interactive, next_interactive = (
            self._get_button_states(
                page, min_eval, max_eval, page_size
            )
        )

        return (
            table_data,
            page_info,
            board_svg,
            details,
            cached_records,  # キャッシュ
            0,  # record_indexをリセット
            record_indicator,  # インジケーター
            analytics_html,  # 分析チャート
            gr.Button(
                interactive=prev_interactive
            ),  # prev_btn状態
            gr.Button(
                interactive=next_interactive
            ),  # next_btn状態
        )

    def _paginate_all_data(
        self,
        min_eval: Optional[int],
        max_eval: Optional[int],
        page: int,
        page_size: int,
    ) -> Tuple[
        List[List[Any]],
        str,
        str,
        Dict[str, Any],
        List[Dict[str, Any]],
        int,
        str,
        str,
        gr.Button,
        gr.Button,
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
             cached_records, record_index, record_indicator, analytics_html,
             prev_btn_state, next_btn_state)
        """
        # 評価値パラメータを明示的にNoneにして全データを取得
        # （引数のmin_eval, max_evalは無視）
        return self._search_and_cache(
            min_eval=None,  # 評価値フィルタなし
            max_eval=None,  # 評価値フィルタなし
            page=page,
            page_size=page_size,
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

    def _get_mock_stats(self) -> Dict[str, Any]:
        """インデックス統計情報を返す．"""
        return {
            "total_records": self.search_index.total_records(),
            "array_type": self.array_type,
            "num_files": len(self.file_paths),
        }

    def _search_by_id_mock(
        self, record_id: str
    ) -> Tuple[str, Dict[str, Any]]:
        """ID検索のモック実装．

        Args:
            record_id: 検索するレコードID

        Returns:
            (board_svg, record_details)のタプル
        """
        logger.info(f"Mock ID search: {record_id}")

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
    ) -> Tuple[
        List[List[Any]],
        str,
        str,
        Dict[str, Any],
        List[Dict[str, Any]],
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
        min_eval: Optional[int],
        max_eval: Optional[int],
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
        min_eval: Optional[int],
        max_eval: Optional[int],
        page_size: int,
    ) -> Tuple[bool, bool]:
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
        min_eval: Optional[int],
        max_eval: Optional[int],
        page_size: int,
    ) -> Tuple[bool, bool]:
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
        current_page_records: List[Dict[str, Any]],
        page_size: int,
        min_eval: Optional[int],
        max_eval: Optional[int],
    ) -> Tuple[
        int,
        int,
        List[List[Any]],
        str,
        str,
        Dict[str, Any],
        List[Dict[str, Any]],
        str,
        str,
        gr.Button,
        gr.Button,
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
             board_svg, details, cached_records, record_indicator, analytics_html,
             prev_record_btn_state, next_record_btn_state)
        """
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

            # ページは変わらないので，現在のデータを返す
            table_data = [
                self.viz_interface.renderer.format_table_row(
                    i + (current_page - 1) * page_size + 1,
                    record,
                )
                for i, record in enumerate(current_page_records)
            ]
            page_info_str = (
                f"ページ {current_page} / {total_pages}"
            )

            # ページ内ナビゲーション時はanalyticsは変わらない
            analytics_html = (
                self.viz_interface.generate_analytics(
                    current_page_records
                )
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

            return (
                current_page,
                new_index,
                table_data,
                page_info_str,
                board_svg,
                details,
                current_page_records,
                record_indicator,
                analytics_html,
                gr.Button(interactive=prev_interactive),
                gr.Button(interactive=next_interactive),
            )

        # ページ境界チェック：最後のページの最後のレコードなら停止
        if current_page >= total_pages:
            # 最後のページの最後のレコード：何もしない（境界で止める）
            table_data = [
                self.viz_interface.renderer.format_table_row(
                    i + (current_page - 1) * page_size + 1,
                    record,
                )
                for i, record in enumerate(current_page_records)
            ]
            page_info_str = (
                f"ページ {current_page} / {total_pages}"
            )
            board_svg, details = (
                self.viz_interface.navigate_within_page(
                    current_page_records, current_record_index
                )
            )
            record_indicator = f"Record {current_record_index + 1} / {num_records}"
            analytics_html = (
                self.viz_interface.generate_analytics(
                    current_page_records
                )
            )

            # ボタン状態：前へは有効，次へは無効
            return (
                current_page,
                current_record_index,
                table_data,
                page_info_str,
                board_svg,
                details,
                current_page_records,
                record_indicator,
                analytics_html,
                gr.Button(
                    interactive=True
                ),  # prev_record_btn有効
                gr.Button(
                    interactive=False
                ),  # next_record_btn無効
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
            analytics_html,
            _,  # prev_btn state（ページナビゲーション用）
            _,  # next_btn state（ページナビゲーション用）
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
            analytics_html,
            gr.Button(interactive=prev_interactive),
            gr.Button(interactive=next_interactive),
        )

    def _navigate_prev_record(
        self,
        current_page: int,
        current_record_index: int,
        current_page_records: List[Dict[str, Any]],
        page_size: int,
        min_eval: Optional[int],
        max_eval: Optional[int],
    ) -> Tuple[
        int,
        int,
        List[List[Any]],
        str,
        str,
        Dict[str, Any],
        List[Dict[str, Any]],
        str,
        str,
        gr.Button,
        gr.Button,
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
             board_svg, details, cached_records, record_indicator, analytics_html,
             prev_record_btn_state, next_record_btn_state)
        """
        num_records = len(current_page_records)
        total_pages = self._calculate_total_pages(
            min_eval, max_eval, page_size
        )

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

            # ページは変わらないので，現在のデータを返す
            table_data = [
                self.viz_interface.renderer.format_table_row(
                    i + (current_page - 1) * page_size + 1,
                    record,
                )
                for i, record in enumerate(current_page_records)
            ]
            page_info_str = (
                f"ページ {current_page} / {total_pages}"
            )

            # ページ内ナビゲーション時はanalyticsは変わらない
            analytics_html = (
                self.viz_interface.generate_analytics(
                    current_page_records
                )
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

            return (
                current_page,
                new_index,
                table_data,
                page_info_str,
                board_svg,
                details,
                current_page_records,
                record_indicator,
                analytics_html,
                gr.Button(interactive=prev_interactive),
                gr.Button(interactive=next_interactive),
            )

        # ページ境界チェック：最初のページの最初のレコードなら停止
        if current_page <= 1:
            # 最初のページの最初のレコード：何もしない（境界で止める）
            table_data = [
                self.viz_interface.renderer.format_table_row(
                    i + (current_page - 1) * page_size + 1,
                    record,
                )
                for i, record in enumerate(current_page_records)
            ]
            page_info_str = (
                f"ページ {current_page} / {total_pages}"
            )
            board_svg, details = (
                self.viz_interface.navigate_within_page(
                    current_page_records, current_record_index
                )
            )
            record_indicator = f"Record {current_record_index + 1} / {num_records}"
            analytics_html = (
                self.viz_interface.generate_analytics(
                    current_page_records
                )
            )

            # ボタン状態：前へは無効，次へは有効
            return (
                current_page,
                current_record_index,
                table_data,
                page_info_str,
                board_svg,
                details,
                current_page_records,
                record_indicator,
                analytics_html,
                gr.Button(
                    interactive=False
                ),  # prev_record_btn無効
                gr.Button(
                    interactive=True
                ),  # next_record_btn有効
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
            analytics_html,
            _,  # prev_btn state（ページナビゲーション用）
            _,  # next_btn state（ページナビゲーション用）
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
        else:
            new_index = 0
            record_indicator = "Record 0 / 0"

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
            analytics_html,
            gr.Button(interactive=prev_interactive),
            gr.Button(interactive=next_interactive),
        )


def launch_server(
    file_paths: List[Path],
    array_type: str,
    port: int,
    share: bool,
    server_name: str,
    model_path: Optional[Path],
    debug: bool,
    use_mock_data: bool = False,
) -> None:
    """Gradioサーバーを起動．

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
    server = GradioVisualizationServer(
        file_paths=file_paths,
        array_type=array_type,
        model_path=model_path,
        use_mock_data=use_mock_data,
    )

    demo = server.create_demo()

    # カスタムCSSを読み込み（Gradio 6ではlaunch()に渡す必要がある）
    custom_css = _load_custom_css()

    logger.info(
        f"Launching Gradio server on {server_name}:{port} "
        f"(share={share}, debug={debug})"
    )

    demo.launch(
        server_name=server_name,
        server_port=port,
        share=share,
        debug=debug,
        show_error=True,
        css=custom_css,
    )
