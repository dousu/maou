"""Path suggestion service (Interface Layer)．

Data Source Management UIのためのパス候補サービス．
"""

import logging
from pathlib import Path
from typing import List, Optional

try:
    from maou._rust.maou_index import PathScanner

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

logger = logging.getLogger(__name__)


class PathSuggestionService:
    """Path suggestion service with Rust backend．"""

    def __init__(self, cache_ttl: int = 60) -> None:
        """Initialize path suggestion service．

        Args:
            cache_ttl: Cache time-to-live in seconds (default: 60)
        """
        self.cache_ttl = cache_ttl
        self._initialized = False

        if RUST_AVAILABLE:
            self.scanner = PathScanner(cache_ttl)
            logger.info(
                f"PathSuggestionService initialized (Rust backend, TTL={cache_ttl}s)"
            )
        else:
            self.scanner = None
            logger.warning(
                f"PathSuggestionService initialized (Python fallback mode, TTL={cache_ttl}s) - "
                "Rust backend not available"
            )

    def get_directory_suggestions(
        self,
        prefix: str,
        base_path: Optional[Path] = None,
        limit: int = 50,
    ) -> List[str]:
        """Get directory path suggestions．

        Args:
            prefix: User input prefix (2+ characters required)
            base_path: Base directory to scan (None = infer from prefix)
            limit: Maximum suggestions to return

        Returns:
            Sorted list of matching directory paths
        """
        if not prefix or len(prefix) < 2:
            return []

        # Expand user path (e.g., ~ -> /home/user)
        prefix_expanded = str(Path(prefix).expanduser())

        # Extract base directory for scanning
        if base_path is None:
            base_path = self._infer_base_path(prefix_expanded)

        # Use Rust scanner if available
        if self.scanner is not None:
            try:
                if self.scanner.is_stale():
                    self.scanner.scan_directories(
                        str(base_path), max_depth=5
                    )
                suggestions = (
                    self.scanner.search_directory_prefix(
                        prefix_expanded, limit
                    )
                )
                return suggestions
            except Exception as e:
                logger.error(
                    f"Rust scanner failed, falling back to Python: {e}"
                )
                # Fall through to Python implementation

        # Python fallback implementation
        suggestions = self._python_directory_search(
            base_path, prefix_expanded, limit
        )

        return suggestions

    def get_file_suggestions(
        self,
        prefix: str,
        base_path: Optional[Path] = None,
        limit: int = 100,
    ) -> List[str]:
        """Get .feather file path suggestions．

        Args:
            prefix: User input prefix (2+ characters required)
            base_path: Base directory to scan
            limit: Maximum suggestions to return

        Returns:
            Sorted list of matching .feather file paths
        """
        if not prefix or len(prefix) < 2:
            return []

        # Expand user path
        prefix_expanded = str(Path(prefix).expanduser())

        # Extract base directory
        if base_path is None:
            base_path = self._infer_base_path(prefix_expanded)

        # Use Rust scanner if available
        if self.scanner is not None:
            try:
                if self.scanner.is_stale():
                    self.scanner.scan_feather_files(
                        str(base_path), recursive=True
                    )
                suggestions = self.scanner.search_file_prefix(
                    prefix_expanded, limit
                )
                return suggestions
            except Exception as e:
                logger.error(
                    f"Rust scanner failed, falling back to Python: {e}"
                )
                # Fall through to Python implementation

        # Python fallback implementation
        suggestions = self._python_file_search(
            base_path, prefix_expanded, limit
        )

        return suggestions

    def preload_directories(
        self,
        base_path: Optional[Path] = None,
        max_depth: int = 2,
        limit: int = 100,
    ) -> List[str]:
        """初期表示用のディレクトリ候補をプリロード．

        既存の get_directory_suggestions とは異なり，prefix入力なしで
        base_path以下のディレクトリを直接スキャンして返す．

        Args:
            base_path: スキャン起点（None = cwd）
            max_depth: スキャン深度（デフォルト: 2）
            limit: 最大候補数（デフォルト: 100）

        Returns:
            ソート済みディレクトリパスのリスト
        """
        if base_path is None:
            base_path = Path.cwd()

        base_str = str(base_path)

        if self.scanner is not None:
            try:
                self.scanner.scan_directories(
                    base_str, max_depth=max_depth
                )
                return self.scanner.search_directory_prefix(
                    base_str, limit
                )
            except Exception as e:
                logger.error(f"Directory preload failed: {e}")

        # Python fallback
        return self._python_directory_search(
            base_path, base_str, limit
        )

    def _infer_base_path(self, prefix: str) -> Path:
        """Infer base directory from prefix．

        Examples:
            /data/shogi/train -> /data/shogi
            ~/projects/maou -> /home/user/projects
            /non/existent/path -> /non/existent (parent)

        Args:
            prefix: User-entered path prefix

        Returns:
            Inferred base directory for scanning
        """
        path = Path(prefix)

        # If path exists and is a directory, use it as base
        if path.exists() and path.is_dir():
            return path

        # If parent exists, use parent as base
        if path.parent.exists():
            return path.parent

        # Fallback: current working directory
        return Path.cwd()

    def _python_directory_search(
        self, base_path: Path, prefix: str, limit: int
    ) -> List[str]:
        """Python fallback for directory search (temporary)．

        This will be replaced by Rust implementation．

        Args:
            base_path: Base directory to search from
            prefix: Prefix to match against
            limit: Maximum results to return

        Returns:
            List of matching directory paths
        """
        try:
            # Use rglob for recursive search (limited depth for performance)
            # Note: This is inefficient for large directories
            # Will be replaced by Rust walker
            directories = []

            # Search with limited depth to avoid hanging
            for path in base_path.rglob("*"):
                if path.is_dir():
                    path_str = str(path)
                    if path_str.startswith(prefix):
                        directories.append(path_str)

                # Limit total scanned items for performance
                if len(directories) >= limit * 10:
                    break

            # Filter prefix matches and sort
            matching = [
                d for d in directories if d.startswith(prefix)
            ]
            return sorted(matching)[:limit]

        except Exception as e:
            logger.error(f"Directory search failed: {e}")
            return []

    def _python_file_search(
        self, base_path: Path, prefix: str, limit: int
    ) -> List[str]:
        """Python fallback for .feather file search (temporary)．

        This will be replaced by Rust implementation．

        Args:
            base_path: Base directory to search from
            prefix: Prefix to match against
            limit: Maximum results to return

        Returns:
            List of matching .feather file paths
        """
        try:
            # Search for .feather files only
            feather_files = []

            for path in base_path.rglob("*.feather"):
                path_str = str(path)
                if path_str.startswith(prefix):
                    feather_files.append(path_str)

                # Limit for performance
                if len(feather_files) >= limit * 10:
                    break

            # Filter and sort
            matching = [
                f for f in feather_files if f.startswith(prefix)
            ]
            return sorted(matching)[:limit]

        except Exception as e:
            logger.error(f"File search failed: {e}")
            return []
