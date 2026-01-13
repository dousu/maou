"""インデックス作成状態管理（インフラ層）．

バックグラウンドスレッドでのインデックス作成の状態を，
スレッドセーフに管理するクラスを提供する．
"""

import threading
import time
from typing import Any, Dict, Literal, Optional


class IndexingState:
    """インデックス作成状態を管理するスレッドセーフなクラス．

    インデックス作成の進捗，状態遷移，時間推定，
    キャンセル機能をスレッドセーフに管理する．

    状態遷移:
        idle → indexing → ready|failed
        ready|failed → indexing (新しいデータソースをロード時)
    """

    def __init__(self) -> None:
        """インスタンスを初期化．"""
        self._status: Literal[
            "idle", "indexing", "ready", "failed"
        ] = "idle"
        self._progress_message: str = ""
        self._error_message: Optional[str] = None
        self._files_processed: int = 0
        self._total_files: int = 0
        self._records_indexed: int = 0
        self._start_time: Optional[float] = None  # Unix timestamp
        self._cancelled: bool = False
        self._lock = threading.Lock()

    def set_indexing(
        self, total_files: int, initial_message: str = "開始中..."
    ) -> None:
        """インデックス作成を開始．

        Args:
            total_files: 処理する全ファイル数
            initial_message: 初期メッセージ
        """
        with self._lock:
            self._status = "indexing"
            self._total_files = total_files
            self._files_processed = 0
            self._records_indexed = 0
            self._progress_message = initial_message
            self._error_message = None
            self._cancelled = False
            self._start_time = time.time()

    def update_progress(
        self, files_done: int, records: int, message: str
    ) -> None:
        """進捗を更新．

        Args:
            files_done: 処理完了したファイル数
            records: インデックス化されたレコード数
            message: 進捗メッセージ
        """
        with self._lock:
            if self._status == "indexing" and not self._cancelled:
                self._files_processed = files_done
                self._records_indexed = records
                self._progress_message = message

    def set_ready(self, total_records: int) -> None:
        """インデックス作成完了．

        Args:
            total_records: インデックス化された総レコード数
        """
        with self._lock:
            if not self._cancelled:
                self._status = "ready"
                self._records_indexed = total_records
                self._progress_message = "インデックス作成完了"

    def set_failed(self, error_message: str) -> None:
        """インデックス作成失敗．

        Args:
            error_message: エラーメッセージ
        """
        with self._lock:
            self._status = "failed"
            self._error_message = error_message
            self._progress_message = "インデックス作成失敗"

    def cancel(self) -> None:
        """インデックス作成をキャンセル．

        進行中のインデックス作成をキャンセルする．
        バックグラウンドスレッドはこのフラグをチェックし，
        適切に終了する必要がある．
        """
        with self._lock:
            self._cancelled = True

    def is_cancelled(self) -> bool:
        """キャンセルフラグを確認．

        Returns:
            キャンセルされている場合はTrue
        """
        with self._lock:
            return self._cancelled

    def is_indexing(self) -> bool:
        """インデックス作成中かを確認．

        Returns:
            インデックス作成中の場合はTrue
        """
        with self._lock:
            return self._status == "indexing"

    def get_status(
        self,
    ) -> Literal["idle", "indexing", "ready", "failed"]:
        """現在の状態を取得．

        Returns:
            現在の状態
        """
        with self._lock:
            return self._status

    def get_progress(self) -> Dict[str, Any]:
        """進捗情報を取得．

        Returns:
            進捗情報を含む辞書
        """
        with self._lock:
            return {
                "files": self._files_processed,
                "total_files": self._total_files,
                "records": self._records_indexed,
                "message": self._progress_message,
            }

    def get_error(self) -> Optional[str]:
        """エラーメッセージを取得．

        Returns:
            エラーメッセージ，エラーがない場合はNone
        """
        with self._lock:
            return self._error_message

    def estimate_remaining_time(self) -> Optional[int]:
        """推定残り時間を計算．

        現在の進捗に基づいて，残りの処理時間を秒単位で推定する．

        Returns:
            推定残り時間（秒），推定できない場合はNone

        Note:
            - 進捗が1%未満の場合は推定値を返さない
            - 処理速度が一定であると仮定して線形補間で計算
        """
        with self._lock:
            # 開始時刻またはtotal_filesが設定されていない場合
            if self._start_time is None or self._total_files == 0:
                return None

            # まだファイル処理が始まっていない場合
            if self._files_processed == 0:
                return None

            # 経過時間を計算
            elapsed = time.time() - self._start_time

            # 進捗率を計算
            progress_ratio = self._files_processed / self._total_files

            # 進捗が1%未満の場合は推定が不正確なためNoneを返す
            if progress_ratio < 0.01:
                return None

            # 全体の予測時間を計算
            estimated_total = elapsed / progress_ratio

            # 残り時間を計算
            remaining = estimated_total - elapsed

            # 負の値にならないようにする
            return max(0, int(remaining))
