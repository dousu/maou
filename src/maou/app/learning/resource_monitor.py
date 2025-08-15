import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Union

import psutil

try:
    import pynvml

    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False


@dataclass(frozen=True)
class ResourceUsage:
    """リソース使用率の統計情報を格納するデータクラス．"""

    cpu_max_percent: float
    memory_max_bytes: int
    memory_max_percent: float
    gpu_max_percent: Optional[float] = None
    gpu_memory_max_bytes: Optional[int] = None
    gpu_memory_total_bytes: Optional[int] = None
    gpu_memory_max_percent: Optional[float] = None

    def to_dict(self) -> Dict[str, Union[float, int, None]]:
        """リソース使用率を辞書形式で返す．"""
        return {
            "cpu_max_percent": self.cpu_max_percent,
            "memory_max_bytes": self.memory_max_bytes,
            "memory_max_percent": self.memory_max_percent,
            "gpu_max_percent": self.gpu_max_percent,
            "gpu_memory_max_bytes": self.gpu_memory_max_bytes,
            "gpu_memory_total_bytes": self.gpu_memory_total_bytes,
            "gpu_memory_max_percent": self.gpu_memory_max_percent,
        }


class SystemResourceMonitor:
    """
    CPUとメモリの使用率を監視するクラス．

    バックグラウンドスレッドで定期的に使用率を測定し，
    最大値を記録する．
    """

    def __init__(
        self, monitoring_interval: float = 0.5
    ) -> None:
        """
        Args:
            monitoring_interval: 監視間隔（秒）
        """
        self.monitoring_interval = monitoring_interval
        self.logger = logging.getLogger(__name__)

        # 監視状態
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None

        # 統計情報
        self._cpu_max_percent = 0.0
        self._memory_max_bytes = 0
        self._memory_max_percent = 0.0
        self._total_memory_bytes = psutil.virtual_memory().total

    def start_monitoring(self) -> None:
        """リソース監視を開始する．"""
        if self._monitoring:
            self.logger.warning(
                "Resource monitoring is already running"
            )
            return

        self.logger.debug("Starting system resource monitoring")
        self._monitoring = True
        self._reset_statistics()

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
        )
        self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        """リソース監視を停止する．"""
        if not self._monitoring:
            return

        self.logger.debug("Stopping system resource monitoring")
        self._monitoring = False

        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None

    def get_resource_usage(self) -> ResourceUsage:
        """現在の統計情報を取得する．"""
        return ResourceUsage(
            cpu_max_percent=self._cpu_max_percent,
            memory_max_bytes=self._memory_max_bytes,
            memory_max_percent=self._memory_max_percent,
        )

    def _monitor_loop(self) -> None:
        """監視ループ（バックグラウンドスレッドで実行）．"""
        while self._monitoring:
            try:
                # CPU使用率を取得（短い間隔で測定）
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self._cpu_max_percent = max(
                    self._cpu_max_percent, cpu_percent
                )

                # メモリ使用率を取得
                memory = psutil.virtual_memory()
                self._memory_max_bytes = max(
                    self._memory_max_bytes, memory.used
                )
                self._memory_max_percent = max(
                    self._memory_max_percent, memory.percent
                )

                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(
                    f"Error during system resource monitoring: {e}"
                )

    def _reset_statistics(self) -> None:
        """統計情報をリセットする．"""
        self._cpu_max_percent = 0.0
        self._memory_max_bytes = 0
        self._memory_max_percent = 0.0


class GPUResourceMonitor:
    """
    NVIDIA GPU使用率とGPUメモリを監視するクラス．

    pynvmlライブラリを使用してNVIDIA GPUの統計を取得する．
    """

    def __init__(
        self,
        gpu_index: int = 0,
        monitoring_interval: float = 0.5,
    ) -> None:
        """
        Args:
            gpu_index: 監視するGPUのインデックス
            monitoring_interval: 監視間隔（秒）
        """
        self.gpu_index = gpu_index
        self.monitoring_interval = monitoring_interval
        self.logger = logging.getLogger(__name__)

        # pynvmlが利用可能かチェック
        if not HAS_PYNVML:
            self.logger.warning(
                "pynvml is not available. GPU monitoring disabled."
            )
            self._gpu_available = False
            return

        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(
                self.gpu_index
            )
            self._gpu_available = True

            # GPU情報を取得
            gpu_info = pynvml.nvmlDeviceGetMemoryInfo(
                self._handle
            )
            self._total_gpu_memory = gpu_info.total

            self.logger.debug(
                f"GPU monitoring initialized for device {gpu_index}"
            )
        except Exception as e:
            self.logger.warning(
                f"Failed to initialize GPU monitoring: {e}"
            )
            self._gpu_available = False

        # 監視状態
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None

        # 統計情報
        self._gpu_max_percent = 0.0
        self._gpu_memory_max_bytes = 0
        self._gpu_memory_max_percent = 0.0
        
        # デフォルト値を設定（GPU利用できない場合に備えて）
        if not self._gpu_available:
            self._total_gpu_memory = 0

    def start_monitoring(self) -> None:
        """GPU監視を開始する．"""
        if not self._gpu_available:
            return

        if self._monitoring:
            self.logger.warning(
                "GPU monitoring is already running"
            )
            return

        self.logger.debug("Starting GPU resource monitoring")
        self._monitoring = True
        self._reset_statistics()

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
        )
        self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        """GPU監視を停止する．"""
        if not self._gpu_available or not self._monitoring:
            return

        self.logger.debug("Stopping GPU resource monitoring")
        self._monitoring = False

        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None

    def get_resource_usage(self) -> ResourceUsage:
        """現在のGPU統計情報を含むResourceUsageを取得する．"""
        if not self._gpu_available:
            return ResourceUsage(
                cpu_max_percent=0.0,
                memory_max_bytes=0,
                memory_max_percent=0.0,
            )

        gpu_memory_max_percent = None
        if self._total_gpu_memory > 0:
            gpu_memory_max_percent = (
                float(self._gpu_memory_max_bytes)
                / float(self._total_gpu_memory)
                * 100.0
            )

        return ResourceUsage(
            cpu_max_percent=0.0,  # SystemResourceMonitorで設定
            memory_max_bytes=0,  # SystemResourceMonitorで設定
            memory_max_percent=0.0,  # SystemResourceMonitorで設定
            gpu_max_percent=self._gpu_max_percent,
            gpu_memory_max_bytes=int(
                self._gpu_memory_max_bytes
            ),
            gpu_memory_total_bytes=int(self._total_gpu_memory),
            gpu_memory_max_percent=gpu_memory_max_percent,
        )

    def _monitor_loop(self) -> None:
        """GPU監視ループ（バックグラウンドスレッドで実行）．"""
        while self._monitoring:
            try:
                # GPU使用率を取得
                utilization = (
                    pynvml.nvmlDeviceGetUtilizationRates(
                        self._handle
                    )
                )
                self._gpu_max_percent = max(
                    self._gpu_max_percent,
                    float(utilization.gpu),
                )

                # GPUメモリ使用量を取得
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(
                    self._handle
                )
                self._gpu_memory_max_bytes = max(
                    self._gpu_memory_max_bytes, memory_info.used
                )

                gpu_memory_percent = (
                    float(memory_info.used)
                    / float(memory_info.total)
                    * 100.0
                )
                self._gpu_memory_max_percent = max(
                    self._gpu_memory_max_percent,
                    gpu_memory_percent,
                )

                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(
                    f"Error during GPU resource monitoring: {e}"
                )

    def _reset_statistics(self) -> None:
        """GPU統計情報をリセットする．"""
        self._gpu_max_percent = 0.0
        self._gpu_memory_max_bytes = 0
        self._gpu_memory_max_percent = 0.0

    @property
    def gpu_available(self) -> bool:
        """GPUが利用可能かどうか．"""
        return self._gpu_available
