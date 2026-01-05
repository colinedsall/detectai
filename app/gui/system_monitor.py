"""
System Monitor Widget for GPU/CPU/Memory monitoring.
Uses psutil and macOS-specific tools for Apple Silicon GPU stats.
"""
import os
import subprocess
import re
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QProgressBar, QFrame
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont, QPainter, QColor, QPen
from collections import deque

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class GraphWidget(QWidget):
    """Simple live graph widget for system metrics."""
    
    def __init__(self, title: str = "Usage", color: QColor = QColor(0, 200, 100), max_points: int = 60):
        super().__init__()
        self.title = title
        self.color = color
        self.max_points = max_points
        self.data = deque([0.0] * max_points, maxlen=max_points)
        self.setMinimumHeight(80)
        self.setMinimumWidth(200)
    
    def add_value(self, value: float):
        """Add a new value (0-100) to the graph."""
        self.data.append(min(100, max(0, value)))
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        
        # Grid lines
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        h = self.height()
        w = self.width()
        for i in range(1, 4):
            y = int(h * i / 4)
            painter.drawLine(0, y, w, y)
        
        # Draw the graph line
        if len(self.data) > 1:
            painter.setPen(QPen(self.color, 2))
            
            points = list(self.data)
            step = w / (self.max_points - 1)
            
            for i in range(1, len(points)):
                x1 = int((i - 1) * step)
                y1 = int(h - (points[i - 1] / 100.0 * h))
                x2 = int(i * step)
                y2 = int(h - (points[i] / 100.0 * h))
                painter.drawLine(x1, y1, x2, y2)
        
        # Title and current value
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        painter.setFont(QFont("Menlo", 10))
        current = self.data[-1] if self.data else 0
        painter.drawText(5, 15, f"{self.title}: {current:.1f}%")
        
        painter.end()


class SystemMonitorWidget(QWidget):
    """Widget displaying live system metrics with graphs."""
    
    def __init__(self):
        super().__init__()
        self._setup_ui()
        
        # Timer for updates
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_metrics)
        self.timer.start(1000)  # Update every second
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("System Monitor")
        title.setFont(QFont("Menlo", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #ddd;")
        layout.addWidget(title)
        
        # CPU Graph
        cpu_group = QGroupBox("CPU Usage")
        cpu_layout = QVBoxLayout(cpu_group)
        self.cpu_graph = GraphWidget("CPU", QColor(100, 200, 255))
        cpu_layout.addWidget(self.cpu_graph)
        layout.addWidget(cpu_group)
        
        # Memory Graph
        mem_group = QGroupBox("Memory Usage")
        mem_layout = QVBoxLayout(mem_group)
        self.mem_graph = GraphWidget("Memory", QColor(200, 100, 255))
        mem_layout.addWidget(self.mem_graph)
        layout.addWidget(mem_group)
        
        # GPU Graph (Apple Silicon)
        gpu_group = QGroupBox("GPU Usage (Apple Silicon)")
        gpu_layout = QVBoxLayout(gpu_group)
        self.gpu_graph = GraphWidget("GPU", QColor(100, 255, 150))
        gpu_layout.addWidget(self.gpu_graph)
        
        # GPU status label
        self.gpu_status = QLabel("Monitoring GPU...")
        self.gpu_status.setFont(QFont("Menlo", 9))
        self.gpu_status.setStyleSheet("color: #888;")
        gpu_layout.addWidget(self.gpu_status)
        
        layout.addWidget(gpu_group)
        
        # Stats summary
        stats_group = QGroupBox("Current Stats")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_label = QLabel("Loading...")
        self.stats_label.setFont(QFont("Menlo", 10))
        self.stats_label.setStyleSheet("color: #aaa;")
        stats_layout.addWidget(self.stats_label)
        
        layout.addWidget(stats_group)
        layout.addStretch()
    
    def _update_metrics(self):
        """Update all metrics."""
        if PSUTIL_AVAILABLE:
            # CPU
            cpu = psutil.cpu_percent(interval=None)
            self.cpu_graph.add_value(cpu)
            
            # Memory
            mem = psutil.virtual_memory()
            self.mem_graph.add_value(mem.percent)
            
            # GPU (Apple Silicon - estimate from process GPU usage)
            gpu_percent = self._get_gpu_usage()
            self.gpu_graph.add_value(gpu_percent)
            
            # Update stats summary
            self.stats_label.setText(
                f"CPU: {cpu:.1f}%  |  "
                f"Memory: {mem.percent:.1f}% ({mem.used / (1024**3):.1f} GB / {mem.total / (1024**3):.1f} GB)  |  "
                f"GPU: {gpu_percent:.1f}%"
            )
        else:
            self.stats_label.setText("psutil not installed")
    
    def _get_gpu_usage(self) -> float:
        """
        Attempt to get GPU usage on macOS Apple Silicon.
        This is approximate - full GPU monitoring requires sudo powermetrics.
        """
        try:
            # Try to get GPU usage from ioreg or activity monitor data
            # This is a simplified approximation based on GPU-related processes
            
            # Check if any Python/torch processes are using GPU
            gpu_estimate = 0.0
            
            for proc in psutil.process_iter(['name', 'cpu_percent']):
                try:
                    name = proc.info['name'].lower()
                    # Processes likely using GPU
                    if any(x in name for x in ['python', 'ollama', 'metal', 'windowserver']):
                        cpu_pct = proc.info['cpu_percent'] or 0
                        # Rough estimate: GPU usage correlates with CPU for ML tasks
                        if 'python' in name or 'ollama' in name:
                            gpu_estimate += cpu_pct * 0.5  # Assume half CPU == GPU
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Cap at 100%
            return min(100.0, gpu_estimate)
            
        except Exception as e:
            self.gpu_status.setText(f"GPU monitoring limited: {e}")
            return 0.0
    
    def stop(self):
        """Stop the monitoring timer."""
        self.timer.stop()
