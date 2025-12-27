# -*- coding: utf-8 -*-
"""
AFC 现代 GUI 前端 (PyQt6)

现代化界面设计：
- 暗色主题
- 圆角窗口和卡片
- 流畅动画效果
- 拖拽上传图片
"""

import sys
import os
from datetime import datetime
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QGraphicsDropShadowEffect,
    QProgressBar, QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox
)
from PyQt6.QtCore import (
    Qt, QSize, QPropertyAnimation, QEasingCurve, QThread, pyqtSignal
)
from PyQt6.QtGui import (
    QFont, QColor, QPainter, QBrush, QPixmap, QDragEnterEvent, QDropEvent, QCursor
)

import requests


# ============== 现代化颜色主题 ==============
class ModernTheme:
    """现代化配色方案"""

    # 浅色主题
    LIGHT = {
        "bg_primary": "#f8f9fa",
        "bg_secondary": "#e9ecef",
        "bg_card": "#ffffff",
        "bg_hover": "#dee2e6",
        "accent": "#4f46e5",
        "accent_hover": "#6366f1",
        "success": "#10b981",
        "warning": "#f59e0b",
        "danger": "#ef4444",
        "text_primary": "#1f2937",
        "text_secondary": "#6b7280",
        "text_muted": "#9ca3af",
        "border": "#e5e7eb",
        "gradient_start": "#f8f9fa",
        "gradient_end": "#e9ecef"
    }

    # 暗色主题
    DARK = {
        "bg_primary": "#1e1e2e",
        "bg_secondary": "#2a2a3c",
        "bg_card": "#353548",
        "bg_hover": "#454560",
        "accent": "#89b4fa",
        "accent_hover": "#b4befe",
        "success": "#a6e3a1",
        "warning": "#f9e2af",
        "danger": "#f38ba8",
        "text_primary": "#cdd6f4",
        "text_secondary": "#a6adc8",
        "text_muted": "#6c7086",
        "border": "#45475a",
        "gradient_start": "#1e1e2e",
        "gradient_end": "#313244"
    }

    @classmethod
    def current(cls):
        return cls.LIGHT


# ============== 自定义圆角卡片组件 ==============
class RoundedCard(QFrame):
    """现代化圆角卡片组件"""

    def __init__(self, parent=None, radius=16, shadow=True):
        super().__init__(parent)
        self.setObjectName("roundedCard")
        self.radius = radius
        self.shadow_enabled = shadow
        self.setup_ui()

    def setup_ui(self):
        self.setContentsMargins(0, 0, 0, 0)
        if self.shadow_enabled:
            shadow = QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(30)
            shadow.setColor(QColor(0, 0, 0, 60))
            shadow.setOffset(0, 8)
            self.setGraphicsEffect(shadow)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        theme = ModernTheme.current()
        rect = self.rect().adjusted(1, 1, -1, -1)

        # 绘制背景
        painter.setPen(Qt.PenStyle.NoPen)
        brush = QBrush(QColor(theme["bg_card"]))
        painter.setBrush(brush)
        painter.drawRoundedRect(rect, self.radius, self.radius)

        super().paintEvent(event)


# ============== 现代化按钮组件 ==============
class ModernButton(QPushButton):
    """现代化按钮组件"""

    def __init__(self, text="", parent=None, color_type="primary", icon=None):
        super().__init__(text, parent)
        self.color_type = color_type
        self.icon_path = icon
        self.setup_ui()

    def setup_ui(self):
        theme = ModernTheme.current()
        colors = {
            "primary": (theme["accent"], theme["accent_hover"]),
            "success": (theme["success"], "#b8c994"),
            "danger": (theme["danger"], "#f5a3b7"),
            "secondary": (theme["text_secondary"], theme["text_primary"])
        }

        self.default_color, self.hover_color = colors.get(
            self.color_type, colors["primary"]
        )

        self.setFixedHeight(48)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        # 字体设置
        font = QFont()
        font.setFamily("Segoe UI, SF Pro Display, -apple-system")
        font.setPixelSize(15)
        font.setWeight(QFont.Weight.Medium)
        self.setFont(font)

        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.default_color};
                color: white;
                border: none;
                border-radius: 12px;
                padding: 0 24px;
            }}
            QPushButton:hover {{
                background-color: {self.hover_color};
            }}
            QPushButton:disabled {{
                background-color: {ModernTheme.current()["text_muted"]};
            }}
        """)


# ============== 图片预览区域组件 ==============
class ImagePreview(QFrame):
    """图片预览区域"""

    imageSelected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_image_path: Optional[str] = None
        self.setup_ui()

    def setup_ui(self):
        theme = ModernTheme.current()
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {theme["bg_secondary"]};
                border: 2px dashed {theme["border"]};
                border-radius: 16px;
            }}
        """)
        self.setMinimumHeight(300)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)

        # 图片预览标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 280)
        layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # 占位文本
        self.placeholder_label = QLabel("No image selected")
        self.placeholder_label.setStyleSheet(f"""
            color: {theme["text_secondary"]};
            font-size: 18px;
        """)
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.placeholder_label)

        # 上传按钮
        self.upload_btn = ModernButton("Select Image", color_type="primary")
        self.upload_btn.setFixedWidth(150)
        self.upload_btn.clicked.connect(self.select_image)
        layout.addWidget(self.upload_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addStretch()

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if file_path:
            self.load_image(file_path)

    def is_image_file(self, path):
        return path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

    def load_image(self, path):
        self.current_image_path = path
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            # 缩放到适合大小
            scaled = pixmap.scaled(
                QSize(400, 280),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            self.image_label.setPixmap(scaled)
            self.placeholder_label.hide()
            self.imageSelected.emit(path)


# ============== 现代化进度条 ==============
class ModernProgressBar(QProgressBar):
    """现代化进度条"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(0, 100)
        self.setValue(0)
        self.setFixedHeight(8)
        self.setStyleSheet(f"""
            QProgressBar {{
                background-color: {ModernTheme.current()["bg_secondary"]};
                border: none;
                border-radius: 4px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 {ModernTheme.current()["accent"]},
                    stop:1 #b4befe
                );
                border-radius: 4px;
            }}
        """)


# ============== 结果显示卡片 ==============
class ResultCard(QWidget):
    """结果显示卡片"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        theme = ModernTheme.current()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)

        # 结果图标
        self.icon_label = QLabel("--")
        self.icon_label.setStyleSheet(f"""
            font-size: 72px;
            color: {theme["text_muted"]};
        """)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.icon_label)

        # 结果文本
        self.result_label = QLabel("Waiting for image...")
        self.result_label.setStyleSheet(f"""
            color: {theme["text_secondary"]};
            font-size: 24px;
            font-weight: bold;
            font-family: 'Segoe UI', SF Pro Display;
        """)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.result_label)

        # 置信度
        self.confidence_label = QLabel("--")
        self.confidence_label.setStyleSheet(f"""
            color: {theme["text_muted"]};
            font-size: 14px;
            font-family: 'Segoe UI', SF Pro Display;
        """)
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.confidence_label)

        # 置信度进度条
        self.confidence_bar = ModernProgressBar()
        self.confidence_bar.setValue(0)
        layout.addWidget(self.confidence_bar)

        # 分割线
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"background-color: {theme['border']};")
        layout.addWidget(line)

        # 概率分布
        prob_layout = QHBoxLayout()

        self.ai_prob_label = self.create_prob_item(
            "AI Generated", theme["danger"]
        )
        prob_layout.addWidget(self.ai_prob_label)

        prob_layout.addSpacing(20)

        self.real_prob_label = self.create_prob_item(
            "Real Photo", theme["success"]
        )
        prob_layout.addWidget(self.real_prob_label)

        layout.addLayout(prob_layout)

    def create_prob_item(self, text, color):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(text)
        label.setStyleSheet(f"""
            color: {ModernTheme.current()["text_secondary"]};
            font-size: 14px;
        """)
        layout.addWidget(label)

        value_label = QLabel("0%")
        value_label.setObjectName("valueLabel")
        value_label.setStyleSheet(f"""
            color: {color};
            font-size: 28px;
            font-weight: bold;
            font-family: 'Segoe UI', SF Pro Display;
        """)
        layout.addWidget(value_label)

        return widget

    def set_result(self, prediction: str, confidence: float,
                   ai_prob: float, real_prob: float):
        theme = ModernTheme.current()

        if prediction == "AI":
            self.icon_label.setText("🤖")
            self.icon_label.setStyleSheet(f"font-size: 72px; color: {theme['danger']};")
            self.result_label.setText("AI Generated")
            self.result_label.setStyleSheet(f"""
                color: {theme['danger']};
                font-size: 24px;
                font-weight: bold;
                font-family: 'Segoe UI', SF Pro Display;
            """)
        else:
            self.icon_label.setText("✅")
            self.icon_label.setStyleSheet(f"font-size: 72px; color: {theme['success']};")
            self.result_label.setText("Real Photo")
            self.result_label.setStyleSheet(f"""
                color: {theme['success']};
                font-size: 24px;
                font-weight: bold;
                font-family: 'Segoe UI', SF Pro Display;
            """)

        self.confidence_label.setText(f"Confidence: {confidence:.1f}%")

        # 动画显示进度条
        self.animate_progress(self.confidence_bar, int(confidence))

        # 更新概率
        ai_label = self.ai_prob_label.findChild(QLabel, "valueLabel")
        real_label = self.real_prob_label.findChild(QLabel, "valueLabel")
        ai_label.setText(f"{ai_prob * 100:.1f}%")
        real_label.setText(f"{real_prob * 100:.1f}%")

    def animate_progress(self, bar: ModernProgressBar, target: int):
        bar.setValue(0)
        animation = QPropertyAnimation(bar, b"value")
        animation.setStartValue(0)
        animation.setEndValue(target)
        animation.setDuration(500)
        animation.setEasingCurve(QEasingCurve.Type.OutQuad)
        animation.start()


# ============== 历史记录表格 ==============
class HistoryTable(QTableWidget):
    """现代化历史记录表格"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        theme = ModernTheme.current()
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(["Time", "File", "Result", "Confidence"])
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)

        # 设置列宽
        self.setColumnWidth(0, 80)
        self.setColumnWidth(1, 150)
        self.setColumnWidth(2, 100)
        self.setColumnWidth(3, 100)

        self.setStyleSheet(f"""
            QTableWidget {{
                background-color: {theme['bg_secondary']};
                border: none;
                border-radius: 12px;
                font-family: 'Segoe UI', SF Pro Display;
                font-size: 14px;
                color: {theme['text_primary']};
                gridline-color: transparent;
            }}
            QTableWidget::item {{
                padding: 12px 16px;
                border-bottom: 1px solid {theme['border']};
            }}
            QTableWidget::item:selected {{
                background-color: {theme['accent']};
                color: white;
            }}
            QHeaderView::section {{
                background-color: {theme['bg_card']};
                color: {theme['text_secondary']};
                font-size: 12px;
                font-weight: bold;
                padding: 12px 16px;
                border: none;
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {theme['bg_hover']};
                border-radius: 4px;
                min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)

    def add_row(self, timestamp: str, filename: str, result: str, confidence: str):
        row = self.rowCount()
        self.insertRow(row)

        theme = ModernTheme.current()
        result_colors = {
            "AI": theme["danger"],
            "Real": theme["success"]
        }

        items = [
            QTableWidgetItem(timestamp),
            QTableWidgetItem(filename),
            QTableWidgetItem(result),
            QTableWidgetItem(confidence)
        ]

        for col, item in enumerate(items):
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if col == 2:  # Result column
                item.setForeground(QColor(result_colors.get(result, theme["text_primary"])))
            else:
                item.setForeground(QColor(theme["text_primary"]))
            self.setItem(row, col, item)


# ============== 预测工作线程 ==============
class PredictWorker(QThread):
    """预测工作线程"""

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, server_url: str, image_path: str):
        super().__init__()
        self.server_url = server_url
        self.image_path = image_path

    def run(self):
        try:
            with open(self.image_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(self.server_url, files=files, timeout=30)

            if response.status_code == 200:
                result = response.json()
                self.finished.emit(result)
            else:
                self.error.emit(f"Server Error: {response.status_code}")
        except requests.exceptions.ConnectionError:
            self.error.emit("Cannot connect to server. Ensure it is running.")
        except Exception as e:
            self.error.emit(str(e))


# ============== 主窗口 ==============
class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Face Detection")
        self.setMinimumSize(1200, 800)
        self.server_url = "http://localhost:5000/predict"
        self.current_image_path: Optional[str] = None
        self.worker: Optional[PredictWorker] = None

        self.setup_ui()
        self.setup_stylesheet()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        # 主布局
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 顶部标题栏
        self.create_title_bar(main_layout)

        # 内容区域
        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(24, 24, 24, 24)
        content_layout.setSpacing(20)

        # 左侧 - 图片上传
        left_panel = RoundedCard()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Image Upload")
        title.setStyleSheet(f"""
            color: {ModernTheme.current()['text_primary']};
            font-size: 20px;
            font-weight: bold;
            font-family: 'Segoe UI', SF Pro Display;
        """)
        left_layout.addWidget(title)

        subtitle = QLabel("Drag & drop or click to select")
        subtitle.setStyleSheet(f"""
            color: {ModernTheme.current()['text_secondary']};
            font-size: 14px;
            margin-bottom: 16px;
        """)
        left_layout.addWidget(subtitle)

        self.image_preview = ImagePreview()
        self.image_preview.imageSelected.connect(self.on_image_selected)
        left_layout.addWidget(self.image_preview)

        # 分析按钮
        self.analyze_btn = ModernButton("Analyze Image", color_type="primary")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.analyze_image)
        left_layout.addWidget(self.analyze_btn)

        # 进度条
        self.progress_bar = ModernProgressBar()
        self.progress_bar.hide()
        left_layout.addWidget(self.progress_bar)

        content_layout.addWidget(left_panel, stretch=1)

        # 右侧 - 结果展示
        right_panel = RoundedCard()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(20, 20, 20, 20)

        # 结果卡片
        self.result_card = ResultCard()
        right_layout.addWidget(self.result_card, stretch=1)

        # 历史记录
        history_title = QLabel("Recent History")
        history_title.setStyleSheet(f"""
            color: {ModernTheme.current()['text_primary']};
            font-size: 18px;
            font-weight: bold;
            margin-top: 16px;
            font-family: 'Segoe UI', SF Pro Display;
        """)
        right_layout.addWidget(history_title)

        self.history_table = HistoryTable()
        self.history_table.setMaximumHeight(200)
        right_layout.addWidget(self.history_table)

        # 清空按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        clear_btn = ModernButton("Clear", color_type="secondary")
        clear_btn.setFixedWidth(100)
        clear_btn.setFixedHeight(36)
        clear_btn.clicked.connect(self.clear_history)
        btn_layout.addWidget(clear_btn)

        right_layout.addLayout(btn_layout)

        content_layout.addWidget(right_panel, stretch=1)

        main_layout.addWidget(content)

        # 状态栏
        self.status_bar = QLabel("Ready")
        self.status_bar.setStyleSheet(f"""
            color: {ModernTheme.current()['text_muted']};
            font-size: 13px;
            padding: 8px 24px;
            background-color: {ModernTheme.current()['bg_secondary']};
        """)
        main_layout.addWidget(self.status_bar)

    def create_title_bar(self, parent_layout):
        """创建标题栏"""
        theme = ModernTheme.current()
        title_bar = QLabel("AI Face Detection")
        title_bar.setStyleSheet(f"""
            color: {theme['text_primary']};
            font-size: 28px;
            font-weight: bold;
            font-family: 'Segoe UI', SF Pro Display;
            padding: 24px 32px 16px;
        """)
        parent_layout.addWidget(title_bar)

        subtitle = QLabel("AI-powered image analysis for detecting generated content")
        subtitle.setStyleSheet(f"""
            color: {theme['text_secondary']};
            font-size: 14px;
            padding: 0 32px 8px;
            font-family: 'Segoe UI', SF Pro Display;
        """)
        parent_layout.addWidget(subtitle)

    def setup_stylesheet(self):
        theme = ModernTheme.current()
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 {theme['gradient_start']},
                    stop:1 {theme['gradient_end']}
                );
            }}
            QWidget {{
                font-family: 'Segoe UI', SF Pro Display, -apple-system;
            }}
        """)

    def on_image_selected(self, path: str):
        """图片选择回调"""
        self.current_image_path = path
        filename = os.path.basename(path)
        self.update_status(f"Selected: {filename}")
        self.analyze_btn.setEnabled(True)
        self.result_card.set_result("Waiting", 0, 0, 0)

    def analyze_image(self):
        """开始分析图片"""
        if not self.current_image_path:
            return

        self.analyze_btn.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.update_status("Analyzing...")

        # 启动工作线程
        self.worker = PredictWorker(self.server_url, self.current_image_path)
        self.worker.finished.connect(self.on_analysis_complete)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.start()

    def on_analysis_complete(self, result: dict):
        """分析完成回调"""
        self.progress_bar.hide()
        self.analyze_btn.setEnabled(True)

        prediction = result.get('prediction', 'Unknown')
        confidence = result.get('confidence', 0)
        probabilities = result.get('probabilities', {})

        ai_prob = probabilities.get('AI', 0)
        real_prob = probabilities.get('Real', 0)

        # 更新结果显示
        self.result_card.set_result(prediction, confidence, ai_prob, real_prob)

        # 添加到历史记录
        timestamp = datetime.now().strftime("%H:%M:%S")
        filename = os.path.basename(self.current_image_path)
        short_name = filename[:15] + ".." if len(filename) > 17 else filename
        result_text = "AI" if prediction == "AI" else "Real"

        self.history_table.add_row(
            timestamp, short_name, result_text, f"{confidence:.1f}%"
        )

        self.update_status("Analysis complete")

    def on_analysis_error(self, error: str):
        """分析错误回调"""
        self.progress_bar.hide()
        self.analyze_btn.setEnabled(True)
        self.update_status(f"Error: {error}")
        QMessageBox.critical(self, "Error", error)

    def clear_history(self):
        """清空历史记录"""
        self.history_table.setRowCount(0)
        self.update_status("History cleared")

    def update_status(self, text: str):
        """更新状态栏"""
        self.status_bar.setText(text)


# ============== 入口点 ==============
def main():
    app = QApplication(sys.argv)

    # 设置应用属性
    app.setApplicationName("AI Face Detection")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
