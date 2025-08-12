"""
图像控制组件 - 统一的按钮+勾选框布局
"""

from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QPushButton, QCheckBox, 
                             QButtonGroup, QVBoxLayout, QLabel)
from PyQt6.QtCore import Qt, pyqtSignal
from ui_constants import StyleSheets, Icons


class ImageControlRow(QWidget):
    """单行图像控制组件：按钮 + 勾选框"""
    
    compute_requested = pyqtSignal(str)  # image_type
    display_toggled = pyqtSignal(str, bool)  # image_type, checked
    
    def __init__(self, image_type: str, display_name: str, parent=None):
        super().__init__(parent)
        self.image_type = image_type
        self.display_name = display_name
        self.setup_ui()
    
    def setup_ui(self):
        """设置UI布局"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(10)
        
        # 计算按钮
        self.compute_button = QPushButton(f"🔄 计算{self.display_name}")
        self.compute_button.setMinimumWidth(120)
        self.compute_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #E8F4F8, stop:1 #D1E7DD);
                border: 1px solid #75B798;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 12px;
                color: #2D5016;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #D1E7DD, stop:1 #A3CFBB);
            }
            QPushButton:pressed {
                background: #A3CFBB;
            }
        """)
        self.compute_button.clicked.connect(lambda: self.compute_requested.emit(self.image_type))
        
        # 显示勾选框
        self.display_checkbox = QCheckBox("显示")
        self.display_checkbox.setStyleSheet("""
            QCheckBox {
                font-size: 12px;
                color: #2C3E50;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 1px solid #95A5A6;
            }
            QCheckBox::indicator:unchecked {
                background: white;
            }
            QCheckBox::indicator:checked {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3498DB, stop:1 #2980B9);
                border: 1px solid #2980B9;
            }
            QCheckBox::indicator:checked {
                color: white;
                font-size: 12px;
                text-align: center;
            }
        """)
        self.display_checkbox.toggled.connect(
            lambda checked: self.display_toggled.emit(self.image_type, checked)
        )
        
        # 添加到布局
        layout.addWidget(self.compute_button)
        layout.addWidget(self.display_checkbox)
        layout.addStretch()  # 推向左侧
    
    def set_computed(self, computed: bool):
        """设置计算状态"""
        if computed:
            self.compute_button.setText(f"✅ {self.display_name}")
            self.display_checkbox.setEnabled(True)
        else:
            self.compute_button.setText(f"🔄 计算{self.display_name}")
            self.display_checkbox.setEnabled(False)
            self.display_checkbox.setChecked(False)
    
    def set_display_checked(self, checked: bool):
        """设置显示状态（不触发信号）"""
        self.display_checkbox.blockSignals(True)
        self.display_checkbox.setChecked(checked)
        self.display_checkbox.blockSignals(False)


class ImageControlPanel(QWidget):
    """图像控制面板 - 管理多个图像控制行"""
    
    compute_requested = pyqtSignal(str)  # image_type
    display_changed = pyqtSignal(str)  # image_type (或None表示清除)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_rows = {}
        self.display_group = QButtonGroup(self)  # 管理互斥选择
        self.display_group.setExclusive(True)
        self.current_display = None
        self.setup_ui()
        self.connect_signals()
    
    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # 图像类型配置
        image_configs = [
            ('sdf', 'SDF场'),
            ('temperature', '温度场'),
            ('voronoi', '泰森多边形')
        ]
        
        # 创建图像控制行
        for image_type, display_name in image_configs:
            row = ImageControlRow(image_type, display_name, self)
            self.image_rows[image_type] = row
            
            # 添加到互斥组
            self.display_group.addButton(row.display_checkbox)
            
            layout.addWidget(row)
        
        layout.addStretch()  # 底部弹性空间
    
    def connect_signals(self):
        """连接信号"""
        for image_type, row in self.image_rows.items():
            # 计算请求
            row.compute_requested.connect(self.compute_requested.emit)
            
            # 显示切换
            row.display_toggled.connect(self._handle_display_toggle)
    
    def _handle_display_toggle(self, image_type: str, checked: bool):
        """处理显示切换"""
        if checked:
            # 选中新的显示
            self.current_display = image_type
            self.display_changed.emit(image_type)
            print(f"[ImageControlPanel] 切换显示到: {image_type}")
        else:
            # 取消选中（可能是手动取消或互斥导致）
            if self.current_display == image_type:
                self.current_display = None
                self.display_changed.emit("")  # 空字符串表示清除显示
                print(f"[ImageControlPanel] 清除显示")
    
    def set_image_computed(self, image_type: str, computed: bool):
        """设置图像计算状态"""
        if image_type in self.image_rows:
            self.image_rows[image_type].set_computed(computed)
    
    def clear_all_display(self):
        """清除所有显示选择"""
        # 临时断开信号避免循环
        for row in self.image_rows.values():
            row.display_checkbox.blockSignals(True)
            row.display_checkbox.setChecked(False)
            row.display_checkbox.blockSignals(False)
        
        self.current_display = None
    
    def get_current_display(self) -> str:
        """获取当前显示的图像类型"""
        return self.current_display
