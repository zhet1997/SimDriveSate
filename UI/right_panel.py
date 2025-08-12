"""
右侧面板组件
包含可视化控制和实时命令行输出
"""

from typing import TYPE_CHECKING
from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QCheckBox, QFrame, QTextEdit, QSplitter)
from PyQt6.QtCore import Qt, pyqtSignal
from ui_constants import StyleSheets, Icons

if TYPE_CHECKING:
    from main_window import MainWindow


class RightPanel:
    """右侧面板管理器"""
    
    def __init__(self, main_window: 'MainWindow'):
        self.main_window = main_window
        self.setup_panel()
    
    def setup_panel(self):
        """创建右侧面板"""
        # 创建dock widget
        self.right_dock = QDockWidget("可视化控制 & 输出", self.main_window)
        self.right_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | 
                                       Qt.DockWidgetArea.RightDockWidgetArea)
        
        # 创建主容器
        main_widget = QWidget()
        
        # 创建垂直分割器（上下分割）
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # 创建上半部分：可视化控制
        self.visualization_widget = self._create_visualization_control_section()
        splitter.addWidget(self.visualization_widget)
        
        # 创建下半部分：命令行输出
        self.console_widget = self._create_console_output_section()
        splitter.addWidget(self.console_widget)
        
        # 设置分割器比例（可视化控制:命令行输出 = 2:3）
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        
        # 设置主布局
        main_layout = QVBoxLayout(main_widget)
        main_layout.addWidget(splitter)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # 设置右侧面板内容
        self.right_dock.setWidget(main_widget)
        
        # 🔧 设置右侧面板初始宽度
        self.right_dock.setMinimumWidth(250)
        self.right_dock.setMaximumWidth(350)
        main_widget.setFixedWidth(280)  # 设置合适的固定宽度
        
        self.main_window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.right_dock)
    
    def _create_visualization_control_section(self):
        """创建可视化控制区域"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 🔧 使用新的统一图像控制面板
        from image_control_widget import ImageControlPanel
        self.image_control_panel = ImageControlPanel(widget)  # 传递widget作为父级
        
        # 连接信号
        self.image_control_panel.compute_requested.connect(self._handle_compute_request)
        self.image_control_panel.display_changed.connect(self._handle_display_change)
        
        layout.addWidget(self.image_control_panel)
        
        # 🆕 添加温度场重构特殊控制组
        self._create_temperature_reconstruction_section(layout)
        
        layout.addStretch()
        
        return widget
    
    def _create_sdf_control_group(self, layout: QVBoxLayout):
        """创建SDF控制组"""
        # SDF组标题
        sdf_frame = QFrame()
        sdf_frame.setFrameStyle(QFrame.Shape.Box)
        sdf_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #BDC3C7;
                border-radius: 5px;
                background-color: #F8F9FA;
                margin: 2px;
            }
        """)
        
        sdf_layout = QVBoxLayout(sdf_frame)
        sdf_layout.setContentsMargins(10, 8, 10, 8)
        sdf_layout.setSpacing(6)
        
        # SDF标题
        sdf_title = QLabel("📊 SDF场显示")
        sdf_title.setStyleSheet("font-weight: bold; color: #34495E;")
        sdf_layout.addWidget(sdf_title)
        
        # SDF显示开关
        self.sdf_show_checkbox = QCheckBox("显示SDF背景")
        self.sdf_show_checkbox.setStyleSheet(StyleSheets.SDF_CHECKBOX)
        self.sdf_show_checkbox.toggled.connect(self.main_window.on_sdf_show_toggled)
        sdf_layout.addWidget(self.sdf_show_checkbox)
        
        # SDF更新按钮（初始隐藏）
        self.sdf_update_button = QPushButton(f"{Icons.UPDATE_SDF} 更新SDF")
        self.sdf_update_button.setVisible(False)
        self.sdf_update_button.clicked.connect(self.main_window.update_sdf_background)
        self.sdf_update_button.setStyleSheet(StyleSheets.SDF_UPDATE_BUTTON)
        sdf_layout.addWidget(self.sdf_update_button)
        
        layout.addWidget(sdf_frame)
    
    def _create_temperature_control_group(self, layout: QVBoxLayout):
        """创建温度场控制组"""
        # 温度场组框架
        temp_frame = QFrame()
        temp_frame.setFrameStyle(QFrame.Shape.Box)
        temp_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #BDC3C7;
                border-radius: 5px;
                background-color: #F8F9FA;
                margin: 2px;
            }
        """)
        
        temp_layout = QVBoxLayout(temp_frame)
        temp_layout.setContentsMargins(10, 8, 10, 8)
        temp_layout.setSpacing(6)
        
        # 温度场标题
        temp_title = QLabel("🌡️ 温度场重构")
        temp_title.setStyleSheet("font-weight: bold; color: #34495E;")
        temp_layout.addWidget(temp_title)
        
        # 温度场显示开关
        self.temperature_show_checkbox = QCheckBox("显示温度场背景")
        self.temperature_show_checkbox.setStyleSheet(StyleSheets.SDF_CHECKBOX)
        # 连接到主窗口的温度场切换方法（需要后续实现）
        self.temperature_show_checkbox.toggled.connect(self._on_temperature_show_toggled)
        temp_layout.addWidget(self.temperature_show_checkbox)
        
        # 温度场重构按钮
        self.temperature_reconstruct_button = QPushButton("🔄 温度场重构")
        self.temperature_reconstruct_button.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
        """)
        # 连接到主窗口的温度重构方法（需要后续实现）
        self.temperature_reconstruct_button.clicked.connect(self._on_temperature_reconstruct)
        temp_layout.addWidget(self.temperature_reconstruct_button)
        
        layout.addWidget(temp_frame)
    
    def _create_thermal_control_group(self, layout: QVBoxLayout):
        """创建热仿真控制组"""
        # 热仿真组框架
        thermal_frame = QFrame()
        thermal_frame.setFrameStyle(QFrame.Shape.Box)
        thermal_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #BDC3C7;
                border-radius: 5px;
                background-color: #F8F9FA;
                margin: 2px;
            }
        """)
        
        thermal_layout = QVBoxLayout(thermal_frame)
        thermal_layout.setContentsMargins(10, 8, 10, 8)
        thermal_layout.setSpacing(6)
        
        # 热仿真标题
        thermal_title = QLabel("🔥 热仿真结果")
        thermal_title.setStyleSheet("font-weight: bold; color: #34495E;")
        thermal_layout.addWidget(thermal_title)
        
        # 热仿真结果显示开关
        self.thermal_show_checkbox = QCheckBox("显示热仿真结果")
        self.thermal_show_checkbox.setStyleSheet(StyleSheets.SDF_CHECKBOX)
        # 连接到主窗口的热仿真结果切换方法（需要后续实现）
        self.thermal_show_checkbox.toggled.connect(self._on_thermal_show_toggled)
        thermal_layout.addWidget(self.thermal_show_checkbox)
        
        # 运行热仿真按钮
        self.thermal_run_button = QPushButton("🚀 运行热仿真")
        self.thermal_run_button.setStyleSheet("""
            QPushButton {
                background-color: #E67E22;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #D68910;
            }
        """)
        # 连接到主窗口的热仿真方法
        self.thermal_run_button.clicked.connect(self.main_window.run_thermal_simulation)
        thermal_layout.addWidget(self.thermal_run_button)
        
        layout.addWidget(thermal_frame)
    
    def _create_console_output_section(self):
        """创建命令行输出区域"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        # 🔧 移除标题，节省空间
        
        # 创建文本输出区域
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setStyleSheet("""
            QTextEdit {
                background-color: #2C3E50;
                color: #ECF0F1;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 10px;
                border: 1px solid #34495E;
                border-radius: 3px;
                padding: 5px;
            }
        """)
        layout.addWidget(self.console_output)
        
        # 添加清除按钮
        clear_button = QPushButton("🗑️ 清除输出")
        clear_button.setStyleSheet("""
            QPushButton {
                background-color: #95A5A6;
                color: white;
                border: none;
                padding: 4px;
                border-radius: 3px;
                max-height: 25px;
            }
            QPushButton:hover {
                background-color: #7F8C8D;
            }
        """)
        clear_button.clicked.connect(self._clear_console)
        layout.addWidget(clear_button)
        
        return widget
    
    def _clear_console(self):
        """清除控制台输出"""
        self.console_output.clear()
        print("[控制台] 输出已清除")
    
    def append_output(self, text: str):
        """添加输出到控制台"""
        self.console_output.append(text)
        # 自动滚动到底部
        scrollbar = self.console_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    # 🆕 新的统一图像控制处理方法
    def _handle_compute_request(self, image_type: str):
        """处理图像计算请求"""
        print(f"[RightPanel] 计算请求: {image_type}")
        
        # 获取图像管理器
        from image_manager import get_image_manager
        image_manager = get_image_manager()
        
        # 获取组件数据
        if hasattr(self.main_window, 'data_sync'):
            component_data = self.main_window.data_sync.get_components_for_calculation()
        else:
            print("[RightPanel] 错误: 无法获取组件数据")
            return
        
        # 准备输入数据
        input_data = {
            'components': component_data,
            'layout_size': (0.1, 0.1)  # 默认布局尺寸
        }
        
        # 执行计算
        success = image_manager.compute_image(image_type, input_data)
        
        # 更新控件状态
        self.image_control_panel.set_image_computed(image_type, success)
        
        if success:
            print(f"[RightPanel] {image_type} 计算成功")
        else:
            print(f"[RightPanel] {image_type} 计算失败")
    
    def _handle_display_change(self, image_type: str):
        """处理图像显示切换"""
        print(f"[RightPanel] 显示切换: {image_type}")
        
        # 获取图像管理器
        from image_manager import get_image_manager
        image_manager = get_image_manager()
        
        if image_type == "":
            # 清除显示
            image_manager.clear_display()
        else:
            # 🔧 原始温度场使用专业显示路径，POD温度场使用ImageManager显示
            if image_type == "temperature":
                print(f"[RightPanel] 温度场已通过专业显示路径显示，跳过ImageManager显示")
                return
            elif image_type == "pod_temperature":
                print(f"[RightPanel] P温度场预测ImageManager显示")
                # POD温度场继续使用ImageManager显示
            
            # 显示指定图像
            success = image_manager.display_image(image_type)
            if not success:
                print(f"[RightPanel] {image_type} 显示失败")
                # 如果显示失败，清除勾选状态
                self.image_control_panel.clear_all_display()
    
    def _create_temperature_reconstruction_section(self, layout: QVBoxLayout):
        """创建温度场重构特殊控制区域"""
        # 创建框架
        recon_frame = QFrame()
        recon_frame.setFrameStyle(QFrame.Shape.Box)
        recon_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #BDC3C7;
                border-radius: 5px;
                background-color: #FFF8DC;
                margin: 2px;
            }
        """)
        
        recon_layout = QVBoxLayout(recon_frame)
        recon_layout.setContentsMargins(10, 8, 10, 8)
        recon_layout.setSpacing(6)
        
        # 标题
        title = QLabel("🔄 温度场重构")
        title.setStyleSheet("font-weight: bold; color: #8B4513;")
        recon_layout.addWidget(title)
        
        # "从温度场获取测点温度"按钮
        self.sample_from_field_button = QPushButton("📊 从温度场获取测点温度")
        self.sample_from_field_button.setStyleSheet("""
            QPushButton {
                background-color: #FF8C00;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #FF7F00;
            }
            QPushButton:pressed {
                background-color: #FF6347;
            }
            QPushButton:disabled {
                background-color: #D3D3D3;
                color: #A0A0A0;
            }
        """)
        self.sample_from_field_button.clicked.connect(self._on_sample_from_temperature_field)
        recon_layout.addWidget(self.sample_from_field_button)
        
        # 说明文字
        info_label = QLabel("需要先计算POD温度场，然后放置传感器测点")
        info_label.setStyleSheet("color: #666; font-size: 10px;")
        info_label.setWordWrap(True)
        recon_layout.addWidget(info_label)
        
        layout.addWidget(recon_frame)
    
    def _on_sample_from_temperature_field(self):
        """从温度场获取测点温度按钮回调"""
        print("[RightPanel] 开始从温度场采样测点温度")
        
        # 获取图像管理器
        from image_manager import get_image_manager
        image_manager = get_image_manager()
        
        # 检查是否有POD温度场
        if not image_manager.is_cached('pod_temperature'):
            print("[RightPanel] 错误: 未找到POD温度场，请先计算POD温度场")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self.main_window, "温度场未计算", 
                               "请先点击'计算POD温度场'按钮计算温度场！")
            return
        
        # 获取所有传感器
        sensors = []
        for item in self.main_window.scene.items():
            if hasattr(item, 'get_state') and item.get_state().get('type') == 'sensor':
                sensors.append(item)
        
        if not sensors:
            print("[RightPanel] 错误: 未找到传感器")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self.main_window, "无传感器", 
                               "请先放置传感器测点！")
            return
        
        # 调用主窗口的采样方法
        try:
            success_count = self.main_window.sample_temperatures_from_pod_field(sensors)
            print(f"[RightPanel] 温度采样完成，成功更新 {success_count} 个传感器")
            
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self.main_window, "采样完成", 
                                   f"已从POD温度场为 {success_count} 个传感器采样温度值！")
            
        except Exception as e:
            print(f"[RightPanel] 温度采样失败: {e}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self.main_window, "采样失败", 
                               f"温度采样失败: {str(e)}")
