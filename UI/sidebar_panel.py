"""
侧边栏面板组件
包含绘制模式控制、SDF控制和组件列表管理
"""

from typing import TYPE_CHECKING
from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QButtonGroup, QFrame, QCheckBox, QScrollArea,
                             QDoubleSpinBox, QTabWidget, QTableWidget, QTableWidgetItem,
                             QHeaderView, QFormLayout, QGridLayout)
from PyQt6.QtCore import Qt
from ui_constants import StyleSheets, Icons, ComponentNames
from ui_utils import format_position_text, format_size_text

if TYPE_CHECKING:
    from main_window import MainWindow


class SidebarPanel:
    """侧边栏面板管理器"""
    
    def __init__(self, main_window: 'MainWindow'):
        self.main_window = main_window
        self.setup_sidebar()
    
    def setup_sidebar(self):
        """创建侧边栏面板"""
        # 创建dock widget
        self.sidebar_dock = QDockWidget("Control Panel", self.main_window)
        self.sidebar_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | 
                                          Qt.DockWidgetArea.RightDockWidgetArea)
        
        # 创建主标签页组件
        self.main_tabs = QTabWidget()
        self.main_tabs.setStyleSheet(StyleSheets.TAB_WIDGET)
        
        # 创建两个主标签页
        self._create_component_layout_tab()
        self._create_temperature_reconstruction_tab()
        
        # 设置侧边栏内容
        self.sidebar_dock.setWidget(self.main_tabs)
        self.main_window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.sidebar_dock)
    
    def _create_component_layout_tab(self):
        """创建元件布局标签页"""
        # 创建标签页内容
        layout_widget = QWidget()
        layout_layout = QVBoxLayout(layout_widget)
        
        # 添加绘制模式区域
        self._create_drawing_mode_section(layout_layout)
        self._create_separator(layout_layout)
        
        # 添加SDF控制区域
        self._create_sdf_section(layout_layout)
        self._create_separator(layout_layout)
        
        # 添加组件管理区域（标签页形式）
        self._create_components_tabs_section(layout_layout)
        
        # 添加弹性空间
        layout_layout.addStretch()
        
        # 添加到主标签页
        self.main_tabs.addTab(layout_widget, "元件布局 (Component Layout)")
    
    def _create_temperature_reconstruction_tab(self):
        """创建温度重构标签页"""
        # 创建标签页内容
        temp_widget = QWidget()
        temp_layout = QVBoxLayout(temp_widget)
        
        # 添加测点放置模式区域
        self._create_sensor_mode_section(temp_layout)
        self._create_separator(temp_layout)
        
        # 添加重构控制区域
        self._create_reconstruction_section(temp_layout)
        self._create_separator(temp_layout)
        
        # 添加测点管理区域（标签页形式）
        self._create_sensors_tabs_section(temp_layout)
        
        # 添加弹性空间
        temp_layout.addStretch()
        
        # 添加到主标签页
        self.main_tabs.addTab(temp_widget, "温度重构 (Temperature Reconstruction)")
    
    def _create_sensor_mode_section(self, layout: QVBoxLayout):
        """创建测点放置模式区域"""
        # 区域标题
        mode_label = QLabel("测点放置模式 Sensor Placement")
        mode_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mode_label.setStyleSheet(StyleSheets.SECTION_LABEL)
        layout.addWidget(mode_label)
        
        # 创建按钮组
        self.sensor_button_group = QButtonGroup()
        
        # 创建按钮布局 - 一行两个按钮
        button_layout = QHBoxLayout()
        
        # 添加测点按钮
        self.add_sensor_button = QPushButton(f"{Icons.ADD_SENSOR} 添加测点")
        self.add_sensor_button.setCheckable(True)
        self.add_sensor_button.setStyleSheet(StyleSheets.DRAWING_MODE_BUTTON)
        self.add_sensor_button.clicked.connect(self._on_add_sensor_clicked)
        self.sensor_button_group.addButton(self.add_sensor_button)
        button_layout.addWidget(self.add_sensor_button)
        
        # 选择模式按钮（用于取消测点放置模式）
        self.select_mode_button = QPushButton(f"{Icons.NONE_MODE} 选择模式")
        self.select_mode_button.setCheckable(True)
        self.select_mode_button.setStyleSheet(StyleSheets.DRAWING_MODE_BUTTON)
        self.select_mode_button.setChecked(True)  # 默认选中
        self.select_mode_button.clicked.connect(self._on_select_mode_clicked)
        self.sensor_button_group.addButton(self.select_mode_button)
        button_layout.addWidget(self.select_mode_button)
        
        layout.addLayout(button_layout)
    
    def _create_reconstruction_section(self, layout: QVBoxLayout):
        """创建重构控制区域"""
        # 区域标题
        recon_label = QLabel("温度场重构 Temperature Reconstruction")
        recon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        recon_label.setStyleSheet(StyleSheets.SECTION_LABEL)
        layout.addWidget(recon_label)
        
        # 执行重构按钮
        self.execute_reconstruction_button = QPushButton(f"{Icons.EXECUTE_RECONSTRUCTION} 执行重构")
        self.execute_reconstruction_button.setStyleSheet(StyleSheets.TEMP_RECONSTRUCTION_BUTTON)
        self.execute_reconstruction_button.clicked.connect(self._on_execute_reconstruction_clicked)
        layout.addWidget(self.execute_reconstruction_button)
    
    def _create_sensors_tabs_section(self, layout: QVBoxLayout):
        """创建测点标签页区域"""
        # 测点管理标题
        sensors_label = QLabel("测点管理 Sensors")
        sensors_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sensors_label.setStyleSheet(StyleSheets.SECTION_LABEL)
        layout.addWidget(sensors_label)
        
        # 创建测点标签页
        self.sensors_tabs = QTabWidget()
        self.sensors_tabs.setStyleSheet(StyleSheets.TAB_WIDGET)
        self.sensors_tabs.setMaximumHeight(300)
        layout.addWidget(self.sensors_tabs)
        
        # 初始显示无测点提示
        self._show_no_sensors_message()
    
    def _show_no_sensors_message(self):
        """显示无测点提示"""
        no_sensors_widget = QWidget()
        no_sensors_layout = QVBoxLayout(no_sensors_widget)
        
        no_sensors_label = QLabel("暂无测点\n请点击'添加测点'后在画布上放置")
        no_sensors_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        no_sensors_label.setStyleSheet(StyleSheets.NO_COMPONENTS_LABEL)
        no_sensors_layout.addWidget(no_sensors_label)
        
        self.sensors_tabs.clear()
        self.sensors_tabs.addTab(no_sensors_widget, "无测点")
    
    def _on_add_sensor_clicked(self):
        """添加测点按钮点击处理"""
        if self.add_sensor_button.isChecked():
            # 进入测点放置模式
            self.main_window.set_sensor_placement_mode(True)
        else:
            # 退出测点放置模式
            self.main_window.set_sensor_placement_mode(False)
    
    def _on_select_mode_clicked(self):
        """选择模式按钮点击处理"""
        if self.select_mode_button.isChecked():
            # 退出测点放置模式，进入选择模式
            self.main_window.set_sensor_placement_mode(False)
    
    def _on_execute_reconstruction_clicked(self):
        """执行重构按钮点击处理"""
        self.main_window.execute_temperature_reconstruction()
    
    def _create_drawing_mode_section(self, layout: QVBoxLayout):
        """创建绘制模式区域"""
        # 绘制模式标题
        mode_label = QLabel("绘制模式 Drawing Mode")
        mode_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mode_label.setStyleSheet(StyleSheets.SECTION_LABEL)
        layout.addWidget(mode_label)
        
        # 创建按钮组，支持取消选中
        self.mode_button_group = QButtonGroup()
        self.mode_button_group.setExclusive(False)  # 允许取消选中
        
        # 创建网格布局，2×2排列
        button_grid = QGridLayout()
        button_grid.setSpacing(8)  # 设置按钮间距
        
        # 矩形按钮
        self.rect_button = QPushButton("🔲 矩形")
        self.rect_button.setCheckable(True)
        self.rect_button.clicked.connect(lambda: self.main_window.toggle_draw_mode('rect'))
        self.rect_button.setStyleSheet(StyleSheets.RECT_BUTTON)
        self.rect_button.setFixedHeight(40)  # 固定高度
        self.mode_button_group.addButton(self.rect_button)
        button_grid.addWidget(self.rect_button, 0, 0)
        
        # 圆形按钮
        self.circle_button = QPushButton("⭕ 圆形")
        self.circle_button.setCheckable(True)
        self.circle_button.clicked.connect(lambda: self.main_window.toggle_draw_mode('circle'))
        self.circle_button.setStyleSheet(StyleSheets.CIRCLE_BUTTON)
        self.circle_button.setFixedHeight(40)  # 固定高度
        self.mode_button_group.addButton(self.circle_button)
        button_grid.addWidget(self.circle_button, 0, 1)
        
        # 胶囊按钮
        self.capsule_button = QPushButton("💊 胶囊")
        self.capsule_button.setCheckable(True)
        self.capsule_button.clicked.connect(lambda: self.main_window.toggle_draw_mode('capsule'))
        self.capsule_button.setStyleSheet(StyleSheets.CAPSULE_BUTTON)
        self.capsule_button.setFixedHeight(40)  # 固定高度
        self.mode_button_group.addButton(self.capsule_button)
        button_grid.addWidget(self.capsule_button, 1, 0)
        
        # 散热器按钮
        self.radiator_button = QPushButton("📐 散热器")
        self.radiator_button.setCheckable(True)
        self.radiator_button.clicked.connect(lambda: self.main_window.toggle_draw_mode('radiator'))
        self.radiator_button.setStyleSheet(StyleSheets.RADIATOR_BUTTON)
        self.radiator_button.setFixedHeight(40)  # 固定高度
        self.mode_button_group.addButton(self.radiator_button)
        button_grid.addWidget(self.radiator_button, 1, 1)
        
        # 将网格布局添加到主布局
        layout.addLayout(button_grid)
    
    def _create_sdf_section(self, layout: QVBoxLayout):
        """创建SDF控制区域"""
        # SDF控制标题
        sdf_label = QLabel("SDF可视化 SDF Visualization")
        sdf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sdf_label.setStyleSheet(StyleSheets.SECTION_LABEL)
        layout.addWidget(sdf_label)
        
        # SDF显示开关
        self.sdf_show_checkbox = QCheckBox("显示SDF背景 Show SDF Background")
        self.sdf_show_checkbox.setStyleSheet(StyleSheets.SDF_CHECKBOX)
        self.sdf_show_checkbox.toggled.connect(self.main_window.on_sdf_show_toggled)
        layout.addWidget(self.sdf_show_checkbox)
        
        # SDF更新按钮（初始隐藏）
        self.sdf_update_button = QPushButton(f"{Icons.UPDATE_SDF} SDF")
        self.sdf_update_button.setVisible(False)
        self.sdf_update_button.clicked.connect(self.main_window.update_sdf_background)
        self.sdf_update_button.setStyleSheet(StyleSheets.SDF_UPDATE_BUTTON)
        layout.addWidget(self.sdf_update_button)
    
    def _create_components_tabs_section(self, layout: QVBoxLayout):
        """创建组件标签页区域"""
        # 组件管理标题
        components_label = QLabel("组件管理 Components")
        components_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        components_label.setStyleSheet(StyleSheets.SECTION_LABEL)
        layout.addWidget(components_label)
        
        # 创建组件标签页
        self.components_tabs = QTabWidget()
        self.components_tabs.setStyleSheet(StyleSheets.TAB_WIDGET)
        self.components_tabs.setMaximumHeight(300)
        layout.addWidget(self.components_tabs)
        
        # 初始显示无组件提示
        self._show_no_components_message()
    
    def _show_no_components_message(self):
        """显示无组件提示"""
        no_components_widget = QWidget()
        no_components_layout = QVBoxLayout(no_components_widget)
        
        no_components_label = QLabel("暂无组件\n请在右侧画布中绘制组件")
        no_components_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        no_components_label.setStyleSheet(StyleSheets.NO_COMPONENTS_LABEL)
        no_components_layout.addWidget(no_components_label)
        
        self.components_tabs.clear()
        self.components_tabs.addTab(no_components_widget, "无组件")
    
    def _create_components_section(self, layout: QVBoxLayout):
        """创建组件列表区域"""
        # 组件列表标题
        components_label = QLabel("Components")
        components_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        components_label.setStyleSheet(StyleSheets.SECTION_LABEL)
        layout.addWidget(components_label)
        
        # 创建滚动区域用于组件列表
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(200)
        scroll_area.setStyleSheet(StyleSheets.SCROLL_AREA)
        
        # 组件列表容器
        self.components_widget = QWidget()
        self.components_layout = QVBoxLayout(self.components_widget)
        self.components_layout.setContentsMargins(5, 5, 5, 5)
        self.components_layout.setSpacing(5)
        
        # 添加"无组件"标签
        self.no_components_label = QLabel("No components added yet")
        self.no_components_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_components_label.setStyleSheet(StyleSheets.NO_COMPONENTS_LABEL)
        self.components_layout.addWidget(self.no_components_label)
        
        self.components_layout.addStretch()
        scroll_area.setWidget(self.components_widget)
        layout.addWidget(scroll_area)
    
    def _create_separator(self, layout: QVBoxLayout):
        """创建分隔线"""
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet(StyleSheets.SEPARATOR)
        layout.addWidget(separator)
    
    def update_components_list(self):
        """更新组件列表显示"""
        # 获取所有组件（排除传感器）
        components = []
        for item in self.main_window.scene.items():
            if hasattr(item, 'get_state') and item.get_state().get('type') != 'sensor':
                components.append(item)
        
        # 清除现有标签页
        self.components_tabs.clear()
        
        if not components:
            # 显示无组件提示
            self._show_no_components_message()
        else:
            # 为每个组件创建标签页
            for i, component_item in enumerate(components):
                self._create_component_tab(component_item, i)
    
    def _create_component_tab(self, component_item, index):
        """为组件创建标签页"""
        state = component_item.get_state()
        
        # 创建标签页内容
        tab_widget = QWidget()
        tab_layout = QVBoxLayout(tab_widget)
        tab_layout.setContentsMargins(10, 10, 10, 10)
        tab_layout.setSpacing(8)
        
        # 组件类型和ID标题（简化显示：编号 + 图标）
        component_icon = ComponentNames.COMPONENT_TYPE_ICONS.get(state['type'], '📦')
        tab_name = f"{index + 1} {component_icon}"
        
        # 创建属性表格
        self._create_property_table(tab_layout, component_item, state)
        
        # 添加删除按钮
        delete_button = QPushButton(f"{Icons.DELETE} 删除此元件")
        delete_button.setStyleSheet(StyleSheets.DELETE_COMPONENT_BUTTON)
        delete_button.clicked.connect(lambda: self._delete_component(component_item))
        tab_layout.addWidget(delete_button)
        
        # 添加弹性空间
        tab_layout.addStretch()
        
        # 添加标签页
        self.components_tabs.addTab(tab_widget, tab_name)
    
    def _create_property_table(self, layout, component_item, state):
        """创建属性表格"""
        # 属性表格标题
        properties_label = QLabel("属性 (Properties)")
        properties_label.setStyleSheet(StyleSheets.COMPONENT_TITLE)
        layout.addWidget(properties_label)
        
        # 创建表格
        table = QTableWidget(0, 2)  # 2列：属性名、值
        table.setHorizontalHeaderLabels(["属性", "值"])
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        table.setMaximumHeight(200)
        table.setStyleSheet(StyleSheets.SCROLL_AREA)
        
        # 添加基础属性
        self._add_property_row(table, "类型", ComponentNames.DISPLAY_NAMES.get(state['type'], state['type']))
        
        # 位置属性（可编辑）
        coords = state['coords']
        meters_coords = self._pixels_to_meters_coords(coords)
        pos_text = f"({meters_coords[0]:.3f}, {meters_coords[1]:.3f})m"
        
        # 创建可编辑的位置行
        position_item = QTableWidgetItem(pos_text)
        position_item.setData(Qt.ItemDataRole.UserRole, component_item)  # 存储组件引用
        position_item.setToolTip("格式: (x, y)m，例如: (0.150, 0.200)m")
        table.insertRow(table.rowCount())
        table.setItem(table.rowCount()-1, 0, QTableWidgetItem("位置"))
        table.setItem(table.rowCount()-1, 1, position_item)
        
        # 设置名称列为只读
        name_item = table.item(table.rowCount()-1, 0)
        name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        
        # 尺寸属性
        if state['type'] == 'circle':
            meters_size = self._pixels_to_meters_size(state['size'])
            size_text = f"半径: {meters_size:.3f}m"
        elif state['type'] == 'radiator':
            start_point = self._pixels_to_meters_coords(state['start_point'])
            end_point = self._pixels_to_meters_coords(state['end_point'])
            size_text = f"起点: ({start_point[0]:.3f}, {start_point[1]:.3f})m\n终点: ({end_point[0]:.3f}, {end_point[1]:.3f})m"
        else:
            meters_size = self._pixels_to_meters_size(state['size'])
            size_text = f"{meters_size[0]:.3f}×{meters_size[1]:.3f}m"
        self._add_property_row(table, "尺寸", size_text)
        
        # 功率属性（可编辑）
        if state['type'] != 'radiator':
            power_item = QTableWidgetItem(f"{state['power']:.1f}W")
            power_item.setData(Qt.ItemDataRole.UserRole, component_item)  # 存储组件引用
            table.insertRow(table.rowCount())
            table.setItem(table.rowCount()-1, 0, QTableWidgetItem("功率"))
            table.setItem(table.rowCount()-1, 1, power_item)
            
            # 连接功率编辑信号
            table.itemChanged.connect(self._on_property_changed)
        else:
            # 散热器的散热效率
            heat_rate = state.get('heat_dissipation_rate', 10.0)
            heat_item = QTableWidgetItem(f"{heat_rate:.1f}")
            heat_item.setData(Qt.ItemDataRole.UserRole, component_item)
            table.insertRow(table.rowCount())
            table.setItem(table.rowCount()-1, 0, QTableWidgetItem("散热效率"))
            table.setItem(table.rowCount()-1, 1, heat_item)
            
            table.itemChanged.connect(self._on_property_changed)
        
        layout.addWidget(table)
    
    def _add_property_row(self, table, name, value):
        """添加属性行"""
        table.insertRow(table.rowCount())
        name_item = QTableWidgetItem(name)
        name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # 只读
        value_item = QTableWidgetItem(str(value))
        value_item.setFlags(value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # 只读
        
        table.setItem(table.rowCount()-1, 0, name_item)
        table.setItem(table.rowCount()-1, 1, value_item)
    
    def _pixels_to_meters_coords(self, coords):
        """将像素坐标转换为米"""
        from ui_utils import convert_coords_to_meters
        return convert_coords_to_meters(coords, self.main_window.scene_scale)
    
    def _pixels_to_meters_size(self, size):
        """将像素尺寸转换为米"""
        from ui_utils import convert_size_to_meters
        return convert_size_to_meters(size, self.main_window.scene_scale)
    
    def _on_property_changed(self, item):
        """属性值变化处理"""
        if item.column() == 1:  # 值列
            component_item = item.data(Qt.ItemDataRole.UserRole)
            if component_item:
                property_name = item.tableWidget().item(item.row(), 0).text()
                try:
                    if property_name == "功率":
                        # 更新功率
                        power_text = item.text().replace('W', '')
                        new_power = float(power_text)
                        self.main_window.update_component_power(component_item, new_power)
                    elif property_name == "散热效率":
                        # 更新散热效率
                        heat_rate = float(item.text())
                        component_item.state['heat_dissipation_rate'] = heat_rate
                        component_item.update()
                    elif property_name == "位置":
                        # 更新位置
                        self._update_component_position(component_item, item)
                except ValueError:
                    # 如果输入无效，恢复原值
                    if property_name == "功率":
                        item.setText(f"{component_item.state['power']:.1f}W")
                    elif property_name == "位置":
                        coords = component_item.state['coords']
                        meters_coords = self._pixels_to_meters_coords(coords)
                        item.setText(f"({meters_coords[0]:.3f}, {meters_coords[1]:.3f})m")
                    elif property_name == "散热效率":
                        item.setText(f"{component_item.state.get('heat_dissipation_rate', 10.0):.1f}")
    
    def _update_component_position(self, component_item, item):
        """更新组件位置"""
        import re
        
        # 解析位置文本：(x, y)m
        text = item.text().strip()
        # 使用正则表达式提取坐标
        match = re.match(r'\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)m?', text)
        
        if match:
            # 获取米单位坐标
            x_meters = float(match.group(1))
            y_meters = float(match.group(2))
            
            # 转换为像素坐标
            x_pixels = x_meters * self.main_window.scene_scale
            y_pixels = y_meters * self.main_window.scene_scale
            
            # 检查边界
            scene_rect = self.main_window.scene.sceneRect()
            if (0 <= x_pixels <= scene_rect.width() and 
                0 <= y_pixels <= scene_rect.height()):
                
                # 更新组件状态
                component_item.state['coords'] = (x_pixels, y_pixels)
                
                # 更新图形位置
                component_item.setPos(x_pixels, y_pixels)
                
                # 触发场景更新
                self.main_window.scene.item_position_changed.emit()
                
                # 更新显示文本为标准格式
                item.setText(f"({x_meters:.3f}, {y_meters:.3f})m")
                
                self.main_window.status_bar.showMessage(f"组件位置已更新至 ({x_meters:.3f}, {y_meters:.3f})m")
            else:
                # 超出边界，恢复原值
                raise ValueError("位置超出场景边界")
        else:
            # 格式错误，抛出异常让上层处理
            raise ValueError("位置格式错误，请使用 (x, y)m 格式")
    
    def _delete_component(self, component_item):
        """删除组件"""
        # 从场景中移除组件
        self.main_window.scene.removeItem(component_item)
        
        # 更新组件列表
        self.update_components_list()
        
        # 强制刷新场景
        self.main_window.scene.update()
        for view in self.main_window.scene.views():
            view.update()
    
    def update_sensor_list(self):
        """更新传感器列表显示（使用标签页格式）"""
        # 获取所有传感器
        sensors = []
        for item in self.main_window.scene.items():
            if hasattr(item, 'get_state') and item.get_state().get('type') == 'sensor':
                sensors.append(item)
        
        # 清除现有标签页
        self.sensors_tabs.clear()
        
        if not sensors:
            # 显示无测点提示
            self._show_no_sensors_message()
        else:
            # 为每个传感器创建标签页
            for i, sensor_item in enumerate(sensors):
                self._create_sensor_tab(sensor_item, i)
    
    def _create_sensor_tab(self, sensor_item, index):
        """为传感器创建标签页"""
        state = sensor_item.get_state()
        
        # 创建标签页名称
        tab_name = f"测点{index + 1}"
        
        # 创建标签页内容
        tab_widget = QWidget()
        tab_layout = QVBoxLayout(tab_widget)
        tab_layout.setContentsMargins(10, 10, 10, 10)
        tab_layout.setSpacing(10)
        
        # 添加传感器信息和编辑功能
        self._create_sensor_properties(tab_layout, sensor_item, state)
        
        # 添加删除按钮
        self._create_sensor_delete_button(tab_layout, sensor_item)
        
        # 添加弹性空间
        tab_layout.addStretch()
        
        # 添加标签页
        self.sensors_tabs.addTab(tab_widget, tab_name)
    
    def _create_sensor_properties(self, layout, sensor_item, state):
        """创建传感器属性区域"""
        # 属性标题
        properties_label = QLabel("属性 (Properties)")
        properties_label.setStyleSheet(StyleSheets.COMPONENT_TITLE)
        layout.addWidget(properties_label)
        
        # 位置信息（只读）
        meters_coords = self._pixels_to_meters_coords(state['coords'])
        pos_label = QLabel(f"位置: ({meters_coords[0]:.3f}, {meters_coords[1]:.3f})m")
        pos_label.setStyleSheet(StyleSheets.COMPONENT_INFO)
        layout.addWidget(pos_label)
        
        # 温度编辑器
        temp_layout = QHBoxLayout()
        temp_label = QLabel("温度 (Temperature):")
        temp_label.setStyleSheet(StyleSheets.COMPONENT_INFO)
        temp_layout.addWidget(temp_label)
        
        temp_spinbox = QDoubleSpinBox()
        temp_spinbox.setRange(0.0, 1000.0)  # 0K到1000K
        temp_spinbox.setSuffix(" K")
        temp_spinbox.setDecimals(1)
        temp_spinbox.setValue(state.get('temperature', 273.15))  # 默认0°C
        temp_spinbox.setStyleSheet(StyleSheets.PROPERTY_EDITOR)
        temp_spinbox.valueChanged.connect(
            lambda value, item=sensor_item: self._on_sensor_temperature_changed(item, value)
        )
        temp_layout.addWidget(temp_spinbox)
        
        layout.addLayout(temp_layout)
    
    def _create_sensor_delete_button(self, layout, sensor_item):
        """创建传感器删除按钮"""
        delete_button = QPushButton(f"{Icons.DELETE} 删除测点")
        delete_button.setStyleSheet(StyleSheets.DELETE_BUTTON)
        delete_button.clicked.connect(lambda: self._delete_sensor(sensor_item))
        layout.addWidget(delete_button)
    
    def _delete_sensor(self, sensor_item):
        """删除传感器"""
        # 从场景中移除
        self.main_window.scene.removeItem(sensor_item)
        
        # 更新传感器列表
        self.update_sensor_list()
        
        # 强制刷新场景
        self.main_window.scene.update()
        for view in self.main_window.scene.views():
            view.update()
    

    
    def _on_sensor_temperature_changed(self, sensor_item, value):
        """处理传感器温度值变化"""
        # 更新传感器状态
        state = sensor_item.get_state()
        state['temperature'] = value
        sensor_item.set_state(state)
        
        # 强制重绘传感器
        sensor_item.update()
        
        # 强制刷新场景
        self.main_window.scene.update()
    
