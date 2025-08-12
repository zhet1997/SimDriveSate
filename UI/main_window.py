"""
主窗口类
包含应用程序的主要界面逻辑、工具栏、文件操作和SDF管理
"""

import sys
import os
from typing import List, Dict, Optional
import yaml
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QGraphicsView, QToolBar, QFileDialog, 
                             QMessageBox, QStatusBar, QGraphicsPixmapItem)
from PyQt6.QtGui import QPainter, QColor, QPen, QAction, QFont, QPixmap
from PyQt6.QtCore import QThread, QTimer
from graphics_scene import CustomGraphicsScene
from graphics_items import create_component_item, RectItem, CircleItem, CapsuleItem
from sidebar_panel import SidebarPanel
from worker_thread import Worker
from ui_constants import (SCENE_SCALE, DEFAULT_LAYOUT_SIZE, DEFAULT_THERMAL_CONDUCTIVITY,
                          DEFAULT_MESH_RESOLUTION, GRID_INTERVAL_METERS, Colors, Icons,
                          ComponentNames, SDFConfig)
from ui_utils import (convert_component_to_meters, create_sdf_figure, create_temperature_figure,
                      calculate_sdf_grid_shape)

# 导入新的模块
try:
    from data_bridge import JSONComponentHandler, DataFormatConverter
    from backends import ThermalSimulationBackend, FieldType
    from visualization import FieldVisualizer, VisualizationConfig, ThermalFieldPlotter
    NEW_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"警告：新功能模块导入失败，部分功能不可用: {e}")
    NEW_FEATURES_AVAILABLE = False

# 添加layout目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'layout'))


class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Satellite Component Visualization & Physics Field Prediction")
        # 🔄 调整窗口尺寸以适应三列式布局
        self.setGeometry(100, 100, 1200, 700)
        
        # 🔧 修复Wayland显示协议兼容性
        self.setMinimumSize(1000, 600)  # 设置最小尺寸
        self.setSizePolicy(
            self.sizePolicy().horizontalPolicy(), 
            self.sizePolicy().verticalPolicy()
        )
        
        # 初始化布局参数
        self.layout_size = DEFAULT_LAYOUT_SIZE
        self.k = DEFAULT_THERMAL_CONDUCTIVITY
        self.mesh_resolution = DEFAULT_MESH_RESOLUTION
        self.scene_scale = SCENE_SCALE
        
        # 创建中心部件和布局
        self.scene = CustomGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.setCentralWidget(self.view)
        
        # SDF背景图层
        self.sdf_background_item: Optional[QGraphicsPixmapItem] = None
        self.sdf_visible = False
        
        # 设置视图
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        self.view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        
        # 设置场景大小
        self._setup_scene()
        
        # 创建UI组件
        self._create_toolbar()
        self._create_sidebar()
        self._create_right_panel()  # 🆕 创建右侧面板
        self._create_status_bar()
        
        # 🆕 扩展侧边栏功能
        try:
            from sidebar_panel_extension import extend_sidebar_panel
            extend_sidebar_panel()
        except ImportError as e:
            print(f"[MainWindow] 警告: 无法加载侧边栏扩展: {e}")
        
        # 初始化工作线程
        self._setup_worker_thread()
        
        # 连接场景更新信号
        self.scene.scene_updated.connect(self.on_scene_updated)
        
        # 存储计算结果图像
        self.result_image: Optional[np.ndarray] = None
        
        # 设置初始绘制模式为None（选择模式）
        self.set_draw_mode(None)
        
        # 初始化组件列表
        self.sidebar.update_components_list()
        
        # 初始化统一数据管理中心
        from component_manager import get_component_manager
        from data_synchronizer import get_data_synchronizer
        
        self.component_manager = get_component_manager()
        self.data_sync = get_data_synchronizer()
        
        # 连接数据变更信号到UI更新
        self.data_sync.ui_update_needed.connect(self.update_components_list)
        
        # 🆕 连接选择同步信号
        self.data_sync.selection_changed.connect(self.on_component_selected)
        self.data_sync.selection_cleared.connect(self.on_selection_cleared)
        
        print("[MainWindow] 数据管理中心初始化完成")
        
        # 添加简单的调试监控
        self.component_manager.component_added.connect(
            lambda cid: print(f"[DEBUG] 组件添加: {cid}")
        )
        self.component_manager.component_removed.connect(
            lambda cid: (
                print(f"[DEBUG] 组件删除: {cid}"),
                self._remove_graphics_item_by_id(cid)
            )
        )
        
        # 初始化热仿真后端（如果新功能可用）
        if NEW_FEATURES_AVAILABLE:
            self.thermal_backend = ThermalSimulationBackend()
            self.thermal_backend.initialize()
        else:
            self.thermal_backend = None
        
        # 🆕 初始化图像管理器
        from image_manager import get_image_manager
        self.image_manager = get_image_manager()
        self.image_manager.set_scene(self.scene)
        
        # 注册图像计算回调
        self._register_image_compute_callbacks()
    
    def _setup_scene(self):
        """设置场景参数"""
        scene_width = self.layout_size[0] * self.scene_scale
        scene_height = self.layout_size[1] * self.scene_scale
        self.scene.setSceneRect(0, 0, scene_width, scene_height)
        
        # 添加坐标网格
        self._add_grid()
    
    def _add_grid(self):
        """添加坐标网格、比例尺和坐标标签"""
        width, height = self.layout_size
        scene_width = width * self.scene_scale
        scene_height = height * self.scene_scale
        grid_interval = GRID_INTERVAL_METERS * self.scene_scale
        
        # 绘制垂直线和X轴坐标标签
        pen = QPen(QColor(*Colors.GRID_LINE), 1)
        for i, x in enumerate(np.arange(0, scene_width + grid_interval, grid_interval)):
            self.scene.addLine(x, 0, x, scene_height, pen)
            # 添加X轴坐标标签（毫米单位）
            x_mm = i * GRID_INTERVAL_METERS * 1000
            if i % 2 == 0:  # 只显示偶数标签，避免拥挤
                text_item = self.scene.addText(f"{x_mm:.0f}", QFont("Arial", 7))
                text_item.setPos(x - 8, scene_height + 5)
                text_item.setDefaultTextColor(QColor(*Colors.GRID_LABEL))
            
        # 绘制水平线和Y轴坐标标签
        for i, y in enumerate(np.arange(0, scene_height + grid_interval, grid_interval)):
            self.scene.addLine(0, y, scene_width, y, pen)
            # 添加Y轴坐标标签（毫米单位）
            y_mm = (height * 1000) - (i * GRID_INTERVAL_METERS * 1000)  # Y轴从上到下递减
            if i % 2 == 0:  # 只显示偶数标签，避免拥挤
                text_item = self.scene.addText(f"{y_mm:.0f}", QFont("Arial", 7))
                text_item.setPos(-25, y - 8)
                text_item.setDefaultTextColor(QColor(*Colors.GRID_LABEL))
            
        # 绘制边界
        border_pen = QPen(QColor(*Colors.BORDER_LINE), 2)
        self.scene.addRect(0, 0, scene_width, scene_height, border_pen)
        
        # 添加坐标轴单位标识
        x_unit_label = self.scene.addText("X (mm)", QFont("Arial", 8, QFont.Weight.Bold))
        x_unit_label.setPos(scene_width/2 - 20, scene_height + 20)
        x_unit_label.setDefaultTextColor(QColor(*Colors.GRID_LABEL))
        
        y_unit_label = self.scene.addText("Y (mm)", QFont("Arial", 8, QFont.Weight.Bold))
        y_unit_label.setPos(-55, scene_height/2 - 10)
        y_unit_label.setDefaultTextColor(QColor(*Colors.GRID_LABEL))
        y_unit_label.setRotation(-90)  # 垂直显示
        
        # 添加比例尺
        self._add_scale_ruler()
    
    def _add_scale_ruler(self):
        """添加比例尺"""
        width, height = self.layout_size
        scene_width = width * self.scene_scale
        scene_height = height * self.scene_scale
        
        # 比例尺位置（右下角）
        ruler_x = scene_width - 80
        ruler_y = scene_height - 30
        
        # 比例尺长度（20mm）
        ruler_length_mm = 20
        ruler_length_pixels = (ruler_length_mm / 1000) * self.scene_scale
        
        # 绘制比例尺线条
        ruler_pen = QPen(QColor(0, 0, 0), 2)
        self.scene.addLine(ruler_x, ruler_y, ruler_x + ruler_length_pixels, ruler_y, ruler_pen)
        
        # 比例尺端点标记
        self.scene.addLine(ruler_x, ruler_y - 3, ruler_x, ruler_y + 3, ruler_pen)
        self.scene.addLine(ruler_x + ruler_length_pixels, ruler_y - 3, 
                          ruler_x + ruler_length_pixels, ruler_y + 3, ruler_pen)
        
        # 比例尺标签
        scale_text = self.scene.addText(f"{ruler_length_mm}mm", QFont("Arial", 9, QFont.Weight.Bold))
        scale_text.setPos(ruler_x + ruler_length_pixels/2 - 15, ruler_y - 20)
        scale_text.setDefaultTextColor(QColor(0, 0, 0))
        
        # 比例尺背景框（提高可读性）
        scale_bg = self.scene.addRect(ruler_x - 5, ruler_y - 25, ruler_length_pixels + 35, 35,
                                     QPen(QColor(200, 200, 200)), QColor(255, 255, 255, 200))
        scale_bg.setZValue(-1)  # 背景在后面
    
    def _create_toolbar(self):
        """创建主工具栏"""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        # YAML文件操作按钮
        load_yaml_action = QAction(f"{Icons.LOAD_FILE} 加载 YAML", self)
        load_yaml_action.triggered.connect(self.load_from_yaml)
        toolbar.addAction(load_yaml_action)
        
        save_yaml_action = QAction(f"{Icons.SAVE_FILE} 保存 YAML", self)
        save_yaml_action.triggered.connect(self.save_to_yaml)
        toolbar.addAction(save_yaml_action)
        
        # 添加分隔符
        toolbar.addSeparator()
        
        # JSON文件操作按钮（仅在新功能可用时添加）
        if NEW_FEATURES_AVAILABLE:
            load_json_action = QAction(f"{Icons.LOAD_FILE} 加载 JSON", self)
            load_json_action.triggered.connect(self.load_from_json)
            toolbar.addAction(load_json_action)
            
            save_json_action = QAction(f"{Icons.SAVE_FILE} 保存 JSON", self)
            save_json_action.triggered.connect(self.save_to_json)
            toolbar.addAction(save_json_action)
            
            # 添加分隔符
            toolbar.addSeparator()
            
            # 热仿真按钮
            thermal_action = QAction("🔥 热仿真", self)
            thermal_action.triggered.connect(self.run_thermal_simulation)
            toolbar.addAction(thermal_action)
    
    def _create_sidebar(self):
        """创建侧边栏"""
        self.sidebar = SidebarPanel(self)
    
    def _create_right_panel(self):
        """🆕 创建右侧面板"""
        from right_panel import RightPanel
        self.right_panel = RightPanel(self)
        
        # 设置输出重定向
        from console_output_redirect import get_output_manager
        self.output_manager = get_output_manager()
        self.output_manager.setup_redirection(self.right_panel)
    
    def _create_status_bar(self):
        """创建状态栏"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def _setup_worker_thread(self):
        """设置工作线程"""
        self.thread = QThread()
        self.worker = Worker()
        self.worker.moveToThread(self.thread)
        self.thread.start()
        
        # 连接工作线程信号
        self.worker.computation_complete.connect(self.on_computation_complete)
        self.worker.temperature_reconstruction_complete.connect(self.on_temperature_reconstruction_complete)
    
    def toggle_draw_mode(self, mode: str):
        """切换绘制模式（支持取消选中）"""
        # 获取当前按钮状态
        current_button = getattr(self.sidebar, f'{mode}_button')
        
        if current_button.isChecked():
            # 如果当前按钮被选中，激活该模式
            # 先取消其他按钮
            self.sidebar.rect_button.setChecked(mode == 'rect')
            self.sidebar.circle_button.setChecked(mode == 'circle')
            self.sidebar.capsule_button.setChecked(mode == 'capsule')
            self.sidebar.radiator_button.setChecked(mode == 'radiator')
            self.set_draw_mode(mode)
        else:
            # 如果当前按钮被取消选中，进入None模式
            self.sidebar.rect_button.setChecked(False)
            self.sidebar.circle_button.setChecked(False)
            self.sidebar.capsule_button.setChecked(False)
            self.sidebar.radiator_button.setChecked(False)
            self.set_draw_mode(None)
    
    def set_draw_mode(self, mode):
        """设置绘制模式并更新状态"""
        # 设置场景的绘制模式
        self.scene.set_draw_mode(mode)
        
        # 更新状态栏显示当前模式
        if mode is None:
            self.status_bar.showMessage(f"绘制模式: {Icons.NONE_MODE} (选择模式)")
        else:
            mode_name = ComponentNames.DISPLAY_NAMES.get(mode, mode)
            self.status_bar.showMessage(f"绘制模式: {mode_name}")
    
    def on_sdf_show_toggled(self, checked: bool):
        """SDF显示开关回调"""
        self.sdf_visible = checked
        # 🔄 SDF控件现在在右侧面板中
        if hasattr(self, 'right_panel'):
            self.right_panel.sdf_update_button.setVisible(checked)
        
        # 安全检查：确保对象有效且未被删除
        if self.sdf_background_item is not None:
            try:
                # 检查对象是否仍在场景中（避免访问已删除的C++对象）
                if self.sdf_background_item.scene() is not None:
                    self.sdf_background_item.setVisible(checked)
                else:
                    # 对象已被删除，重置引用
                    self.sdf_background_item = None
            except RuntimeError:
                # C++对象已被删除，重置引用
                self.sdf_background_item = None
        
        if checked:
            self.status_bar.showMessage("SDF display enabled")
        else:
            self.status_bar.showMessage("SDF display disabled")
    
    def update_components_list(self):
        """更新组件列表显示"""
        self.sidebar.update_components_list()
    
    def update_component_power(self, component_item, new_power):
        """更新组件功率值"""
        if hasattr(component_item, 'state'):
            component_item.state['power'] = new_power
            # 触发重绘以更新显示的功率值
            component_item.update()
            self.status_bar.showMessage(f"Updated power to {new_power:.1f}W")
    
    def set_sensor_placement_mode(self, enabled: bool):
        """设置测点放置模式"""
        # 设置场景的测点放置模式
        self.scene.set_sensor_placement_mode(enabled)
        
        # 更新测点按钮状态
        if enabled:
            # 启用测点模式
            self.sidebar.add_sensor_button.setChecked(True)
            self.sidebar.select_mode_button.setChecked(False)
            
            # 取消所有绘制模式
            self.sidebar.rect_button.setChecked(False)
            self.sidebar.circle_button.setChecked(False)
            self.sidebar.capsule_button.setChecked(False)
            self.sidebar.radiator_button.setChecked(False)
            self.set_draw_mode(None)
            self.status_bar.showMessage("测点放置模式已激活 - 请在画布上点击放置测点")
        else:
            # 禁用测点模式，进入选择模式
            self.sidebar.add_sensor_button.setChecked(False)
            self.sidebar.select_mode_button.setChecked(True)
            self.status_bar.showMessage("选择模式已激活")
    
    def execute_temperature_reconstruction(self):
        """执行温度场重构"""
        # 获取所有传感器
        sensors = []
        for item in self.scene.items():
            if hasattr(item, 'get_state') and item.get_state().get('type') == 'sensor':
                sensors.append(item)
        
        if len(sensors) < 1:
            QMessageBox.warning(self, "传感器不足", 
                               "温度场重构需要至少1个传感器测点！\nTemperature reconstruction requires at least 1 sensor point!")
            return
        
        try:
            # 收集传感器数据用于重构（使用用户设置的温度值）
            sensor_data = []
            for sensor in sensors:
                state = sensor.get_state()
                temperature = state.get('temperature', 0.0)  # 如果没有温度值，使用0K
                # 将像素坐标转换为网格坐标系
                pixel_coords = state['coords']
                grid_shape = calculate_sdf_grid_shape(self.layout_size)
                # 简化坐标转换：像素坐标 -> 米坐标 -> 网格坐标
                meter_coords = (pixel_coords[0] / self.scene_scale, pixel_coords[1] / self.scene_scale)
                scene_coords = (meter_coords[0] * grid_shape[0] / self.layout_size[0],
                               meter_coords[1] * grid_shape[1] / self.layout_size[1])
                sensor_data.append({
                    'position': scene_coords,
                    'temperature': temperature
                })
            
            print(f"开始温度场重构：{len(sensors)}个传感器")
            for i, data in enumerate(sensor_data):
                print(f"  传感器{i+1}: 位置={data['position']}, 温度={data['temperature']}K")
            
            self.status_bar.showMessage(f"正在执行泰森多边形温度场重构... ({len(sensors)}个测点)")
            
            # 使用工作线程执行温度重构
            grid_shape = calculate_sdf_grid_shape(self.layout_size)
            QTimer.singleShot(0, lambda: self.worker.compute_temperature_reconstruction(sensor_data, grid_shape))
            
        except Exception as e:
            QMessageBox.critical(self, "重构失败", f"温度场重构失败: {str(e)}")
    
    def delete_selected(self):
        """删除选中的组件（已移除，功能转移到组件标签页中的删除按钮）"""
        # 这个方法已经不再使用，删除功能已转移到侧边栏的各个组件标签页中
        pass
    
    def load_from_yaml(self):
        """从YAML文件加载布局"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Layout File", "", "YAML Files (*.yaml *.yml)"
        )
        
        if not file_path:
            return
            
        try:
            # 使用PyYAML解析文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                
            # 更新布局参数
            layout_info = data.get('layout_info', {})
            self.layout_size = layout_info.get('size', self.layout_size)
            self.k = layout_info.get('thermal_conductivity', self.k)
            self.mesh_resolution = layout_info.get('mesh_resolution', self.mesh_resolution)
            
            # 设置场景大小
            scene_width = self.layout_size[0] * self.scene_scale
            scene_height = self.layout_size[1] * self.scene_scale
            self.scene.setSceneRect(0, 0, scene_width, scene_height)
            
            # 清空当前场景
            self.scene.clear()
            
            # 根据components列表创建元件
            components = data.get('components', [])
            for comp in components:
                self._create_item_from_state(comp)
                
            self.status_bar.showMessage(f"Layout loaded from {file_path}")
            
            # 更新组件列表
            self.update_components_list()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")
    
    def _create_item_from_state(self, component_state: Dict):
        """根据组件状态创建图形项"""
        try:
            # 创建图形元件
            item = create_component_item(component_state)
            
            # 添加到场景
            self.scene.addItem(item)
            x, y = component_state['coords']
            # 如果坐标是米单位，需要转换为像素单位
            if x <= self.layout_size[0] and y <= self.layout_size[1]:  # 检查是否为米单位
                x *= self.scene_scale
                y *= self.scene_scale
            item.setPos(x, y)  # 设置位置
            
        except Exception as e:
            print(f"Failed to create item from state: {e}")
    
    def save_to_yaml(self):
        """保存布局到YAML文件"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Layout File", "", "YAML Files (*.yaml *.yml)"
        )
        
        if not file_path:
            return
            
        try:
            # 🔄 从数据管理器获取组件数据（新方式）
            print("[YAML保存] 从数据管理器获取组件数据")
            all_components = self.data_sync.get_all_components()
            
            # 转换为UI格式（YAML保存需要UI格式）
            components = []
            for comp_data in all_components:
                ui_comp_data = self._convert_manager_data_to_ui(comp_data)
                components.append(ui_comp_data)
            
            print(f"[YAML保存] 准备保存 {len(components)} 个组件")
                    
            # 构建数据结构
            data = {
                'layout_info': {
                    'size': self.layout_size,
                    'thermal_conductivity': self.k,
                    'mesh_resolution': self.mesh_resolution,
                    'validity': True,
                    'creation_time': "Generated from UI"
                },
                'components': components,
                'boundary_conditions': {
                    'Dirichlet': [],
                    'Neumann': []
                }
            }
            
            # 写入YAML文件
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, sort_keys=False, allow_unicode=True)
                
            self.status_bar.showMessage(f"Layout saved to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")
    
    def load_from_json(self):
        """从JSON文件加载组件布局"""
        if not NEW_FEATURES_AVAILABLE:
            QMessageBox.warning(self, "功能不可用", "JSON功能需要新模块支持，请检查安装")
            return
            
        file_path, _ = QFileDialog.getOpenFileName(
            self, "加载JSON组件文件", "", "JSON Files (*.json)"
        )
        
        if not file_path:
            return
            
        try:
            # 使用JSON处理器加载数据
            components_dg, metadata = JSONComponentHandler.load_components_from_json(file_path)
            
            # 转换为UI格式
            ui_components = DataFormatConverter.data_generator_to_ui(components_dg, self.scene_scale)
            
            # 更新布局参数（如果元数据中有）
            if "layout_info" in metadata:
                layout_info = metadata["layout_info"]
                if "size" in layout_info:
                    self.layout_size = tuple(layout_info["size"])
                if "thermal_conductivity" in layout_info:
                    self.k = layout_info["thermal_conductivity"]
                if "mesh_resolution" in layout_info:
                    self.mesh_resolution = tuple(layout_info["mesh_resolution"])
            
            # 设置场景大小
            scene_width = self.layout_size[0] * self.scene_scale
            scene_height = self.layout_size[1] * self.scene_scale
            self.scene.setSceneRect(0, 0, scene_width, scene_height)
            
            # 清空当前场景
            self.scene.clear()
            self._add_grid()  # 重新添加网格
            
            # 🔄 使用数据同步器处理JSON加载（新方式）
            print(f"[JSON加载] 通过数据管理器加载 {len(components_dg)} 个组件")
            self.data_sync.handle_json_load(components_dg)
            
            # 🔄 从数据管理器重新创建UI显示
            all_components = self.data_sync.get_all_components()
            for comp_data in all_components:
                # 将数据管理器格式转换为UI显示格式
                ui_comp_data = self._convert_manager_data_to_ui(comp_data)
                self._create_item_from_ui_state(ui_comp_data)
            
            self.status_bar.showMessage(f"从JSON文件加载了 {len(all_components)} 个组件: {file_path}")
            
            # UI更新会通过信号自动触发，但这里手动调用确保同步
            self.update_components_list()
            
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"无法加载JSON文件: {str(e)}")
    
    def save_to_json(self):
        """保存组件布局到JSON文件"""
        if not NEW_FEATURES_AVAILABLE:
            QMessageBox.warning(self, "功能不可用", "JSON功能需要新模块支持，请检查安装")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存JSON组件文件", "", "JSON Files (*.json)"
        )
        
        if not file_path:
            return
            
        try:
            # 🔄 从数据管理器获取组件数据（新方式）
            print("[JSON保存] 从数据管理器获取组件数据")
            dg_components = self.data_sync.get_components_for_calculation()
            
            if not dg_components:
                QMessageBox.warning(self, "警告", "没有组件可保存")
                return
            
            print(f"[JSON保存] 准备保存 {len(dg_components)} 个组件")
            
            # 构建元数据
            metadata = {
                "layout_domain": self.layout_size,
                "thermal_conductivity": self.k,
                "mesh_resolution": self.mesh_resolution,
                "creation_time": "Generated from UI",
                "total_components": len(dg_components)
            }
            
            # 保存为JSON文件
            JSONComponentHandler.save_components_to_json(
                components=dg_components,
                file_path=file_path,
                metadata=metadata,
                format_type="full_sample"
            )
            
            self.status_bar.showMessage(f"保存了 {len(dg_components)} 个组件到JSON文件: {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"无法保存JSON文件: {str(e)}")
    
    def run_thermal_simulation(self):
        """运行热仿真计算"""
        if not NEW_FEATURES_AVAILABLE or self.thermal_backend is None:
            QMessageBox.warning(self, "功能不可用", "热仿真功能需要新模块支持，请检查安装")
            return
            
        try:
            # 🔄 从数据管理器获取组件数据（新方式）
            print("[热仿真] 从数据管理器获取组件数据")
            dg_components = self.data_sync.get_components_for_calculation()
            
            if not dg_components:
                QMessageBox.warning(self, "警告", "没有组件可进行热仿真")
                return
            
            print(f"[热仿真] 获取到 {len(dg_components)} 个组件用于计算")
            
            # 🔧 添加详细的输入数据日志
            print(f"[热仿真] 输入组件数据:")
            for i, comp in enumerate(dg_components[:3]):  # 只显示前3个组件
                print(f"  组件{i}: center={comp.get('center')}, power={comp.get('power')}, type={comp.get('type')}")
            
            # 创建热仿真输入数据
            input_data = DataFormatConverter.create_thermal_simulation_input(
                components=dg_components,
                layout_domain=self.layout_size,
                boundary_temperature=298.0  # 默认室温
            )
            
            print(f"[热仿真] 格式化后的layout_domain: {input_data.get('layout_domain')}")
            
            self.status_bar.showMessage("正在进行热仿真计算...")
            
            # 计算温度场
            grid_shape = (256, 256)
            result = self.thermal_backend.compute_field(
                input_data=input_data,
                field_type=FieldType.TEMPERATURE,
                grid_shape=grid_shape
            )
            
            if result.is_valid():
                # 显示温度场
                self._display_thermal_result(result)
                
                self.status_bar.showMessage(f"热仿真完成，计算时间: {result.computation_time:.2f}秒")
            else:
                QMessageBox.critical(self, "计算失败", f"热仿真计算失败: {result.error_info}")
                self.status_bar.showMessage("热仿真计算失败")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"热仿真过程中发生错误: {str(e)}")
            self.status_bar.showMessage("热仿真计算失败")
    
    def _convert_manager_data_to_ui(self, comp_data: Dict) -> Dict:
        """将数据管理器格式转换为UI显示格式"""
        ui_data = {
            'id': comp_data.get('id'),
            'type': comp_data.get('type', comp_data.get('shape')),  # 兼容shape字段
            'power': comp_data.get('power', 0.0)
        }
        
        # 转换坐标（从米到像素）
        center = comp_data.get('center', [0, 0])
        ui_data['coords'] = (center[0] * self.scene_scale, center[1] * self.scene_scale)
        
        # 转换尺寸（根据类型处理）
        comp_type = ui_data['type']
        if comp_type == 'rect':
            width = comp_data.get('width', 0.01) * self.scene_scale
            height = comp_data.get('height', 0.01) * self.scene_scale
            ui_data['size'] = [width, height]
        elif comp_type == 'circle':
            radius = comp_data.get('radius', 0.005) * self.scene_scale
            ui_data['size'] = [radius * 2, radius * 2]  # 直径
            ui_data['radius'] = radius
        elif comp_type == 'capsule':
            length = comp_data.get('length', 0.02) * self.scene_scale
            width = comp_data.get('width', 0.01) * self.scene_scale
            ui_data['size'] = [length, width]
        
        return ui_data
    
    def _create_item_from_ui_state(self, ui_component_state: Dict):
        """根据UI组件状态创建图形项"""
        try:
            # 创建图形元件
            item = create_component_item(ui_component_state)
            
            # 添加到场景
            self.scene.addItem(item)
            x, y = ui_component_state['coords']
            item.setPos(x, y)  # UI坐标已经是像素单位
            
        except Exception as e:
            print(f"Failed to create item from UI state: {e}")
    
    def _display_thermal_result(self, thermal_result):
        """显示热仿真结果"""
        print(f"[_display_thermal_result] 开始显示热仿真结果")
        try:
            scene_width = self.layout_size[0] * self.scene_scale
            scene_height = self.layout_size[1] * self.scene_scale
            print(f"[_display_thermal_result] 场景尺寸: {scene_width}x{scene_height}")
            
            # 创建可视化配置
            config = VisualizationConfig(
                scene_width=scene_width,
                scene_height=scene_height,
                layout_domain=self.layout_size
            )
            config.style.show_colorbar = True
            config.style.title = f"温度场分布 ({thermal_result.metadata.get('min_temperature', 0):.1f}K - {thermal_result.metadata.get('max_temperature', 0):.1f}K)"
            
            # 生成可视化图像
            pixmap = FieldVisualizer.create_field_visualization(thermal_result, config)
            
            # 更新温度场背景
            self.update_temperature_background_image(thermal_result.field_data)
            
        except Exception as e:
            print(f"Failed to display thermal result: {e}")
            import traceback
            traceback.print_exc()
    
    def on_scene_updated(self, scene_data: List[Dict]):
        """当场景更新时，触发后台计算"""
        self.status_bar.showMessage("Computing...")
        # 使用合适的网格形状
        grid_shape = (256, 256)
        # 使用QTimer.singleShot来避免直接在信号槽中调用moveToThread
        QTimer.singleShot(0, lambda: self.worker.compute(scene_data, grid_shape))
    
    def on_computation_complete(self, result: np.ndarray):
        """当SDF计算完成时，更新状态栏并显示结果"""
        self.status_bar.showMessage("SDF Computation Complete")
        self.result_image = result
        self.update_sdf_background_image(result)
    
    def on_temperature_reconstruction_complete(self, result: np.ndarray):
        """当温度重构完成时，更新状态栏并显示结果"""
        self.status_bar.showMessage("温度场重构完成 - Temperature Reconstruction Complete")
        self.result_image = result
        # 使用新的温度场数据更新背景
        self.update_temperature_background_image(result)
    
    def update_sdf_background(self):
        """手动更新SDF背景"""
        # 🔄 从数据管理器获取组件数据（新方式）
        print("[SDF计算] 从数据管理器获取组件数据")
        components = self.data_sync.get_components_for_calculation()
        
        if not components:
            QMessageBox.information(self, "No Components", "Please add some components first!")
            return
        
        print(f"[SDF计算] 获取到 {len(components)} 个组件用于计算")
            
        self.status_bar.showMessage("Updating SDF...")
        # 计算SDF网格形状
        grid_shape = calculate_sdf_grid_shape(self.layout_size)
        QTimer.singleShot(0, lambda: self.worker.compute(components, grid_shape))
    
    def update_sdf_background_image(self, sdf_array: np.ndarray):
        """更新SDF背景图像"""
        try:
            # 获取场景尺寸
            scene_width = self.layout_size[0] * self.scene_scale
            scene_height = self.layout_size[1] * self.scene_scale
            
            # 使用新的统一图像集成接口
            from image_integration_interface import QtMatplotlibIntegration, ImageIntegrationConfig
            
            # 创建SDF图像
            pixmap = create_sdf_figure(sdf_array, scene_width, scene_height)
            
            # 验证图像尺寸
            if not QtMatplotlibIntegration.verify_image_dimensions(pixmap, scene_width, scene_height):
                print(f"Warning: SDF image size mismatch. Expected: {scene_width}x{scene_height}, "
                      f"Actual: {pixmap.width()}x{pixmap.height()}")
            
            # 使用统一接口添加图像到场景
            self.sdf_background_item = QtMatplotlibIntegration.add_image_to_scene(
                scene=self.scene,
                pixmap=pixmap,
                position=(0, 0),
                z_value=ImageIntegrationConfig.BACKGROUND_Z_VALUE,
                replace_existing=self.sdf_background_item
            )
            
            # 根据当前设置显示或隐藏
            self.sdf_background_item.setVisible(self.sdf_visible)
            
            self.status_bar.showMessage("SDF Updated")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to update SDF background: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_temperature_background_image(self, temp_array: np.ndarray):
        """更新温度场背景图像"""
        try:
            # 获取场景尺寸
            scene_width = self.layout_size[0] * self.scene_scale
            scene_height = self.layout_size[1] * self.scene_scale
            
            # 使用新的统一图像集成接口
            from image_integration_interface import QtMatplotlibIntegration, ImageIntegrationConfig
            
            # 创建温度场图像，使用泰森多边形可视化
            pixmap = create_temperature_figure(temp_array, scene_width, scene_height)
            
            # 验证图像尺寸
            if not QtMatplotlibIntegration.verify_image_dimensions(pixmap, scene_width, scene_height):
                print(f"Warning: Temperature image size mismatch. Expected: {scene_width}x{scene_height}, "
                      f"Actual: {pixmap.width()}x{pixmap.height()}")
            
            # 使用统一接口添加图像到场景
            self.sdf_background_item = QtMatplotlibIntegration.add_image_to_scene(
                scene=self.scene,
                pixmap=pixmap,
                position=(0, 0),
                z_value=ImageIntegrationConfig.BACKGROUND_Z_VALUE,
                replace_existing=self.sdf_background_item
            )
            
            # 自动显示温度场
            self.sdf_background_item.setVisible(True)
            self.sdf_visible = True
            # 🔧 修复checkbox访问错误
            if hasattr(self.sidebar, 'sdf_show_checkbox'):
                self.sidebar.sdf_show_checkbox.setChecked(True)
            
            # 关闭测点放置模式，切换到选择模式
            self.set_sensor_placement_mode(False)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to update temperature background: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # === 🆕 选择同步处理方法 ===
    
    def on_component_selected(self, component_id: str):
        """处理组件被选中事件"""
        print(f"[MainWindow] 组件被选中: {component_id}")
        
        # 1. 切换侧边栏到对应标签页
        try:
            self.sidebar.select_component_tab(component_id)
        except Exception as e:
            print(f"[MainWindow] 侧边栏切换失败: {e}")
        
        # 2. 高亮对应的图形元件
        try:
            self._highlight_component_in_scene(component_id)
        except Exception as e:
            print(f"[MainWindow] 图形高亮失败: {e}")
    
    def on_selection_cleared(self):
        """处理选择被清除事件"""
        print("[MainWindow] 选择被清除")
        
        # 清除所有图形元件的高亮
        try:
            self._clear_all_highlights()
        except Exception as e:
            print(f"[MainWindow] 清除高亮失败: {e}")
    
    def _highlight_component_in_scene(self, component_id: str):
        """在场景中高亮指定组件"""
        # 首先清除所有高亮
        self._clear_all_highlights()
        
        # 查找并高亮指定组件
        for item in self.scene.items():
            if hasattr(item, 'get_state') and hasattr(item, 'set_highlighted'):
                state = item.get_state()
                if state.get('id') == component_id:
                    item.set_highlighted(True)
                    print(f"[MainWindow] 高亮组件: {component_id}")
                    break
    
    def _clear_all_highlights(self):
        """清除场景中所有组件的高亮"""
        for item in self.scene.items():
            if hasattr(item, 'set_highlighted'):
                item.set_highlighted(False)
    
    def _remove_graphics_item_by_id(self, component_id: str):
        """🆕 通过组件ID从场景中移除对应的图像项"""
        try:
            for item in self.scene.items():
                if hasattr(item, 'get_state'):
                    state = item.get_state()
                    if state.get('id') == component_id:
                        print(f"[MainWindow] 从场景移除组件图像: {component_id}")
                        self.scene.removeItem(item)
                        # 强制刷新场景
                        self.scene.update()
                        for view in self.scene.views():
                            view.update()
                        break
        except Exception as e:
            print(f"[MainWindow] 移除图像项失败: {e}")
    
    def _register_image_compute_callbacks(self):
        """注册图像计算回调函数"""
        # SDF计算回调
        def compute_sdf(input_data=None):
            if input_data is None:
                # 获取当前组件数据
                input_data = {
                    'components': self.data_sync.get_components_for_calculation(),
                    'layout_size': (0.1, 0.1)
                }
            
            from sdf_backend import SDFBackend
            
            sdf_backend = SDFBackend()
            components = input_data['components']
            grid_shape = (50, 50)  # 默认网格大小
            
            try:
                # 🔧 传递正确的layout_size参数
                layout_size = input_data.get('layout_size', (0.1, 0.1))
                print(f"[SDF计算回调] 传递布局尺寸: {layout_size}")
                sdf_array = sdf_backend.compute(components, grid_shape, layout_size)
                
                # 🔧 恢复旧版本的尺寸适配逻辑
                # 计算场景尺寸
                scene_width = self.layout_size[0] * self.scene_scale
                scene_height = self.layout_size[1] * self.scene_scale
                
                # 使用专门的SDF图像创建函数（包含尺寸适配）
                from ui_utils import create_sdf_figure
                pixmap = create_sdf_figure(sdf_array, scene_width, scene_height)
                
                print(f"[SDF计算] 图像尺寸: {pixmap.width()}x{pixmap.height()}, 场景尺寸: {scene_width}x{scene_height}")
                
                return pixmap
            except Exception as e:
                print(f"[SDF计算] 失败: {e}")
                return None
        
        # 温度场计算回调
        def compute_temperature(input_data=None):
            if input_data is None:
                # 🔧 使用与旧版本相同的数据格式化方式
                print("[温度场计算回调] 使用DataFormatConverter进行数据格式化")
                dg_components = self.data_sync.get_components_for_calculation()
                
                # 使用旧版本的数据格式化器
                from data_bridge.format_converter import DataFormatConverter
                input_data = DataFormatConverter.create_thermal_simulation_input(
                    components=dg_components,
                    layout_domain=self.layout_size,  # 使用当前布局尺寸
                    boundary_temperature=298.0  # 默认室温
                )
                print(f"[温度场计算回调] 格式化后的数据: layout_domain={input_data.get('layout_domain')}, 组件数={len(input_data.get('components', []))}")
                
                # 🔧 添加详细的输入数据日志
                print(f"[温度场计算回调] 输入组件数据:")
                formatted_components = input_data.get('components', [])
                for i, comp in enumerate(formatted_components[:3]):  # 只显示前3个组件
                    print(f"  组件{i}: center={comp.get('center')}, power={comp.get('power')}, shape={comp.get('shape')}")
            
            if self.thermal_backend:
                from backends.base_backend import FieldType
                # 🔧 使用与旧版本相同的网格尺寸
                grid_shape = (256, 256)  # 与旧版本保持一致
                result = self.thermal_backend.compute_field(input_data, FieldType.TEMPERATURE, grid_shape)
                
                if result.is_valid():
                    # 🔧 使用与热仿真按钮完全相同的显示路径
                    print(f"[温度场计算] 计算成功，使用专业显示路径")
                    
                    # 直接调用热仿真的显示方法
                    self._display_thermal_result(result)
                    
                    # 为了与ImageManager兼容，仍然返回图像
                    # 但实际显示已经通过_display_thermal_result完成
                    if isinstance(result.field_data, np.ndarray):
                        # 计算场景尺寸
                        scene_width = self.layout_size[0] * self.scene_scale
                        scene_height = self.layout_size[1] * self.scene_scale
                        
                        # 使用专门的温度场图像创建函数
                        from ui_utils import create_temperature_figure
                        pixmap = create_temperature_figure(result.field_data, scene_width, scene_height)
                        
                        print(f"[温度场计算] 图像尺寸: {pixmap.width()}x{pixmap.height()}, 场景尺寸: {scene_width}x{scene_height}")
                        return pixmap
                    else:
                        return result.field_data  # 已经是QPixmap
                else:
                    print(f"[温度场计算] 失败: {result.error_info}")
                    return None
            else:
                print("[温度场计算] 热仿真后端未可用")
                return None
        
        # 泰森多边形计算回调
        def compute_voronoi(input_data=None):
            if input_data is None:
                input_data = {
                    'components': self.data_sync.get_components_for_calculation(),
                    'layout_size': (0.1, 0.1)
                }
            
            from backends.voronoi_backend import VoronoiBackend
            from backends.base_backend import FieldType
            
            voronoi_backend = VoronoiBackend()
            result = voronoi_backend.compute_field(input_data, FieldType.VORONOI)
            
            if result.is_valid():
                # 🔧 泰森多边形已经返回QPixmap，但可能需要尺寸检查
                pixmap = result.field_data
                if isinstance(pixmap, QPixmap):
                    scene_width = self.layout_size[0] * self.scene_scale
                    scene_height = self.layout_size[1] * self.scene_scale
                    print(f"[泰森多边形计算] 图像尺寸: {pixmap.width()}x{pixmap.height()}, 场景尺寸: {scene_width}x{scene_height}")
                
                return result.field_data  # QPixmap
            else:
                print(f"[泰森多边形计算] 失败: {result.error_info}")
                return None
        
        # 温度场预测计算回调
        def compute_pod_temperature(input_data=None):
            if input_data is None:
                # 获取组件数据并转换为POD API格式
                print("[温度场预测计算回调] 开始数据准备")
                dg_components = self.data_sync.get_components_for_calculation()
                
                if not dg_components:
                    print("[温度场预测计算回调] 错误: 没有组件数据")
                    return None
                
                input_data = {
                    'components': dg_components,
                    'layout_size': self.layout_size
                }
                print(f"[温度场预测计算回调] 准备计算 {len(dg_components)} 个组件的POD温度场")
            
            try:
                # 导入POD后端
                from backends.pod_temperature_backend import PODTemperatureBackend
                from backends.base_backend import FieldType
                
                # 创建POD后端实例
                pod_backend = PODTemperatureBackend()
                
                # 初始化后端
                if not pod_backend.initialize():
                    print("[温度场预测] 后端初始化失败")
                    return None
                
                # 温度场预测
                grid_shape = (256, 256)  # 与原始温度场保持一致
                result = pod_backend.compute_field(input_data, FieldType.TEMPERATURE, grid_shape)
                
                if result.is_valid():
                    print(f"[温度场预测] 计算成功")
                    print(f"  温度范围: [{result.metadata.get('min_temperature', 0):.2f}, {result.metadata.get('max_temperature', 0):.2f}]K")
                    
                    # 将结果转换为QPixmap
                    if isinstance(result.field_data, np.ndarray):
                        # 计算场景尺寸
                        scene_width = self.layout_size[0] * self.scene_scale
                        scene_height = self.layout_size[1] * self.scene_scale
                        
                        # 使用专门的温度场图像创建函数
                        from ui_utils import create_temperature_figure
                        pixmap = create_temperature_figure(result.field_data, scene_width, scene_height)
                        
                        print(f"[温度场预测] 图像尺寸: {pixmap.width()}x{pixmap.height()}, 场景尺寸: {scene_width}x{scene_height}")
                        return pixmap
                    else:
                        return result.field_data  # 已经是QPixmap
                else:
                    print(f"[温度场预测] 失败: {result.error_info}")
                    return None
                    
            except Exception as e:
                print(f"[温度场预测] 异常: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        # POD重构计算回调
        def compute_pod_reconstruction(input_data=None):
            # 收集所有传感器数据
            print("[POD重构计算回调] 开始收集传感器数据")
            sensors = []
            for item in self.scene.items():
                if hasattr(item, 'get_state') and item.get_state().get('type') == 'sensor':
                    sensors.append(item)
            
            if not sensors:
                print("[POD重构计算回调] 错误: 没有传感器数据")
                return None
            
            # 准备测点数据
            measurement_points = []
            temperature_values = []
            
            for sensor in sensors:
                state = sensor.get_state()
                temperature = state.get('temperature', 298.0)  # 默认室温
                
                # 获取像素坐标并转换为POD API需要的坐标系(0-255)
                pixel_coords = state['coords']
                
                # 转换坐标系：像素坐标 → POD网格坐标 (0-255)
                scene_width = self.layout_size[0] * self.scene_scale
                scene_height = self.layout_size[1] * self.scene_scale
                
                grid_x = (pixel_coords[0] / scene_width) * 255
                grid_y = (pixel_coords[1] / scene_height) * 255
                
                # 边界检查
                grid_x = max(0, min(255, grid_x))
                grid_y = max(0, min(255, grid_y))
                
                measurement_points.append((grid_x, grid_y))
                temperature_values.append(temperature)
            
            print(f"[POD重构计算回调] 收集到 {len(measurement_points)} 个测点")
            print(f"  温度范围: [{min(temperature_values):.2f}, {max(temperature_values):.2f}]K")
            
            try:
                # 导入POD后端
                from backends.pod_temperature_backend import PODTemperatureBackend
                
                # 创建POD后端实例
                pod_backend = PODTemperatureBackend()
                
                # 初始化后端
                if not pod_backend.initialize():
                    print("[POD重构] 后端初始化失败")
                    return None
                
                # 执行GA重构
                result = pod_backend.reconstruct_temperature_field(
                    measurement_points,
                    temperature_values
                )
                
                if result.is_valid():
                    print(f"[POD重构] GA重构成功")
                    print(f"  重构温度范围: [{result.metadata.get('min_temperature', 0):.2f}, {result.metadata.get('max_temperature', 0):.2f}]K")
                    print(f"  测点平均误差: {result.metadata.get('validation_metrics', {}).get('point_mae', 0):.4f}")
                    
                    # 将结果转换为QPixmap
                    if isinstance(result.field_data, np.ndarray):
                        # 计算场景尺寸
                        scene_width = self.layout_size[0] * self.scene_scale
                        scene_height = self.layout_size[1] * self.scene_scale
                        
                        # 使用专门的温度场图像创建函数
                        from ui_utils import create_temperature_figure
                        pixmap = create_temperature_figure(result.field_data, scene_width, scene_height)
                        
                        print(f"[POD重构] 图像尺寸: {pixmap.width()}x{pixmap.height()}, 场景尺寸: {scene_width}x{scene_height}")
                        return pixmap
                    else:
                        return result.field_data  # 已经是QPixmap
                else:
                    print(f"[POD重构] 失败: {result.error_info}")
                    return None
                    
            except Exception as e:
                print(f"[POD重构] 异常: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        # 注册回调
        self.image_manager.register_compute_callback('sdf', compute_sdf)
        self.image_manager.register_compute_callback('temperature', compute_temperature)
        self.image_manager.register_compute_callback('pod_temperature', compute_pod_temperature)
        self.image_manager.register_compute_callback('pod_reconstruction', compute_pod_reconstruction)
        self.image_manager.register_compute_callback('voronoi', compute_voronoi)

    
    def closeEvent(self, event):
        """关闭窗口时，退出工作线程和清理资源"""
        self.thread.quit()
        self.thread.wait()
        
        # 🆕 清理输出重定向
        if hasattr(self, 'output_manager'):
            self.output_manager.cleanup()
        
        event.accept()
    
    def sample_temperatures_from_pod_field(self, sensors):
        """从POD温度场为传感器采样温度值
        
        Args:
            sensors: 传感器列表
            
        Returns:
            int: 成功更新的传感器数量
        """
        # 获取图像管理器
        from image_manager import get_image_manager
        image_manager = get_image_manager()
        
        # 检查POD温度场是否存在
        if not image_manager.is_cached('pod_temperature'):
            raise RuntimeError("POD温度场未计算，请先计算POD温度场")
        
        # 获取POD温度场数据
        try:
            # 通过POD后端重新计算以获取原始数组数据
            from backends.pod_temperature_backend import PODTemperatureBackend
            
            pod_backend = PODTemperatureBackend()
            if not pod_backend.initialize():
                raise RuntimeError("POD后端初始化失败")
            
            # 获取当前组件数据
            dg_components = self.data_sync.get_components_for_calculation()
            if not dg_components:
                raise RuntimeError("没有组件数据")
            
            input_data = {
                'components': dg_components,
                'layout_size': self.layout_size
            }
            
            # 重新计算温度场获取数组数据
            grid_shape = (256, 256)
            result = pod_backend.compute_field(input_data, pod_backend.get_supported_field_types()[0], grid_shape)
            
            if not result.is_valid():
                raise RuntimeError(f"温度场计算失败: {result.error_info}")
            
            temp_field = result.field_data
            print(f"[温度采样] 获取到温度场数据: {temp_field.shape}")
            
        except Exception as e:
            raise RuntimeError(f"获取温度场数据失败: {str(e)}")
        
        # 为每个传感器采样温度
        success_count = 0
        scene_width = self.layout_size[0] * self.scene_scale
        scene_height = self.layout_size[1] * self.scene_scale
        
        for sensor in sensors:
            try:
                state = sensor.get_state()
                pixel_coords = state['coords']
                
                # 坐标转换：像素坐标 → 温度场网格坐标
                norm_x = pixel_coords[0] / scene_width
                norm_y = pixel_coords[1] / scene_height
                
                # 网格坐标 (0-255)
                grid_x = int(norm_x * 255)
                grid_y = int(norm_y * 255)
                
                # 边界检查
                grid_x = max(0, min(255, grid_x))
                grid_y = max(0, min(255, grid_y))
                
                # 从温度场采样
                sampled_temp = float(temp_field[grid_y, grid_x])
                
                # 更新传感器温度
                state['temperature'] = sampled_temp
                sensor.set_state(state)  # 使用正确的方法名
                
                print(f"[温度采样] 传感器({pixel_coords[0]:.0f},{pixel_coords[1]:.0f}) -> 网格({grid_x},{grid_y}) -> {sampled_temp:.2f}K")
                success_count += 1
                
            except Exception as e:
                print(f"[温度采样] 传感器采样失败: {e}")
                continue
        
        # 更新传感器显示
        self.sidebar.update_sensor_list()
        
        # 强制刷新场景以更新温度显示
        self.scene.update()
        for view in self.scene.views():
            view.update()
        
        print(f"[温度采样] 完成，成功更新 {success_count}/{len(sensors)} 个传感器")
        return success_count