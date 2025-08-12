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
from PyQt6.QtGui import QPainter, QColor, QPen, QAction
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

# 添加layout目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'layout'))


class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Satellite Component Visualization & Physics Field Prediction")
        self.setGeometry(100, 100, 1200, 800)
        
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
        self._create_status_bar()
        
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
    
    def _setup_scene(self):
        """设置场景参数"""
        scene_width = self.layout_size[0] * self.scene_scale
        scene_height = self.layout_size[1] * self.scene_scale
        self.scene.setSceneRect(0, 0, scene_width, scene_height)
        
        # 添加坐标网格
        self._add_grid()
    
    def _add_grid(self):
        """添加坐标网格"""
        width, height = self.layout_size
        scene_width = width * self.scene_scale
        scene_height = height * self.scene_scale
        grid_interval = GRID_INTERVAL_METERS * self.scene_scale
        
        # 绘制垂直线
        pen = QPen(QColor(*Colors.GRID_LINE), 1)
        for x in np.arange(0, scene_width + grid_interval, grid_interval):
            self.scene.addLine(x, 0, x, scene_height, pen)
            
        # 绘制水平线
        for y in np.arange(0, scene_height + grid_interval, grid_interval):
            self.scene.addLine(0, y, scene_width, y, pen)
            
        # 绘制边界
        border_pen = QPen(QColor(*Colors.BORDER_LINE), 2)
        self.scene.addRect(0, 0, scene_width, scene_height, border_pen)
    
    def _create_toolbar(self):
        """创建主工具栏"""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        # 文件操作按钮
        load_action = QAction(f"{Icons.LOAD_FILE} YAML", self)
        load_action.triggered.connect(self.load_from_yaml)
        toolbar.addAction(load_action)
        
        # 添加分隔符
        toolbar.addSeparator()
        
        # 保存按钮（移到工具栏右侧）
        save_action = QAction(f"{Icons.SAVE_FILE} YAML", self)
        save_action.triggered.connect(self.save_to_yaml)
        toolbar.addAction(save_action)
    
    def _create_sidebar(self):
        """创建侧边栏"""
        self.sidebar = SidebarPanel(self)
    
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
        self.sidebar.sdf_update_button.setVisible(checked)
        
        if self.sdf_background_item:
            self.sdf_background_item.setVisible(checked)
        
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
            # 收集元件数据
            components = []
            for item in self.scene.items():
                if hasattr(item, 'get_state'):
                    components.append(item.get_state())
                    
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
        # 收集当前场景中的组件数据
        components = []
        for item in self.scene.items():
            if hasattr(item, 'get_state'):
                # 转换到米单位用于计算
                state = convert_component_to_meters(item.get_state(), self.scene_scale)
                components.append(state)
        
        if not components:
            QMessageBox.information(self, "No Components", "Please add some components first!")
            return
            
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
            self.sidebar.sdf_show_checkbox.setChecked(True)
            
            # 关闭测点放置模式，切换到选择模式
            self.set_sensor_placement_mode(False)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to update temperature background: {str(e)}")
            import traceback
            traceback.print_exc()
    

    
    def closeEvent(self, event):
        """关闭窗口时，退出工作线程"""
        self.thread.quit()
        self.thread.wait()
        event.accept()