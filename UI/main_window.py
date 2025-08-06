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
from ui_utils import (convert_component_to_meters, create_matplotlib_figure,
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
        load_action = QAction(f"{Icons.LOAD_FILE} Load YAML", self)
        load_action.triggered.connect(self.load_from_yaml)
        toolbar.addAction(load_action)
        
        save_action = QAction(f"{Icons.SAVE_FILE} Save YAML", self)
        save_action.triggered.connect(self.save_to_yaml)
        toolbar.addAction(save_action)
        
        # 分隔符
        toolbar.addSeparator()
        
        # 删除选中项按钮
        delete_action = QAction(f"{Icons.DELETE} Delete Selected", self)
        delete_action.triggered.connect(self.delete_selected)
        toolbar.addAction(delete_action)
    
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
            self.set_draw_mode(mode)
        else:
            # 如果当前按钮被取消选中，进入None模式
            self.sidebar.rect_button.setChecked(False)
            self.sidebar.circle_button.setChecked(False)
            self.sidebar.capsule_button.setChecked(False)
            self.set_draw_mode(None)
    
    def set_draw_mode(self, mode):
        """设置绘制模式并更新状态"""
        # 设置场景的绘制模式
        self.scene.set_draw_mode(mode)
        
        # 更新状态栏显示当前模式
        if mode is None:
            self.status_bar.showMessage(f"Draw Mode: {Icons.NONE_MODE} None (Selection Mode)")
        else:
            mode_name = ComponentNames.DISPLAY_NAMES.get(mode, mode)
            mode_icon = Icons.DRAW_MODES.get(mode, '')
            self.status_bar.showMessage(f"Draw Mode: {mode_icon} {mode_name}")
    
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
    
    def delete_selected(self):
        """删除选中的组件"""
        items_to_delete = []
        for item in self.scene.selectedItems():
            if isinstance(item, (RectItem, CircleItem, CapsuleItem)):
                items_to_delete.append(item)
        
        # 批量删除并强制刷新场景
        for item in items_to_delete:
            self.scene.removeItem(item)
        
        # 强制场景更新以确保图形完全消失
        if items_to_delete:
            self.scene.update()
            # 确保视图也更新
            for view in self.scene.views():
                view.update()
            # 更新组件列表
            self.update_components_list()
    
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
        """当计算完成时，更新状态栏并显示结果"""
        self.status_bar.showMessage("Computation Complete")
        self.result_image = result
        self.update_sdf_background_image(result)
    
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
            
            # 创建matplotlib图像
            pixmap = create_matplotlib_figure(sdf_array, scene_width, scene_height)
            
            # 移除旧的SDF背景
            if self.sdf_background_item:
                self.scene.removeItem(self.sdf_background_item)
            
            # 创建新的背景图像项
            self.sdf_background_item = QGraphicsPixmapItem(pixmap)
            self.sdf_background_item.setPos(0, 0)
            self.sdf_background_item.setZValue(SDFConfig.Z_VALUE)  # 置于最底层
            self.scene.addItem(self.sdf_background_item)
            
            # 根据当前设置显示或隐藏
            self.sdf_background_item.setVisible(self.sdf_visible)
            
            self.status_bar.showMessage("SDF Updated")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to update SDF background: {str(e)}")
    
    def closeEvent(self, event):
        """关闭窗口时，退出工作线程"""
        self.thread.quit()
        self.thread.wait()
        event.accept()