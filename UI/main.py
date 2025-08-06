import sys
import os
import random
from typing import List, Dict, Optional, Tuple, Union
import yaml
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QGraphicsItem, QGraphicsRectItem, QGraphicsEllipseItem,
    QToolBar, QFileDialog, QMessageBox, QStatusBar, QGraphicsSceneMouseEvent,
    QStyleOptionGraphicsItem, QStyle, QDialog, QVBoxLayout, QLabel, QDialogButtonBox,
    QDockWidget, QPushButton, QButtonGroup, QWidget, QHBoxLayout, QFrame,
    QGraphicsPixmapItem
)
from PyQt6.QtGui import QPainter, QColor, QPen, QAction, QBrush, QFont, QFontDatabase, QPixmap, QImage
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, QObject, QThread, QTimer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')  # 使用Qt后端

# 导入后端计算模块接口
from interfaces import ComputationBackend
# 导入SDF后端实现
from sdf_backend import SDFBackend

# 添加layout目录到Python路径，以便导入Satellite2DLayout
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'layout'))
from layout import Satellite2DLayout


# 自定义图形元件
class RectItem(QGraphicsItem):
    def __init__(self, component_state: Dict):
        super().__init__()
        self.state = component_state
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        
        # 设置字体 - 使用系统默认字体
        self.font = QFont()
        self.font.setFamily("Arial, sans-serif")
        self.font.setPointSize(10)

    def boundingRect(self) -> QRectF:
        w, h = self.state['size']
        return QRectF(-w/2, -h/2, w, h)

    def paint(self, painter: QPainter, option, widget):
        w, h = self.state['size']
        rect = QRectF(-w/2, -h/2, w, h)  # 以中心为原点绘制
        
        # 设置画笔和画刷
        painter.setPen(QPen(QColor(0, 0, 255), 1))  # 蓝色边框
        if option.state & QStyle.StateFlag.State_Selected:
            painter.setBrush(QColor(100, 100, 255, 150))  # 选中时半透明蓝色填充
        else:
            painter.setBrush(QColor(173, 216, 230, 150))  # 默认淡蓝色填充
        painter.drawRect(rect)

        # 绘制功率标签
        painter.setFont(self.font)
        painter.setPen(QPen(QColor(0, 0, 0)))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"P={self.state['power']:.1f}W")

    def get_state(self) -> Dict:
        return self.state

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.scene():
            # 更新坐标
            new_pos = value
            self.state['coords'] = (new_pos.x(), new_pos.y())
            # 通知场景更新
            self.scene().item_position_changed.emit()
        return super().itemChange(change, value)


class CircleItem(QGraphicsItem):
    def __init__(self, component_state: Dict):
        super().__init__()
        self.state = component_state
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        
        # 设置字体 - 使用系统默认字体
        self.font = QFont()
        self.font.setFamily("Arial, sans-serif")
        self.font.setPointSize(10)

    def boundingRect(self) -> QRectF:
        r = self.state['size']
        return QRectF(-r, -r, 2*r, 2*r)

    def paint(self, painter: QPainter, option, widget):
        r = self.state['size']
        rect = QRectF(-r, -r, 2*r, 2*r)  # 以中心为原点绘制
        
        # 设置画笔和画刷
        painter.setPen(QPen(QColor(255, 0, 0), 1))  # 红色边框
        if option.state & QStyle.StateFlag.State_Selected:
            painter.setBrush(QColor(255, 100, 100, 150))  # 选中时半透明红色填充
        else:
            painter.setBrush(QColor(255, 182, 193, 150))  # 默认浅红色填充
        painter.drawEllipse(rect)

        # 绘制功率标签
        painter.setFont(self.font)
        painter.setPen(QPen(QColor(0, 0, 0)))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"P={self.state['power']:.1f}W")

    def get_state(self) -> Dict:
        return self.state

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.scene():
            # 更新坐标
            new_pos = value
            self.state['coords'] = (new_pos.x(), new_pos.y())
            # 通知场景更新
            self.scene().item_position_changed.emit()
        return super().itemChange(change, value)


class CapsuleItem(QGraphicsItem):
    def __init__(self, component_state: Dict):
        super().__init__()
        self.state = component_state
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        
        # 设置字体 - 使用系统默认字体
        self.font = QFont()
        self.font.setFamily("Arial, sans-serif")
        self.font.setPointSize(10)

    def boundingRect(self) -> QRectF:
        length, width = self.state['size']
        return QRectF(-length/2, -width/2, length, width)

    def paint(self, painter: QPainter, option, widget):
        length, width = self.state['size']
        # 以中心为原点绘制
        rect_length = length - width  # 矩形部分长度
        
        # 设置画笔和画刷
        painter.setPen(QPen(QColor(0, 128, 0), 1))  # 绿色边框
        if option.state & QStyle.StateFlag.State_Selected:
            painter.setBrush(QColor(100, 255, 100, 150))  # 选中时半透明绿色填充
        else:
            painter.setBrush(QColor(144, 238, 144, 150))  # 默认浅绿色填充
        
        # 绘制矩形部分
        rect = QRectF(-rect_length/2, -width/2, rect_length, width)
        painter.drawRect(rect)
        
        # 绘制左右半圆
        left_circle_rect = QRectF(-rect_length/2 - width/2, -width/2, width, width)
        right_circle_rect = QRectF(rect_length/2 - width/2, -width/2, width, width)
        painter.drawEllipse(left_circle_rect)
        painter.drawEllipse(right_circle_rect)

        # 绘制功率标签
        painter.setFont(self.font)
        painter.setPen(QPen(QColor(0, 0, 0)))
        painter.drawText(QRectF(-length/2, -width/2, length, width), 
                         Qt.AlignmentFlag.AlignCenter, f"P={self.state['power']:.1f}W")

    def get_state(self) -> Dict:
        return self.state

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.scene():
            # 更新坐标
            new_pos = value
            self.state['coords'] = (new_pos.x(), new_pos.y())
            # 通知场景更新
            self.scene().item_position_changed.emit()
        return super().itemChange(change, value)


# 自定义场景类，用于发送信号和支持拖拽绘制
class CustomGraphicsScene(QGraphicsScene):
    # 定义信号
    item_position_changed = pyqtSignal()
    scene_updated = pyqtSignal(list)  # 发送场景数据
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.item_position_changed.connect(self.on_item_moved)
        
        # 拖拽绘制相关属性
        self.drawing = False
        self.start_point = QPointF()
        self.temp_item = None
        self.current_draw_mode = None  # 当前绘制模式：None, 'rect', 'circle', 'capsule'
        self.component_id_counter = 0
        
    def set_draw_mode(self, mode: str):
        """设置当前绘制模式"""
        self.current_draw_mode = mode
        
    def on_item_moved(self):
        # 当元件移动时，收集场景数据并发送信号
        self.collect_and_emit_scene_data()
        
    def collect_and_emit_scene_data(self):
        components = []
        for item in self.items():
            if hasattr(item, 'get_state'):
                components.append(item.get_state())
        self.scene_updated.emit(components)
        
    def mousePressEvent(self, event):
        # 检查是否点击在现有组件上
        clicked_item = self.itemAt(event.scenePos(), self.views()[0].transform())
        
        # 如果点击的是现有组件，允许选择和移动
        if clicked_item and isinstance(clicked_item, (RectItem, CircleItem, CapsuleItem)):
            super().mousePressEvent(event)
            return
            
        # 如果是左键点击空白区域且有选中绘制模式，开始绘制
        if event.button() == Qt.MouseButton.LeftButton and self.current_draw_mode is not None:
            self.start_point = event.scenePos()
            self.drawing = True
            self.create_preview_item()
        
        super().mousePressEvent(event)
        
    def create_preview_item(self):
        """创建预览项目"""
        if self.current_draw_mode is None:
            return
            
        pen = QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.DashLine)
        
        if self.current_draw_mode == 'rect':
            self.temp_item = QGraphicsRectItem(QRectF(self.start_point, self.start_point))
            self.temp_item.setPen(pen)
        elif self.current_draw_mode == 'circle':
            self.temp_item = QGraphicsEllipseItem(QRectF(self.start_point, self.start_point))
            self.temp_item.setPen(pen)
        elif self.current_draw_mode == 'capsule':
            # 胶囊形用矩形预览
            self.temp_item = QGraphicsRectItem(QRectF(self.start_point, self.start_point))
            pen.setColor(Qt.GlobalColor.green)
            self.temp_item.setPen(pen)
            
        if self.temp_item:
            self.addItem(self.temp_item)
    
    def mouseMoveEvent(self, event):
        if self.drawing and self.temp_item:
            current_point = event.scenePos()
            
            if self.current_draw_mode in ['rect', 'capsule']:
                # 矩形和胶囊形使用矩形预览
                rect = QRectF(self.start_point, current_point).normalized()
                self.temp_item.setRect(rect)
            elif self.current_draw_mode == 'circle':
                # 圆形预览 - 以拖拽矩形的中心为圆心
                rect = QRectF(self.start_point, current_point).normalized()
                # 使用较短边作为直径，保持圆形
                size = min(rect.width(), rect.height())
                # 计算圆形应该放置的位置（以拖拽矩形中心为圆心）
                center_x = rect.center().x()
                center_y = rect.center().y()
                radius = size / 2
                circle_rect = QRectF(center_x - radius, center_y - radius, size, size)
                self.temp_item.setRect(circle_rect)
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.drawing = False
            
            if self.temp_item:
                # 获取最终区域
                end_point = event.scenePos()
                final_rect = QRectF(self.start_point, end_point).normalized()
                
                # 移除预览项目
                self.removeItem(self.temp_item)
                self.temp_item = None
                
                # 检查是否有效（避免只是点击）并且在边界内
                if final_rect.width() > 10 and final_rect.height() > 10:
                    # 检查绘制区域是否完全在场景边界内
                    scene_rect = self.sceneRect()
                    if scene_rect.contains(final_rect):
                        self.create_permanent_item(final_rect)
                    else:
                        # 绘制区域超出边界，显示提示
                        from PyQt6.QtWidgets import QMessageBox
                        # 在主线程中显示消息框
                        if self.views():
                            view = self.views()[0]
                            if hasattr(view, 'window'):
                                QMessageBox.warning(view.window(), "Invalid Drawing", 
                                                   "Drawing area must be completely within the canvas bounds!")
                        print("Warning: Drawing area exceeds canvas bounds")
        
        super().mouseReleaseEvent(event)
    
    def create_permanent_item(self, rect: QRectF):
        """创建永久组件"""
        # 如果没有选择绘制模式，不创建组件
        if self.current_draw_mode is None:
            return
            
        # 计算中心点和尺寸
        center_x = rect.center().x()
        center_y = rect.center().y()
        
        # 生成随机功率
        power = random.uniform(5.0, 50.0)
        
        if self.current_draw_mode == 'rect':
            size = (rect.width(), rect.height())
            component_state = {
                'id': self.component_id_counter,
                'type': 'rect',
                'size': size,
                'coords': (center_x, center_y),
                'power': power,
                'center': (center_x, center_y)
            }
            item = RectItem(component_state)
            
        elif self.current_draw_mode == 'circle':
            # 使用较短边作为直径
            radius = min(rect.width(), rect.height()) / 2
            component_state = {
                'id': self.component_id_counter,
                'type': 'circle',
                'size': radius,
                'coords': (center_x, center_y),
                'power': power,
                'center': (center_x, center_y)
            }
            item = CircleItem(component_state)
            
        elif self.current_draw_mode == 'capsule':
            size = (rect.width(), rect.height())
            component_state = {
                'id': self.component_id_counter,
                'type': 'capsule',
                'size': size,
                'coords': (center_x, center_y),
                'power': power,
                'center': (center_x, center_y)
            }
            item = CapsuleItem(component_state)
        
        # 添加到场景
        self.addItem(item)
        item.setPos(center_x, center_y)
        
        # 增加ID计数器
        self.component_id_counter += 1
        
        # 更新组件列表
        if self.views():
            view = self.views()[0]
            if hasattr(view, 'window') and hasattr(view.window(), 'update_components_list'):
                view.window().update_components_list()


# 工作线程类
class Worker(QObject):
    # 定义信号
    computation_complete = pyqtSignal(np.ndarray)  # 发送计算结果
    
    def __init__(self):
        super().__init__()
        # 初始化后端计算模块
        self.backend: Optional[ComputationBackend] = SDFBackend()  # 默认使用SDF后端
        
    def set_backend(self, backend: ComputationBackend):
        self.backend = backend
        
    def compute(self, scene_data: List[Dict], grid_shape: Tuple[int, int]):
        if self.backend is not None:
            result = self.backend.compute(scene_data, grid_shape)
            self.computation_complete.emit(result)


# 主窗口类
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Satellite Component Visualization & Physics Field Prediction")
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化布局参数
        self.layout_size = (1.0, 1.0)  # 1m x 1m
        self.k = 100.0  # 热导率
        self.mesh_resolution = 50  # 网格分辨率
        
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
        
        # 设置场景大小 - 转换为像素单位便于显示
        self.scene_scale = 500  # 每米500像素
        scene_width = self.layout_size[0] * self.scene_scale
        scene_height = self.layout_size[1] * self.scene_scale
        self.scene.setSceneRect(0, 0, scene_width, scene_height)
        
        # 添加坐标网格
        self.add_grid()
        
        # 创建工具栏
        self.create_toolbar()
        
        # 创建侧边栏
        self.create_sidebar()
        
        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # 连接场景更新信号到处理函数
        self.scene.scene_updated.connect(self.on_scene_updated)
        
        # 初始化工作线程
        self.thread = QThread()
        self.worker = Worker()
        self.worker.moveToThread(self.thread)
        self.thread.start()
        
        # 连接工作线程信号
        self.worker.computation_complete.connect(self.on_computation_complete)
        
        # 存储计算结果图像
        self.result_image: Optional[np.ndarray] = None
        
        # 设置初始绘制模式为None（选择模式）
        self.set_draw_mode(None)
        
        # 初始化组件列表
        self.update_components_list()
        
    def add_grid(self):
        """添加坐标网格"""
        width, height = self.layout_size
        scene_width = width * self.scene_scale
        scene_height = height * self.scene_scale
        grid_interval = 0.1 * self.scene_scale  # 10cm间隔，转换为像素
        
        # 绘制垂直线
        pen = QPen(QColor(200, 200, 200), 1)  # 增加线条粗细
        for x in np.arange(0, scene_width + grid_interval, grid_interval):
            self.scene.addLine(x, 0, x, scene_height, pen)
            
        # 绘制水平线
        for y in np.arange(0, scene_height + grid_interval, grid_interval):
            self.scene.addLine(0, y, scene_width, y, pen)
            
        # 绘制边界
        border_pen = QPen(QColor(0, 0, 0), 2)  # 增加边界线粗细
        self.scene.addRect(0, 0, scene_width, scene_height, border_pen)
        
    def create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        # 文件操作按钮
        load_action = QAction("📁 Load YAML", self)
        load_action.triggered.connect(self.load_from_yaml)
        toolbar.addAction(load_action)
        
        save_action = QAction("💾 Save YAML", self)
        save_action.triggered.connect(self.save_to_yaml)
        toolbar.addAction(save_action)
        
        # 分隔符
        toolbar.addSeparator()
        
        # 删除选中项按钮
        delete_action = QAction("🗑️ Delete Selected", self)
        delete_action.triggered.connect(self.delete_selected)
        toolbar.addAction(delete_action)
        
    def create_sidebar(self):
        """创建侧边栏面板"""
        # 创建dock widget
        self.sidebar_dock = QDockWidget("Drawing Tools", self)
        self.sidebar_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        
        # 创建侧边栏内容widget
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        
        # 绘制模式标题
        mode_label = QLabel("Drawing Mode")
        mode_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mode_label.setStyleSheet("font-weight: bold; font-size: 14px; margin: 10px;")
        sidebar_layout.addWidget(mode_label)
        
        # 创建按钮组，支持取消选中
        self.mode_button_group = QButtonGroup()
        self.mode_button_group.setExclusive(False)  # 允许取消选中
        
        # 矩形按钮
        self.rect_button = QPushButton()
        self.rect_button.setText("Rectangle")
        self.rect_button.setCheckable(True)
        self.rect_button.clicked.connect(lambda: self.toggle_draw_mode('rect'))
        self.rect_button.setStyleSheet("""
            QPushButton {
                padding: 15px;
                font-size: 12px;
                border: 2px solid #3498db;
                border-radius: 8px;
                background-color: #ecf0f1;
                margin: 5px;
            }
            QPushButton:checked {
                background-color: #3498db;
                color: white;
            }
            QPushButton:hover {
                background-color: #bdc3c7;
            }
        """)
        self.mode_button_group.addButton(self.rect_button)
        sidebar_layout.addWidget(self.rect_button)
        
        # 圆形按钮
        self.circle_button = QPushButton()
        self.circle_button.setText("Circle")
        self.circle_button.setCheckable(True)
        self.circle_button.clicked.connect(lambda: self.toggle_draw_mode('circle'))
        self.circle_button.setStyleSheet("""
            QPushButton {
                padding: 15px;
                font-size: 12px;
                border: 2px solid #e74c3c;
                border-radius: 8px;
                background-color: #ecf0f1;
                margin: 5px;
            }
            QPushButton:checked {
                background-color: #e74c3c;
                color: white;
            }
            QPushButton:hover {
                background-color: #bdc3c7;
            }
        """)
        self.mode_button_group.addButton(self.circle_button)
        sidebar_layout.addWidget(self.circle_button)
        
        # 胶囊按钮
        self.capsule_button = QPushButton()
        self.capsule_button.setText("Capsule")
        self.capsule_button.setCheckable(True)
        self.capsule_button.clicked.connect(lambda: self.toggle_draw_mode('capsule'))
        self.capsule_button.setStyleSheet("""
            QPushButton {
                padding: 15px;
                font-size: 12px;
                border: 2px solid #27ae60;
                border-radius: 8px;
                background-color: #ecf0f1;
                margin: 5px;
            }
            QPushButton:checked {
                background-color: #27ae60;
                color: white;
            }
            QPushButton:hover {
                background-color: #bdc3c7;
            }
        """)
        self.mode_button_group.addButton(self.capsule_button)
        sidebar_layout.addWidget(self.capsule_button)
        
        # 添加分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("margin: 15px 5px;")
        sidebar_layout.addWidget(separator)
        
        # SDF控制区域
        sdf_label = QLabel("SDF Visualization")
        sdf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sdf_label.setStyleSheet("font-weight: bold; font-size: 14px; margin: 10px;")
        sidebar_layout.addWidget(sdf_label)
        
        # SDF显示开关
        from PyQt6.QtWidgets import QCheckBox
        self.sdf_show_checkbox = QCheckBox("Show SDF Background")
        self.sdf_show_checkbox.setStyleSheet("""
            QCheckBox {
                font-size: 11px;
                margin: 5px;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
        """)
        self.sdf_show_checkbox.toggled.connect(self.on_sdf_show_toggled)
        sidebar_layout.addWidget(self.sdf_show_checkbox)
        
        # SDF更新按钮（初始隐藏）
        self.sdf_update_button = QPushButton("🔄 Update SDF")
        self.sdf_update_button.setVisible(False)
        self.sdf_update_button.clicked.connect(self.update_sdf_background)
        self.sdf_update_button.setStyleSheet("""
            QPushButton {
                padding: 8px;
                font-size: 11px;
                border: 2px solid #9b59b6;
                border-radius: 6px;
                background-color: #ecf0f1;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #9b59b6;
                color: white;
            }
        """)
        sidebar_layout.addWidget(self.sdf_update_button)
        
        # 添加另一个分隔线
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.HLine)
        separator2.setFrameShadow(QFrame.Shadow.Sunken)
        separator2.setStyleSheet("margin: 15px 5px;")
        sidebar_layout.addWidget(separator2)
        
        # 组件列表区域
        components_label = QLabel("Components")
        components_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        components_label.setStyleSheet("font-weight: bold; font-size: 14px; margin: 10px;")
        sidebar_layout.addWidget(components_label)
        
        # 创建滚动区域用于组件列表
        from PyQt6.QtWidgets import QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(200)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                background-color: #ffffff;
                margin: 5px;
            }
        """)
        
        # 组件列表容器
        self.components_widget = QWidget()
        self.components_layout = QVBoxLayout(self.components_widget)
        self.components_layout.setContentsMargins(5, 5, 5, 5)
        self.components_layout.setSpacing(5)
        
        # 添加"无组件"标签
        self.no_components_label = QLabel("No components added yet")
        self.no_components_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_components_label.setStyleSheet("color: #7f8c8d; font-style: italic; padding: 20px;")
        self.components_layout.addWidget(self.no_components_label)
        
        self.components_layout.addStretch()
        scroll_area.setWidget(self.components_widget)
        sidebar_layout.addWidget(scroll_area)
        
        # 添加弹性空间
        sidebar_layout.addStretch()
        
        # 设置侧边栏内容
        self.sidebar_dock.setWidget(sidebar_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.sidebar_dock)
    
    def toggle_draw_mode(self, mode: str):
        """切换绘制模式（支持取消选中）"""
        # 获取当前按钮状态
        current_button = getattr(self, f'{mode}_button')
        
        if current_button.isChecked():
            # 如果当前按钮被选中，激活该模式
            # 先取消其他按钮
            self.rect_button.setChecked(mode == 'rect')
            self.circle_button.setChecked(mode == 'circle')
            self.capsule_button.setChecked(mode == 'capsule')
            self.set_draw_mode(mode)
        else:
            # 如果当前按钮被取消选中，进入None模式
            self.rect_button.setChecked(False)
            self.circle_button.setChecked(False)
            self.capsule_button.setChecked(False)
            self.set_draw_mode(None)
    
    def set_draw_mode(self, mode):
        """设置绘制模式并更新状态"""
        # 设置场景的绘制模式
        self.scene.set_draw_mode(mode)
        
        # 更新状态栏显示当前模式
        if mode is None:
            self.status_bar.showMessage("Draw Mode: ❌ None (Selection Mode)")
        else:
            mode_names = {'rect': 'Rectangle', 'circle': 'Circle', 'capsule': 'Capsule'}
            mode_icons = {'rect': '🔲', 'circle': '⭕', 'capsule': '🏷️'}
            self.status_bar.showMessage(f"Draw Mode: {mode_icons.get(mode, '')} {mode_names.get(mode, mode)}")
    
    def on_sdf_show_toggled(self, checked: bool):
        """SDF显示开关回调"""
        self.sdf_visible = checked
        self.sdf_update_button.setVisible(checked)
        
        if self.sdf_background_item:
            self.sdf_background_item.setVisible(checked)
        
        if checked:
            self.status_bar.showMessage("SDF display enabled")
        else:
            self.status_bar.showMessage("SDF display disabled")
    
    def update_components_list(self):
        """更新组件列表显示"""
        # 清除现有组件项
        for i in reversed(range(self.components_layout.count())):
            child = self.components_layout.itemAt(i).widget()
            if child and child != self.no_components_label:
                child.setParent(None)
        
        # 获取所有组件
        components = []
        for item in self.scene.items():
            if hasattr(item, 'get_state'):
                components.append(item)
        
        if not components:
            # 显示无组件标签
            self.no_components_label.setVisible(True)
        else:
            # 隐藏无组件标签
            self.no_components_label.setVisible(False)
            
            # 为每个组件创建编辑项
            for i, component_item in enumerate(components):
                self.create_component_editor(component_item, i)
        
        # 确保stretch在最后
        self.components_layout.addStretch()
    
    def create_component_editor(self, component_item, index):
        """为组件创建编辑器"""
        from PyQt6.QtWidgets import QLineEdit, QDoubleSpinBox
        
        state = component_item.get_state()
        
        # 创建组件编辑容器
        editor_frame = QFrame()
        editor_frame.setFrameStyle(QFrame.Shape.Box)
        editor_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                background-color: #f8f9fa;
                margin: 2px;
                padding: 5px;
            }
        """)
        
        editor_layout = QVBoxLayout(editor_frame)
        editor_layout.setContentsMargins(8, 5, 8, 5)
        editor_layout.setSpacing(3)
        
        # 组件标题
        title_label = QLabel(f"{state['type'].title()} #{index + 1}")
        title_label.setStyleSheet("font-weight: bold; font-size: 11px; color: #2c3e50;")
        editor_layout.addWidget(title_label)
        
        # 位置信息（只读）
        coords = state['coords']
        pos_label = QLabel(f"Position: ({coords[0]:.2f}, {coords[1]:.2f})m")
        pos_label.setStyleSheet("font-size: 10px; color: #7f8c8d;")
        editor_layout.addWidget(pos_label)
        
        # 尺寸信息（只读）
        if state['type'] == 'circle':
            size_text = f"Radius: {state['size']:.2f}m"
        else:
            size = state['size']
            size_text = f"Size: {size[0]:.2f}×{size[1]:.2f}m"
        size_label = QLabel(size_text)
        size_label.setStyleSheet("font-size: 10px; color: #7f8c8d;")
        editor_layout.addWidget(size_label)
        
        # 功率编辑器
        power_layout = QHBoxLayout()
        power_layout.setContentsMargins(0, 0, 0, 0)
        
        power_label = QLabel("Power:")
        power_label.setStyleSheet("font-size: 10px; font-weight: bold;")
        power_layout.addWidget(power_label)
        
        power_spinbox = QDoubleSpinBox()
        power_spinbox.setRange(0.0, 1000.0)
        power_spinbox.setSingleStep(0.1)
        power_spinbox.setDecimals(1)
        power_spinbox.setSuffix(" W")
        power_spinbox.setValue(state['power'])
        power_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                font-size: 10px;
                padding: 2px;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
            }
        """)
        
        # 连接信号以实时更新
        power_spinbox.valueChanged.connect(
            lambda value, item=component_item: self.update_component_power(item, value)
        )
        
        power_layout.addWidget(power_spinbox)
        power_layout.addStretch()
        
        editor_layout.addLayout(power_layout)
        
        # 添加到列表布局中（在stretch之前）
        self.components_layout.insertWidget(self.components_layout.count() - 1, editor_frame)
    
    def update_component_power(self, component_item, new_power):
        """更新组件功率值"""
        if hasattr(component_item, 'state'):
            component_item.state['power'] = new_power
            # 触发重绘以更新显示的功率值
            component_item.update()
            self.status_bar.showMessage(f"Updated power to {new_power:.1f}W")
        
    def delete_selected(self):
        # 删除选中的元件
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
        # 弹出文件选择对话框
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
            
            # 设置场景大小 - 转换为像素单位
            scene_width = self.layout_size[0] * self.scene_scale
            scene_height = self.layout_size[1] * self.scene_scale
            self.scene.setSceneRect(0, 0, scene_width, scene_height)
            
            # 清空当前场景
            self.scene.clear()
            
            # 根据components列表创建元件
            components = data.get('components', [])
            for comp in components:
                self.create_item_from_state(comp)
                
            self.status_bar.showMessage(f"Layout loaded from {file_path}")
            
            # 更新组件列表
            self.update_components_list()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")
            
    def create_item_from_state(self, component_state: Dict):
        component_type = component_state['type']
        
        # 创建图形元件
        if component_type == 'rect':
            item = RectItem(component_state)
        elif component_type == 'circle':
            item = CircleItem(component_state)
        elif component_type == 'capsule':
            item = CapsuleItem(component_state)
        else:
            return  # 不支持的类型
            
        # 添加到场景
        self.scene.addItem(item)
        x, y = component_state['coords']
        # 如果坐标是米单位，需要转换为像素单位
        if x <= self.layout_size[0] and y <= self.layout_size[1]:  # 检查是否为米单位
            x *= self.scene_scale
            y *= self.scene_scale
        item.setPos(x, y)  # 设置位置
        # 确保元件在场景边界内
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        
    def save_to_yaml(self):
        # 弹出文件保存对话框
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
                    
            # 构建与Satellite2DLayout.to_yaml()一致的数据结构
            data = {
                'layout_info': {
                    'size': self.layout_size,
                    'thermal_conductivity': self.k,
                    'mesh_resolution': self.mesh_resolution,
                    'validity': True,  # 默认为有效
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
        # 当场景更新时，触发后台计算
        self.status_bar.showMessage("Computing...")
        # 在工作线程中执行计算
        # 这里我们使用一个简单的网格形状 (256, 256)
        grid_shape = (256, 256)
        # 使用QTimer.singleShot来避免直接在信号槽中调用moveToThread
        QTimer.singleShot(0, lambda: self.worker.compute(scene_data, grid_shape))
        

        
    def on_computation_complete(self, result: np.ndarray):
        # 当计算完成时，更新状态栏并显示结果
        self.status_bar.showMessage("Computation Complete")
        self.result_image = result
        self.update_sdf_background_image(result)
    
    def update_sdf_background(self):
        """手动更新SDF背景"""
        # 收集当前场景中的组件数据
        components = []
        for item in self.scene.items():
            if hasattr(item, 'get_state'):
                # 转换坐标回米单位用于计算
                state = item.get_state().copy()
                coords = state['coords']
                state['coords'] = (coords[0] / self.scene_scale, coords[1] / self.scene_scale)
                # 转换尺寸回米单位
                if state['type'] == 'rect':
                    size = state['size']
                    state['size'] = (size[0] / self.scene_scale, size[1] / self.scene_scale)
                elif state['type'] == 'circle':
                    state['size'] = state['size'] / self.scene_scale
                elif state['type'] == 'capsule':
                    size = state['size']
                    state['size'] = (size[0] / self.scene_scale, size[1] / self.scene_scale)
                components.append(state)
        
        if not components:
            QMessageBox.information(self, "No Components", "Please add some components first!")
            return
            
        self.status_bar.showMessage("Updating SDF...")
        # 计算与Qt网格匹配的SDF分辨率
        # Qt网格：0.1米间隔，场景1×1米 = 10×10网格
        # 为了高质量显示，使用5倍分辨率：50×50
        grid_resolution = int(self.layout_size[0] / 0.1) * 5  # 10 * 5 = 50
        grid_shape = (grid_resolution, grid_resolution)  # (50, 50)
        QTimer.singleShot(0, lambda: self.worker.compute(components, grid_shape))
    
    def update_sdf_background_image(self, sdf_array: np.ndarray):
        """更新SDF背景图像"""
        try:
            # 获取场景尺寸
            scene_width = self.layout_size[0] * self.scene_scale
            scene_height = self.layout_size[1] * self.scene_scale
            
            # 创建matplotlib图形，精确匹配场景像素尺寸
            dpi = 100
            fig_width = scene_width / dpi
            fig_height = scene_height / dpi
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
            fig.patch.set_facecolor('none')  # 透明背景
            ax.set_facecolor('none')
            
            # 创建SDF图像 - 像素级精确对齐
            im = ax.imshow(sdf_array, cmap='coolwarm', origin='upper',
                          extent=[0, scene_width, 0, scene_height],
                          alpha=0.6, interpolation='nearest')  # 使用最近邻插值避免模糊
            ax.axis('off')  # 隐藏坐标轴
            
            # 设置精确的边距和布局
            ax.set_xlim(0, scene_width)
            ax.set_ylim(0, scene_height)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
            
            # 保存为图像
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                       transparent=True, pad_inches=0)
            buf.seek(0)
            
            # 转换为QPixmap
            qimg = QImage()
            qimg.loadFromData(buf.getvalue())
            pixmap = QPixmap.fromImage(qimg)
            
            # 移除旧的SDF背景
            if self.sdf_background_item:
                self.scene.removeItem(self.sdf_background_item)
            
            # 创建新的背景图像项
            self.sdf_background_item = QGraphicsPixmapItem(pixmap)
            self.sdf_background_item.setPos(0, 0)
            self.sdf_background_item.setZValue(-1)  # 置于最底层
            self.scene.addItem(self.sdf_background_item)
            
            # 根据当前设置显示或隐藏
            self.sdf_background_item.setVisible(self.sdf_visible)
            
            # 清理matplotlib资源
            plt.close(fig)
            
            self.status_bar.showMessage("SDF Updated")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to update SDF background: {str(e)}")
    

    

        
    def closeEvent(self, event):
        # 关闭窗口时，退出工作线程
        self.thread.quit()
        self.thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置应用程序字体
    font = QFont()
    font.setFamily("Arial, sans-serif")
    font.setPointSize(10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())