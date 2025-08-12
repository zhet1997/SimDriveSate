"""
自定义图形场景
处理拖拽绘制、鼠标事件和组件创建
"""

from typing import Optional
from PyQt6.QtWidgets import (QGraphicsScene, QGraphicsRectItem, QGraphicsEllipseItem, 
                             QMessageBox)
from PyQt6.QtGui import QPen
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal
from graphics_items import RectItem, CircleItem, CapsuleItem, RadiatorItem, SensorPointItem, create_component_item
from ui_constants import Colors, MIN_COMPONENT_SIZE
from ui_utils import (generate_random_power, create_circle_rect_from_drag, 
                      validate_drawing_bounds)


class CustomGraphicsScene(QGraphicsScene):
    """自定义图形场景，支持拖拽绘制和组件管理"""
    
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
        self.current_draw_mode = None  # 当前绘制模式：None, 'rect', 'circle', 'capsule', 'radiator'
        self.component_id_counter = 0
        
        # 测点放置模式
        self.sensor_placement_mode = False
        
    def set_draw_mode(self, mode: Optional[str]):
        """设置当前绘制模式"""
        self.current_draw_mode = mode
        
    def set_sensor_placement_mode(self, enabled: bool):
        """设置测点放置模式"""
        self.sensor_placement_mode = enabled
        
    def on_item_moved(self):
        """当元件移动时，收集场景数据并发送信号"""
        self.collect_and_emit_scene_data()
        
    def collect_and_emit_scene_data(self):
        """收集并发送场景数据"""
        components = []
        for item in self.items():
            if hasattr(item, 'get_state'):
                components.append(item.get_state())
        self.scene_updated.emit(components)
        
    def mousePressEvent(self, event):
        """处理鼠标按下事件"""
        # 检查是否点击在现有组件上
        clicked_item = self.itemAt(event.scenePos(), self.views()[0].transform())
        
        # 如果点击的是现有组件，允许选择和移动
        if clicked_item and isinstance(clicked_item, (RectItem, CircleItem, CapsuleItem, RadiatorItem, SensorPointItem)):
            super().mousePressEvent(event)
            return
        
        # 如果是测点放置模式
        if event.button() == Qt.MouseButton.LeftButton and self.sensor_placement_mode:
            self.place_sensor(event.scenePos())
            return
            
        # 如果是左键点击空白区域且有选中绘制模式，开始绘制
        if event.button() == Qt.MouseButton.LeftButton and self.current_draw_mode is not None:
            self.start_point = event.scenePos()
            self.drawing = True
            
            # 散热器模式需要检查边界约束
            if self.current_draw_mode == 'radiator':
                if self.is_on_boundary(event.scenePos()):
                    self.create_preview_item()
                else:
                    # 不在边界上，显示提示
                    if self.views():
                        view = self.views()[0]
                        if hasattr(view, 'window'):
                            QMessageBox.warning(view.window(), "Invalid Position", 
                                               "散热器只能放置在边界上！\nRadiators can only be placed on boundaries!")
                    self.drawing = False
                    return
            else:
                self.create_preview_item()
        
        super().mousePressEvent(event)
    
    def is_on_boundary(self, point):
        """检查点是否在边界上"""
        margin = 10  # 边界容差
        scene_rect = self.sceneRect()
        
        # 检查是否靠近边界
        on_left = abs(point.x() - scene_rect.left()) <= margin
        on_right = abs(point.x() - scene_rect.right()) <= margin
        on_top = abs(point.y() - scene_rect.top()) <= margin
        on_bottom = abs(point.y() - scene_rect.bottom()) <= margin
        
        return on_left or on_right or on_top or on_bottom
    
    def place_sensor(self, position):
        """放置传感器"""
        # 创建传感器状态
        sensor_state = {
            'id': self.component_id_counter,
            'type': 'sensor',
            'coords': (position.x(), position.y()),
            'temperature': 0.0  # 默认温度为0K
        }
        
        # 创建传感器图形项
        sensor_item = create_component_item(sensor_state)
        
        # 添加到场景
        self.addItem(sensor_item)
        sensor_item.setPos(position.x(), position.y())
        
        # 增加ID计数器
        self.component_id_counter += 1
        
        # 更新传感器列表
        if self.views():
            view = self.views()[0]
            if hasattr(view, 'window') and hasattr(view.window(), 'sidebar'):
                view.window().sidebar.update_sensor_list()
        
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
        elif self.current_draw_mode == 'radiator':
            # 散热器用线段预览
            from PyQt6.QtWidgets import QGraphicsLineItem
            self.temp_item = QGraphicsLineItem()
            pen.setColor(Qt.GlobalColor.magenta)
            pen.setWidth(4)
            self.temp_item.setPen(pen)
            
        if self.temp_item:
            self.addItem(self.temp_item)
    
    def mouseMoveEvent(self, event):
        """处理鼠标移动事件"""
        if self.drawing and self.temp_item:
            current_point = event.scenePos()
            
            if self.current_draw_mode in ['rect', 'capsule']:
                # 矩形和胶囊形使用矩形预览
                rect = QRectF(self.start_point, current_point).normalized()
                self.temp_item.setRect(rect)
            elif self.current_draw_mode == 'circle':
                # 圆形预览 - 以拖拽矩形的中心为圆心
                circle_rect = create_circle_rect_from_drag(self.start_point, current_point)
                self.temp_item.setRect(circle_rect)
            elif self.current_draw_mode == 'radiator':
                # 散热器线段预览，约束终点到边界
                constrained_end = self.constrain_to_boundary(current_point)
                from PyQt6.QtCore import QLineF
                line = QLineF(self.start_point, constrained_end)
                self.temp_item.setLine(line)
        
        super().mouseMoveEvent(event)
    
    def constrain_to_boundary(self, point):
        """将点约束到与起始点相同的边界上，强制散热器严格水平或垂直"""
        scene_rect = self.sceneRect()
        
        # 确定起始点在哪个边界上
        margin = 10  # 减小边界容差，提高精确度
        start_on_left = abs(self.start_point.x() - scene_rect.left()) <= margin
        start_on_right = abs(self.start_point.x() - scene_rect.right()) <= margin
        start_on_top = abs(self.start_point.y() - scene_rect.top()) <= margin
        start_on_bottom = abs(self.start_point.y() - scene_rect.bottom()) <= margin
        
        # 强制约束到完全相同的边界线上
        if start_on_left:
            # 沿左边界绘制（严格垂直）
            constrained_y = max(scene_rect.top(), min(scene_rect.bottom(), point.y()))
            return QPointF(scene_rect.left(), constrained_y)  # X坐标严格等于左边界
        elif start_on_right:
            # 沿右边界绘制（严格垂直）
            constrained_y = max(scene_rect.top(), min(scene_rect.bottom(), point.y()))
            return QPointF(scene_rect.right(), constrained_y)  # X坐标严格等于右边界
        elif start_on_top:
            # 沿上边界绘制（严格水平）
            constrained_x = max(scene_rect.left(), min(scene_rect.right(), point.x()))
            return QPointF(constrained_x, scene_rect.top())  # Y坐标严格等于上边界
        elif start_on_bottom:
            # 沿下边界绘制（严格水平）
            constrained_x = max(scene_rect.left(), min(scene_rect.right(), point.x()))
            return QPointF(constrained_x, scene_rect.bottom())  # Y坐标严格等于下边界
        else:
            # 如果起始点不在边界上，选择最近的边界并强制对齐
            dist_left = abs(self.start_point.x() - scene_rect.left())
            dist_right = abs(self.start_point.x() - scene_rect.right())
            dist_top = abs(self.start_point.y() - scene_rect.top())
            dist_bottom = abs(self.start_point.y() - scene_rect.bottom())
            
            min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
            
            if min_dist == dist_left:
                # 强制对齐到左边界，垂直绘制
                constrained_y = max(scene_rect.top(), min(scene_rect.bottom(), point.y()))
                return QPointF(scene_rect.left(), constrained_y)
            elif min_dist == dist_right:
                # 强制对齐到右边界，垂直绘制
                constrained_y = max(scene_rect.top(), min(scene_rect.bottom(), point.y()))
                return QPointF(scene_rect.right(), constrained_y)
            elif min_dist == dist_top:
                # 强制对齐到上边界，水平绘制
                constrained_x = max(scene_rect.left(), min(scene_rect.right(), point.x()))
                return QPointF(constrained_x, scene_rect.top())
            else:
                # 强制对齐到下边界，水平绘制
                constrained_x = max(scene_rect.left(), min(scene_rect.right(), point.x()))
                return QPointF(constrained_x, scene_rect.bottom())
    
    def align_point_to_boundary(self, point, scene_rect):
        """将点对齐到最近的边界"""
        margin = 10
        
        # 计算到各边界的距离
        dist_left = abs(point.x() - scene_rect.left())
        dist_right = abs(point.x() - scene_rect.right())
        dist_top = abs(point.y() - scene_rect.top())
        dist_bottom = abs(point.y() - scene_rect.bottom())
        
        min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
        
        # 对齐到最近的边界
        if min_dist == dist_left:
            return QPointF(scene_rect.left(), point.y())
        elif min_dist == dist_right:
            return QPointF(scene_rect.right(), point.y())
        elif min_dist == dist_top:
            return QPointF(point.x(), scene_rect.top())
        else:
            return QPointF(point.x(), scene_rect.bottom())
    
    def mouseReleaseEvent(self, event):
        """处理鼠标释放事件"""
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.drawing = False
            
            if self.temp_item:
                # 获取最终点位
                end_point = event.scenePos()
                
                # 移除预览项目
                self.removeItem(self.temp_item)
                self.temp_item = None
                
                if self.current_draw_mode == 'radiator':
                    # 散热器创建逻辑
                    constrained_end = self.constrain_to_boundary(end_point)
                    line_length = ((constrained_end.x() - self.start_point.x())**2 + 
                                   (constrained_end.y() - self.start_point.y())**2)**0.5
                    
                    if line_length > MIN_COMPONENT_SIZE:
                        # 创建散热器
                        self.create_radiator(self.start_point, constrained_end)
                else:
                    # 其他组件的创建逻辑
                    final_rect = QRectF(self.start_point, end_point).normalized()
                    
                    # 检查是否有效（避免只是点击）并且在边界内
                    if final_rect.width() > MIN_COMPONENT_SIZE and final_rect.height() > MIN_COMPONENT_SIZE:
                        # 检查绘制区域是否完全在场景边界内
                        if validate_drawing_bounds(final_rect, self.sceneRect()):
                            self.create_permanent_item(final_rect)
                        else:
                            # 绘制区域超出边界，显示提示
                            if self.views():
                                view = self.views()[0]
                                if hasattr(view, 'window'):
                                    QMessageBox.warning(view.window(), "Invalid Drawing", 
                                                       "Drawing area must be completely within the canvas bounds!")
                            print("Warning: Drawing area exceeds canvas bounds")
        
        super().mouseReleaseEvent(event)
    
    def create_radiator(self, start_point, end_point):
        """创建散热器"""
        # 确保起始点和终点都严格对齐到边界
        scene_rect = self.sceneRect()
        
        # 对齐起始点到最近的边界
        aligned_start = self.align_point_to_boundary(start_point, scene_rect)
        
        # 对齐终点（已经通过constrain_to_boundary处理过）
        aligned_end = end_point
        
        # 创建散热器状态
        radiator_state = {
            'id': self.component_id_counter,
            'type': 'radiator',
            'start_point': (aligned_start.x(), aligned_start.y()),
            'end_point': (aligned_end.x(), aligned_end.y()),
            'coords': ((aligned_start.x() + aligned_end.x()) / 2, (aligned_start.y() + aligned_end.y()) / 2),  # 中心点
            'heat_dissipation_rate': 10.0  # 默认散热效率
        }
        
        # 创建图形项
        item = create_component_item(radiator_state)
        
        # 添加到场景（散热器不需要setPos，因为坐标已在绘制中处理）
        self.addItem(item)
        
        # 增加ID计数器
        self.component_id_counter += 1
        
        # 更新组件列表
        if self.views():
            view = self.views()[0]
            if hasattr(view, 'window') and hasattr(view.window(), 'update_components_list'):
                view.window().update_components_list()
    
    def create_permanent_item(self, rect: QRectF):
        """创建永久组件"""
        # 如果没有选择绘制模式，不创建组件
        if self.current_draw_mode is None:
            return
            
        # 计算中心点和尺寸
        center_x = rect.center().x()
        center_y = rect.center().y()
        
        # 生成随机功率
        power = generate_random_power()
        
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
        
        # 创建图形项
        item = create_component_item(component_state)
        
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