"""
自定义图形场景
处理拖拽绘制、鼠标事件和组件创建
"""

from typing import Optional
from PyQt6.QtWidgets import (QGraphicsScene, QGraphicsRectItem, QGraphicsEllipseItem, 
                             QMessageBox)
from PyQt6.QtGui import QPen
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal
from graphics_items import RectItem, CircleItem, CapsuleItem, create_component_item
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
        self.current_draw_mode = None  # 当前绘制模式：None, 'rect', 'circle', 'capsule'
        self.component_id_counter = 0
        
    def set_draw_mode(self, mode: Optional[str]):
        """设置当前绘制模式"""
        self.current_draw_mode = mode
        
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
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """处理鼠标释放事件"""
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