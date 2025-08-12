"""
自定义图形组件
包含RectItem、CircleItem、CapsuleItem等可绘制的图形元件
"""

from typing import Dict
from PyQt6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QStyle
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtCore import Qt, QRectF
from ui_constants import Colors, StyleSheets
from ui_utils import create_component_font


class BaseComponentItem(QGraphicsItem):
    """组件基类，提供公共功能"""
    
    def __init__(self, component_state: Dict):
        super().__init__()
        self.state = component_state
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        
        # 设置字体
        self.font = create_component_font()
        
        # 拖动状态控制
        self.is_dragging = False
        self.drag_start_pos = None

    def get_state(self) -> Dict:
        """获取组件状态"""
        return self.state
    
    def set_state(self, new_state: Dict):
        """设置组件状态"""
        self.state = new_state
        # 状态更新后，触发重绘
        self.update()

    def itemChange(self, change, value):
        """处理组件状态变化"""
        # 拖动期间不触发任何计算，只在mouseReleaseEvent中处理
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        """鼠标按下事件 - 开始拖动"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = True
            self.drag_start_pos = self.pos()
        super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件 - 结束拖动"""
        if event.button() == Qt.MouseButton.LeftButton and self.is_dragging:
            self.is_dragging = False
            # 检查位置是否真的改变了
            if self.drag_start_pos != self.pos():
                # 更新状态并触发计算
                new_pos = self.pos()
                self.state['coords'] = (new_pos.x(), new_pos.y())
                if self.scene():
                    self.scene().item_position_changed.emit()
                    # 更新组件列表显示
                    if self.scene().views():
                        view = self.scene().views()[0]
                        if hasattr(view, 'window') and hasattr(view.window(), 'update_components_list'):
                            view.window().update_components_list()
        super().mouseReleaseEvent(event)

    def _draw_power_label(self, painter: QPainter, rect: QRectF):
        """绘制功率标签的通用方法"""
        painter.setFont(self.font)
        painter.setPen(QPen(QColor(*Colors.TEXT_COLOR)))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"P={self.state['power']:.1f}W")


class RectItem(BaseComponentItem):
    """矩形组件"""
    
    def boundingRect(self) -> QRectF:
        w, h = self.state['size']
        return QRectF(-w/2, -h/2, w, h)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget):
        w, h = self.state['size']
        rect = QRectF(-w/2, -h/2, w, h)  # 以中心为原点绘制
        
        # 设置画笔和画刷
        painter.setPen(QPen(QColor(*Colors.RECT_BORDER), 1))
        if option.state & QStyle.StateFlag.State_Selected:
            painter.setBrush(QColor(*Colors.RECT_SELECTED))
        else:
            painter.setBrush(QColor(*Colors.RECT_FILL))
        painter.drawRect(rect)

        # 绘制功率标签
        self._draw_power_label(painter, rect)


class CircleItem(BaseComponentItem):
    """圆形组件"""
    
    def boundingRect(self) -> QRectF:
        r = self.state['size']
        return QRectF(-r, -r, 2*r, 2*r)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget):
        r = self.state['size']
        rect = QRectF(-r, -r, 2*r, 2*r)  # 以中心为原点绘制
        
        # 设置画笔和画刷
        painter.setPen(QPen(QColor(*Colors.CIRCLE_BORDER), 1))
        if option.state & QStyle.StateFlag.State_Selected:
            painter.setBrush(QColor(*Colors.CIRCLE_SELECTED))
        else:
            painter.setBrush(QColor(*Colors.CIRCLE_FILL))
        painter.drawEllipse(rect)

        # 绘制功率标签
        self._draw_power_label(painter, rect)


class CapsuleItem(BaseComponentItem):
    """胶囊组件"""
    
    def boundingRect(self) -> QRectF:
        length, width = self.state['size']
        return QRectF(-length/2, -width/2, length, width)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget):
        length, width = self.state['size']
        
        # 设置画笔和画刷
        painter.setPen(QPen(QColor(*Colors.CAPSULE_BORDER), 1))
        if option.state & QStyle.StateFlag.State_Selected:
            painter.setBrush(QColor(*Colors.CAPSULE_SELECTED))
        else:
            painter.setBrush(QColor(*Colors.CAPSULE_FILL))
        
        # 使用QPainterPath创建统一的胶囊形状
        from PyQt6.QtGui import QPainterPath
        
        capsule_path = QPainterPath()
        
        # 创建圆角矩形（胶囊形状）
        # 圆角半径为宽度的一半，这样就形成了胶囊形状
        radius = width / 2
        
        # 胶囊的边界矩形
        capsule_rect = QRectF(-length/2, -width/2, length, width)
        
        # 添加圆角矩形到路径
        capsule_path.addRoundedRect(capsule_rect, radius, radius)
        
        # 绘制统一的胶囊轮廓
        painter.drawPath(capsule_path)
        
        # 绘制功率标签
        full_rect = QRectF(-length/2, -width/2, length, width)
        self._draw_power_label(painter, full_rect)


class RadiatorItem(BaseComponentItem):
    """散热器组件（线段）"""
    
    def boundingRect(self) -> QRectF:
        start_x, start_y = self.state['start_point']
        end_x, end_y = self.state['end_point']
        # 创建包含线段的矩形，考虑线宽
        line_width = 8  # 散热器线宽
        min_x = min(start_x, end_x) - line_width/2
        max_x = max(start_x, end_x) + line_width/2
        min_y = min(start_y, end_y) - line_width/2
        max_y = max(start_y, end_y) + line_width/2
        return QRectF(min_x, min_y, max_x - min_x, max_y - min_y)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget):
        start_x, start_y = self.state['start_point']
        end_x, end_y = self.state['end_point']
        
        # 设置画笔
        line_width = 8
        painter.setPen(QPen(QColor(*Colors.RADIATOR_BORDER), line_width))
        if option.state & QStyle.StateFlag.State_Selected:
            painter.setPen(QPen(QColor(*Colors.RADIATOR_SELECTED), line_width))
        
        # 绘制散热器线段 - 使用QPointF避免类型错误
        from PyQt6.QtCore import QPointF
        start_point = QPointF(start_x, start_y)
        end_point = QPointF(end_x, end_y)
        painter.drawLine(start_point, end_point)
        
        # 绘制散热效率标签
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        text_rect = QRectF(mid_x - 30, mid_y - 10, 60, 20)
        painter.setFont(self.font)
        painter.setPen(QPen(QColor(*Colors.TEXT_COLOR)))
        heat_rate = self.state.get('heat_dissipation_rate', 10.0)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, f"H={heat_rate:.1f}")


class SensorPointItem(BaseComponentItem):
    """温度测点组件"""
    
    def boundingRect(self) -> QRectF:
        radius = 8  # 测点半径
        return QRectF(-radius, -radius, 2*radius, 2*radius)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget):
        radius = 8
        rect = QRectF(-radius, -radius, 2*radius, 2*radius)
        
        # 设置画笔和画刷
        painter.setPen(QPen(QColor(*Colors.SENSOR_BORDER), 2))
        if option.state & QStyle.StateFlag.State_Selected:
            painter.setBrush(QColor(*Colors.SENSOR_SELECTED))
        else:
            painter.setBrush(QColor(*Colors.SENSOR_FILL))
        
        # 绘制圆形测点
        painter.drawEllipse(rect)
        
        # 绘制十字标记
        painter.setPen(QPen(QColor(*Colors.SENSOR_BORDER), 2))
        from PyQt6.QtCore import QPointF
        # 使用QPointF对象避免类型错误
        painter.drawLine(QPointF(-radius/2, 0), QPointF(radius/2, 0))
        painter.drawLine(QPointF(0, -radius/2), QPointF(0, radius/2))
        
        # 绘制温度值
        if 'temperature' in self.state and self.state['temperature'] is not None:
            temp = self.state['temperature']
            painter.setFont(self.font)
            painter.setPen(QPen(QColor(*Colors.TEXT_COLOR)))
            text_rect = QRectF(-20, radius + 5, 40, 15)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, f"{temp:.1f}K")
        else:
            # 显示默认温度0K
            painter.setFont(self.font)
            painter.setPen(QPen(QColor(*Colors.TEXT_COLOR)))
            text_rect = QRectF(-20, radius + 5, 40, 15)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, "0.0K")


# 组件工厂函数
def create_component_item(component_state: Dict) -> BaseComponentItem:
    """根据组件状态创建对应的图形项"""
    component_type = component_state['type']
    
    if component_type == 'rect':
        return RectItem(component_state)
    elif component_type == 'circle':
        return CircleItem(component_state)
    elif component_type == 'capsule':
        return CapsuleItem(component_state)
    elif component_type == 'radiator':
        return RadiatorItem(component_state)
    elif component_type == 'sensor':
        return SensorPointItem(component_state)
    else:
        raise ValueError(f"Unsupported component type: {component_type}")