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

    def get_state(self) -> Dict:
        """获取组件状态"""
        return self.state

    def itemChange(self, change, value):
        """处理组件状态变化"""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.scene():
            # 更新坐标
            new_pos = value
            self.state['coords'] = (new_pos.x(), new_pos.y())
            # 通知场景更新
            self.scene().item_position_changed.emit()
        return super().itemChange(change, value)

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
        # 以中心为原点绘制
        rect_length = length - width  # 矩形部分长度
        
        # 设置画笔和画刷
        painter.setPen(QPen(QColor(*Colors.CAPSULE_BORDER), 1))
        if option.state & QStyle.StateFlag.State_Selected:
            painter.setBrush(QColor(*Colors.CAPSULE_SELECTED))
        else:
            painter.setBrush(QColor(*Colors.CAPSULE_FILL))
        
        # 绘制矩形部分
        rect = QRectF(-rect_length/2, -width/2, rect_length, width)
        painter.drawRect(rect)
        
        # 绘制左右半圆
        left_circle_rect = QRectF(-rect_length/2 - width/2, -width/2, width, width)
        right_circle_rect = QRectF(rect_length/2 - width/2, -width/2, width, width)
        painter.drawEllipse(left_circle_rect)
        painter.drawEllipse(right_circle_rect)

        # 绘制功率标签
        full_rect = QRectF(-length/2, -width/2, length, width)
        self._draw_power_label(painter, full_rect)


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
    else:
        raise ValueError(f"Unsupported component type: {component_type}")