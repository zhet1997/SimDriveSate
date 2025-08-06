"""
UI工具函数
包含坐标转换、图像处理等公共工具方法
"""

import io
import random
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import QRectF
from ui_constants import DEFAULT_POWER_RANGE, SCENE_SCALE


def pixels_to_meters(pixels: float, scene_scale: float = SCENE_SCALE) -> float:
    """将像素坐标转换为米单位"""
    return pixels / scene_scale


def meters_to_pixels(meters: float, scene_scale: float = SCENE_SCALE) -> float:
    """将米单位转换为像素坐标"""
    return meters * scene_scale


def convert_coords_to_meters(coords: Tuple[float, float], scene_scale: float = SCENE_SCALE) -> Tuple[float, float]:
    """将坐标从像素转换为米"""
    x, y = coords
    return (x / scene_scale, y / scene_scale)


def convert_coords_to_pixels(coords: Tuple[float, float], scene_scale: float = SCENE_SCALE) -> Tuple[float, float]:
    """将坐标从米转换为像素"""
    x, y = coords
    return (x * scene_scale, y * scene_scale)


def convert_size_to_meters(size, scene_scale: float = SCENE_SCALE):
    """将尺寸从像素转换为米"""
    if isinstance(size, (list, tuple)):
        return tuple(s / scene_scale for s in size)
    else:
        return size / scene_scale


def convert_size_to_pixels(size, scene_scale: float = SCENE_SCALE):
    """将尺寸从米转换为像素"""
    if isinstance(size, (list, tuple)):
        return tuple(s * scene_scale for s in size)
    else:
        return size * scene_scale


def generate_random_power(min_power: float = None, max_power: float = None) -> float:
    """生成随机功率值"""
    if min_power is None:
        min_power = DEFAULT_POWER_RANGE[0]
    if max_power is None:
        max_power = DEFAULT_POWER_RANGE[1]
    return random.uniform(min_power, max_power)


def create_circle_rect_from_drag(start_point, current_point) -> QRectF:
    """根据拖拽创建圆形的边界矩形（以中心为圆心）"""
    # 获取拖拽矩形
    rect = QRectF(start_point, current_point).normalized()
    
    # 使用较短边作为直径，保持圆形
    size = min(rect.width(), rect.height())
    
    # 计算圆形应该放置的位置（以拖拽矩形中心为圆心）
    center_x = rect.center().x()
    center_y = rect.center().y()
    radius = size / 2
    
    return QRectF(center_x - radius, center_y - radius, size, size)


def create_matplotlib_figure(sdf_array: np.ndarray, scene_width: float, scene_height: float) -> QPixmap:
    """创建SDF的matplotlib图像并转换为QPixmap"""
    from ui_constants import SDFConfig
    
    # 创建matplotlib图形，精确匹配场景像素尺寸
    dpi = SDFConfig.DPI
    fig_width = scene_width / dpi
    fig_height = scene_height / dpi
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    fig.patch.set_facecolor('none')  # 透明背景
    ax.set_facecolor('none')
    
    # 创建SDF图像 - 像素级精确对齐
    im = ax.imshow(sdf_array, cmap=SDFConfig.COLORMAP, origin='upper',
                  extent=[0, scene_width, 0, scene_height],
                  alpha=SDFConfig.ALPHA, interpolation=SDFConfig.INTERPOLATION)
    ax.axis('off')  # 隐藏坐标轴
    
    # 设置精确的边距和布局
    ax.set_xlim(0, scene_width)
    ax.set_ylim(0, scene_height)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    
    # 保存为图像
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', 
               transparent=True, pad_inches=0)
    buf.seek(0)
    
    # 转换为QPixmap
    qimg = QImage()
    qimg.loadFromData(buf.getvalue())
    pixmap = QPixmap.fromImage(qimg)
    
    # 清理matplotlib资源
    plt.close(fig)
    
    return pixmap


def convert_component_to_meters(component_state: Dict, scene_scale: float = SCENE_SCALE) -> Dict:
    """将组件状态中的坐标和尺寸转换为米单位"""
    state = component_state.copy()
    
    # 转换坐标
    coords = state['coords']
    state['coords'] = convert_coords_to_meters(coords, scene_scale)
    
    # 转换尺寸
    if state['type'] == 'circle':
        state['size'] = convert_size_to_meters(state['size'], scene_scale)
    else:
        size = state['size']
        state['size'] = convert_size_to_meters(size, scene_scale)
    
    return state


def setup_application_font(family: str = None, size: int = None):
    """设置应用程序字体"""
    from ui_constants import DEFAULT_FONT_FAMILY, DEFAULT_FONT_SIZE
    
    font = QFont()
    font.setFamily(family or DEFAULT_FONT_FAMILY)
    font.setPointSize(size or DEFAULT_FONT_SIZE)
    return font


def create_component_font(size: int = None):
    """创建组件专用字体"""
    from ui_constants import DEFAULT_FONT_FAMILY, COMPONENT_FONT_SIZE
    
    font = QFont()
    font.setFamily(DEFAULT_FONT_FAMILY)
    font.setPointSize(size or COMPONENT_FONT_SIZE)
    return font


def format_position_text(coords: Tuple[float, float], scene_scale: float = SCENE_SCALE) -> str:
    """格式化位置文本显示"""
    meters_coords = convert_coords_to_meters(coords, scene_scale)
    return f"Position: ({meters_coords[0]:.2f}, {meters_coords[1]:.2f})m"


def format_size_text(component_type: str, size, scene_scale: float = SCENE_SCALE) -> str:
    """格式化尺寸文本显示"""
    if component_type == 'circle':
        meters_size = convert_size_to_meters(size, scene_scale)
        return f"Radius: {meters_size:.2f}m"
    else:
        meters_size = convert_size_to_meters(size, scene_scale)
        return f"Size: {meters_size[0]:.2f}×{meters_size[1]:.2f}m"


def validate_drawing_bounds(rect: QRectF, scene_rect: QRectF) -> bool:
    """验证绘制区域是否在场景边界内"""
    return scene_rect.contains(rect)


def calculate_sdf_grid_shape(layout_size: Tuple[float, float]) -> Tuple[int, int]:
    """计算SDF网格形状"""
    from ui_constants import GRID_INTERVAL_METERS, SDFConfig
    
    # 计算与Qt网格匹配的SDF分辨率
    grid_resolution = int(layout_size[0] / GRID_INTERVAL_METERS) * SDFConfig.GRID_MULTIPLIER
    return (grid_resolution, grid_resolution)