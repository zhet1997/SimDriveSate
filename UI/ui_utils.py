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


def create_matplotlib_figure(data_array: np.ndarray, scene_width: float, scene_height: float, 
                            colormap: str = 'coolwarm', alpha: float = 0.6, 
                            interpolation: str = 'nearest', 
                            colorbar_label: str = 'Value', 
                            show_colorbar: bool = True) -> QPixmap:
    """
    通用的matplotlib图像创建函数，支持SDF、温度场等多种数据可视化
    已重构使用新的统一图像集成接口
    
    Args:
        data_array: 要可视化的数据数组
        scene_width: 场景宽度（像素）
        scene_height: 场景高度（像素）
        colormap: 颜色映射名称
        alpha: 透明度
        interpolation: 插值方法
        colorbar_label: 色条标签
        show_colorbar: 是否显示色条
    
    Returns:
        QPixmap: 转换后的图像
    """
    # 使用新的统一图像集成接口
    from image_integration_interface import QtMatplotlibIntegration
    
    return QtMatplotlibIntegration.create_qt_compatible_image(
        data_array=data_array,
        scene_width=scene_width,
        scene_height=scene_height,
        colormap=colormap,
        alpha=alpha,
        interpolation=interpolation,
        show_colorbar=show_colorbar,
        colorbar_label=colorbar_label
    )


def create_sdf_figure(sdf_array: np.ndarray, scene_width: float, scene_height: float) -> QPixmap:
    """创建SDF图像（向后兼容的封装函数）"""
    # 使用新的统一接口，专门为SDF优化
    from image_integration_interface import create_sdf_background
    return create_sdf_background(sdf_array, scene_width, scene_height)


def create_temperature_figure(temp_array: np.ndarray, scene_width: float, scene_height: float) -> QPixmap:
    """创建温度场图像（泰森多边形风格）"""
    # 使用新的统一接口，专门为温度场优化，移除ColorBar避免比例失调
    from image_integration_interface import create_temperature_background
    return create_temperature_background(temp_array, scene_width, scene_height)


def convert_component_to_meters(component_state: Dict, scene_scale: float = SCENE_SCALE) -> Dict:
    """将组件状态中的坐标和尺寸转换为米单位"""
    state = component_state.copy()
    
    # 转换坐标
    coords = state['coords']
    state['coords'] = convert_coords_to_meters(coords, scene_scale)
    
    # 转换尺寸（只有物理组件有size属性）
    if state['type'] in ['rect', 'circle', 'capsule']:
        if state['type'] == 'circle':
            state['size'] = convert_size_to_meters(state['size'], scene_scale)
        else:
            size = state['size']
            state['size'] = convert_size_to_meters(size, scene_scale)
    
    # 散热器组件需要转换起始点和终点
    if state['type'] == 'radiator':
        state['start_point'] = convert_coords_to_meters(state['start_point'], scene_scale)
        state['end_point'] = convert_coords_to_meters(state['end_point'], scene_scale)
    
    # 传感器组件不需要额外转换（只有coords和temperature）
    
    return state


def setup_application_font(family: str = None, size: int = None):
    """设置应用程序字体，支持中文和emoji显示"""
    from ui_constants import DEFAULT_FONT_SIZE
    from PyQt6.QtGui import QFontDatabase
    
    # 获取系统可用字体
    available_fonts = QFontDatabase.families()
    
    # 首选字体列表（基于系统实际可用字体优化）
    preferred_fonts = [
        "Noto Sans CJK SC",           # Google Noto 简体中文（最佳选择）
        "WenQuanYi Micro Hei",        # 文泉驿微米黑（开源中文字体）
        "WenQuanYi Zen Hei",          # 文泉驿正黑
        "Microsoft YaHei UI",         # Windows 微软雅黑
        "PingFang SC",                # macOS 苹方简体
        "Source Han Sans SC",         # Adobe 思源黑体
        "Arial Unicode MS",           # Unicode 支持
        "SimHei", "SimSun"            # Windows 默认中文字体
    ]
    
    # 通用Unicode字体回退（适用于WSL等环境）
    universal_fonts = [
        "DejaVu Sans", "Liberation Sans", "FreeSans", "Arial", "Helvetica", "sans-serif"
    ]
    
    # 构建实际可用的字体链
    font_chain = []
    
    # 添加首选字体（如果可用）
    for font_name in preferred_fonts:
        if font_name in available_fonts:
            font_chain.append(font_name)
    
    # 添加通用字体
    for font_name in universal_fonts:
        if font_name in available_fonts:
            font_chain.append(font_name)
    
    # 如果没有找到任何字体，使用系统默认
    if not font_chain:
        font_chain = ["sans-serif"]
    
    # 创建字体
    font = QFont()
    if len(font_chain) == 1:
        font.setFamily(font_chain[0])
    else:
        font.setFamilies(font_chain)  # Qt6 支持字体回退链
    
    font.setPointSize(size or DEFAULT_FONT_SIZE)
    
    # 启用抗锯齿和亚像素渲染（提高显示质量）
    font.setStyleHint(QFont.StyleHint.SansSerif)
    font.setHintingPreference(QFont.HintingPreference.PreferDefaultHinting)
    
    return font


def create_component_font(size: int = None):
    """创建组件专用字体，支持中文显示"""
    from ui_constants import COMPONENT_FONT_SIZE
    from PyQt6.QtGui import QFontDatabase
    
    # 获取可用字体
    available_fonts = QFontDatabase.families()
    
    # 组件字体优先级（简洁版）
    preferred_fonts = [
        "Noto Sans CJK SC", "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
        "Microsoft YaHei UI", "DejaVu Sans", "Arial", "sans-serif"
    ]
    
    # 构建可用字体链
    font_chain = []
    for font_name in preferred_fonts:
        if font_name in available_fonts:
            font_chain.append(font_name)
    
    if not font_chain:
        font_chain = ["sans-serif"]
    
    font = QFont()
    if len(font_chain) == 1:
        font.setFamily(font_chain[0])
    else:
        font.setFamilies(font_chain)
    
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