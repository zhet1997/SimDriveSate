"""
绘图配置和样式定义
集中管理所有可视化的样式配置
"""

from enum import Enum
from typing import Dict, Any, Tuple
from dataclasses import dataclass


class ColorMaps(Enum):
    """色彩映射枚举"""
    # 温度场专用
    THERMAL_HOT = "hot"
    THERMAL_PLASMA = "plasma"
    THERMAL_INFERNO = "inferno"
    
    # 应力场专用
    STRESS_VIRIDIS = "viridis"
    STRESS_COOLWARM = "coolwarm"
    
    # SDF专用
    SDF_SEISMIC = "seismic"
    SDF_RDB = "RdBu"
    
    # 通用
    GENERAL_JET = "jet"
    GENERAL_RAINBOW = "rainbow"


@dataclass
class PlotStyle:
    """绘图样式配置"""
    colormap: str = "viridis"
    alpha: float = 0.8
    interpolation: str = "bilinear"
    show_colorbar: bool = True
    colorbar_label: str = "Value"
    grid_lines: bool = False
    contour_lines: bool = False
    title: str = ""
    
    # 字体配置
    title_fontsize: int = 14
    label_fontsize: int = 12
    colorbar_fontsize: int = 10
    
    # 尺寸配置
    figure_dpi: int = 100
    line_width: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'colormap': self.colormap,
            'alpha': self.alpha,
            'interpolation': self.interpolation,
            'show_colorbar': self.show_colorbar,
            'colorbar_label': self.colorbar_label,
            'grid_lines': self.grid_lines,
            'contour_lines': self.contour_lines,
            'title': self.title,
            'title_fontsize': self.title_fontsize,
            'label_fontsize': self.label_fontsize,
            'colorbar_fontsize': self.colorbar_fontsize,
            'figure_dpi': self.figure_dpi,
            'line_width': self.line_width
        }


class PresetStyles:
    """预设样式库"""
    
    @staticmethod
    def thermal_field_style() -> PlotStyle:
        """温度场专用样式"""
        return PlotStyle(
            colormap=ColorMaps.THERMAL_PLASMA.value,
            alpha=0.85,
            interpolation="bilinear",
            show_colorbar=True,
            colorbar_label="温度 (K)",
            contour_lines=True,
            title="温度场分布"
        )
    
    @staticmethod
    def sdf_field_style() -> PlotStyle:
        """SDF专用样式"""
        return PlotStyle(
            colormap=ColorMaps.SDF_RDB.value,
            alpha=0.7,
            interpolation="nearest",
            show_colorbar=True,
            colorbar_label="距离 (m)",
            grid_lines=False,
            title="有符号距离场"
        )
    
    @staticmethod
    def stress_field_style() -> PlotStyle:
        """应力场专用样式"""
        return PlotStyle(
            colormap=ColorMaps.STRESS_COOLWARM.value,
            alpha=0.8,
            interpolation="bilinear",
            show_colorbar=True,
            colorbar_label="应力 (Pa)",
            contour_lines=True,
            title="应力分布"
        )
    
    @staticmethod
    def minimal_style() -> PlotStyle:
        """简洁样式（适合嵌入UI）"""
        return PlotStyle(
            colormap=ColorMaps.GENERAL_JET.value,
            alpha=0.8,
            interpolation="bilinear",
            show_colorbar=False,  # 不显示色条，节省空间
            grid_lines=False,
            title="",
            title_fontsize=10,
            label_fontsize=8
        )


class VisualizationConstants:
    """可视化常量定义"""
    
    # 默认分辨率
    DEFAULT_DPI = 100
    DEFAULT_ALPHA = 0.8
    
    # 色条配置
    COLORBAR_FRACTION = 0.046
    COLORBAR_PAD = 0.04
    
    # 等高线配置
    DEFAULT_CONTOUR_LEVELS = 20
    CONTOUR_ALPHA = 0.6
    
    # 网格配置
    GRID_ALPHA = 0.3
    GRID_COLOR = 'gray'
    GRID_LINESTYLE = '--'
    
    # 支持的图像格式
    SUPPORTED_FORMATS = ['png', 'jpg', 'svg', 'pdf']
    
    # 尺寸限制
    MAX_FIGURE_SIZE = (20, 20)  # 英寸
    MIN_FIGURE_SIZE = (2, 2)    # 英寸
