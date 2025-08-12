"""
物理场可视化统一接口
支持多种物理场的可视化，包括温度场、应力场、SDF等
"""

import io
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.contour import QuadContourSet
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QPointF

from .plot_configs import PlotStyle, PresetStyles, VisualizationConstants

# 修复导入路径问题
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from backends.base_backend import ComputationResult, FieldType
except ImportError:
    # 尝试相对导入作为后备
    from ..backends.base_backend import ComputationResult, FieldType


class VisualizationConfig:
    """可视化配置类"""
    
    def __init__(self, 
                 scene_width: float,
                 scene_height: float,
                 layout_domain: Tuple[float, float] = (0.2, 0.2),
                 style: Optional[PlotStyle] = None):
        self.scene_width = scene_width
        self.scene_height = scene_height
        self.layout_domain = layout_domain
        self.style = style or PresetStyles.minimal_style()
        
    def get_figure_size(self) -> Tuple[float, float]:
        """计算合适的图形尺寸"""
        dpi = self.style.figure_dpi
        fig_width = self.scene_width / dpi
        fig_height = self.scene_height / dpi
        
        # 应用尺寸限制
        max_w, max_h = VisualizationConstants.MAX_FIGURE_SIZE
        min_w, min_h = VisualizationConstants.MIN_FIGURE_SIZE
        
        fig_width = max(min_w, min(max_w, fig_width))
        fig_height = max(min_h, min(max_h, fig_height))
        
        return (fig_width, fig_height)


class FieldVisualizer:
    """统一的物理场可视化器"""
    
    @staticmethod
    def create_field_visualization(computation_result: ComputationResult,
                                 config: VisualizationConfig) -> QPixmap:
        """创建物理场可视化图像
        
        Args:
            computation_result: 计算结果
            config: 可视化配置
            
        Returns:
            QPixmap: Qt兼容的图像
        """
        if not computation_result.is_valid():
            return FieldVisualizer._create_error_image(config, computation_result.error_info)
        
        # 根据场类型选择合适的样式
        style = FieldVisualizer._get_field_style(computation_result.field_type, config.style)
        
        # 创建matplotlib图形
        fig_width, fig_height = config.get_figure_size()
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), 
                              dpi=style.figure_dpi,
                              facecolor='white')
        
        try:
            # 绘制主要场数据
            field_data = computation_result.field_data
            extent = [0, config.layout_domain[0], 0, config.layout_domain[1]]
            
            # 主图像
            im = ax.imshow(field_data, 
                          cmap=style.colormap,
                          alpha=style.alpha,
                          interpolation=style.interpolation,
                          extent=extent,
                          origin='lower',  # 确保坐标系正确
                          aspect='equal')
            
            # 添加等高线（如果启用）
            if style.contour_lines:
                FieldVisualizer._add_contour_lines(ax, field_data, extent, style)
            
            # 添加网格（如果启用）
            if style.grid_lines:
                FieldVisualizer._add_grid_lines(ax, style)
            
            # 添加色条（如果启用）
            if style.show_colorbar:
                FieldVisualizer._add_colorbar(fig, im, style, computation_result)
            
            # 设置标题和标签
            FieldVisualizer._set_labels_and_title(ax, style, config)
            
            # 移除边距，确保图像填满整个画布
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            ax.set_xlim(0, config.layout_domain[0])
            ax.set_ylim(0, config.layout_domain[1])
            
            # 转换为QPixmap
            pixmap = FieldVisualizer._fig_to_pixmap(fig)
            
        finally:
            plt.close(fig)  # 确保释放资源
            
        return pixmap
    
    @staticmethod
    def _get_field_style(field_type: FieldType, base_style: PlotStyle) -> PlotStyle:
        """根据场类型获取合适的样式"""
        if field_type == FieldType.TEMPERATURE:
            style = PresetStyles.thermal_field_style()
        elif field_type == FieldType.SDF:
            style = PresetStyles.sdf_field_style()
        elif field_type == FieldType.STRESS:
            style = PresetStyles.stress_field_style()
        else:
            style = PresetStyles.minimal_style()
        
        # 保留base_style中的UI相关设置
        style.show_colorbar = base_style.show_colorbar
        style.figure_dpi = base_style.figure_dpi
        style.title = base_style.title
        
        return style
    
    @staticmethod
    def _add_contour_lines(ax, field_data: np.ndarray, extent: List[float], style: PlotStyle):
        """添加等高线"""
        try:
            x = np.linspace(extent[0], extent[1], field_data.shape[1])
            y = np.linspace(extent[2], extent[3], field_data.shape[0])
            X, Y = np.meshgrid(x, y)
            
            contours = ax.contour(X, Y, field_data, 
                                levels=VisualizationConstants.DEFAULT_CONTOUR_LEVELS,
                                colors='black',
                                alpha=VisualizationConstants.CONTOUR_ALPHA,
                                linewidths=style.line_width)
        except Exception as e:
            print(f"Warning: Failed to add contour lines: {e}")
    
    @staticmethod
    def _add_grid_lines(ax, style: PlotStyle):
        """添加网格线"""
        ax.grid(True, 
               color=VisualizationConstants.GRID_COLOR,
               alpha=VisualizationConstants.GRID_ALPHA,
               linestyle=VisualizationConstants.GRID_LINESTYLE,
               linewidth=style.line_width)
    
    @staticmethod
    def _add_colorbar(fig, im, style: PlotStyle, result: ComputationResult):
        """添加色条"""
        try:
            cbar = fig.colorbar(im, 
                               fraction=VisualizationConstants.COLORBAR_FRACTION,
                               pad=VisualizationConstants.COLORBAR_PAD)
            cbar.set_label(style.colorbar_label, fontsize=style.colorbar_fontsize)
            
            # 添加单位信息（如果在元数据中）
            if 'unit' in result.metadata:
                unit = result.metadata['unit']
                current_label = cbar.get_label()
                if unit and unit not in current_label:
                    cbar.set_label(f"{current_label} ({unit})", fontsize=style.colorbar_fontsize)
                    
        except Exception as e:
            print(f"Warning: Failed to add colorbar: {e}")
    
    @staticmethod
    def _set_labels_and_title(ax, style: PlotStyle, config: VisualizationConfig):
        """设置标签和标题"""
        if style.title:
            ax.set_title(style.title, fontsize=style.title_fontsize)
        
        ax.set_xlabel("X (m)", fontsize=style.label_fontsize)
        ax.set_ylabel("Y (m)", fontsize=style.label_fontsize)
        
        # 隐藏刻度（适合UI嵌入）
        if not style.show_colorbar:
            ax.set_xticks([])
            ax.set_yticks([])
    
    @staticmethod
    def _fig_to_pixmap(fig) -> QPixmap:
        """将matplotlib图形转换为QPixmap"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', 
                   pad_inches=0, facecolor='white', dpi=fig.dpi)
        buf.seek(0)
        
        # 转换为QImage然后QPixmap
        qimg = QImage.fromData(buf.getvalue())
        return QPixmap.fromImage(qimg)
    
    @staticmethod
    def _create_error_image(config: VisualizationConfig, error_msg: Optional[str]) -> QPixmap:
        """创建错误图像"""
        fig_width, fig_height = config.get_figure_size()
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), 
                              dpi=config.style.figure_dpi,
                              facecolor='lightgray')
        
        ax.text(0.5, 0.5, f"计算错误\n{error_msg or '未知错误'}", 
               ha='center', va='center', transform=ax.transAxes,
               fontsize=12, color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        pixmap = FieldVisualizer._fig_to_pixmap(fig)
        plt.close(fig)
        
        return pixmap
