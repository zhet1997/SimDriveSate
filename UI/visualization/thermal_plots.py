"""
温度场专用绘图功能
提供温度场可视化的专门优化
"""

from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtGui import QPixmap

from .field_visualizer import FieldVisualizer, VisualizationConfig
from .plot_configs import PlotStyle, PresetStyles

# 修复导入路径问题
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from backends.base_backend import ComputationResult, FieldType
except ImportError:
    # 尝试相对导入作为后备
    from ..backends.base_backend import ComputationResult, FieldType


class ThermalFieldPlotter:
    """温度场专用绘图器"""
    
    @staticmethod
    def create_thermal_visualization(temperature_data: np.ndarray,
                                   scene_width: float,
                                   scene_height: float,
                                   layout_domain: Tuple[float, float] = (0.2, 0.2),
                                   metadata: Optional[Dict[str, Any]] = None,
                                   style_override: Optional[PlotStyle] = None) -> QPixmap:
        """创建温度场可视化
        
        Args:
            temperature_data: 温度数据数组
            scene_width: 场景宽度（像素）
            scene_height: 场景高度（像素）
            layout_domain: 布局域尺寸（米）
            metadata: 元数据
            style_override: 样式覆盖
            
        Returns:
            QPixmap: 温度场图像
        """
        # 创建计算结果对象
        metadata = metadata or {"unit": "K", "description": "温度场分布"}
        result = ComputationResult(
            field_data=temperature_data,
            field_type=FieldType.TEMPERATURE,
            metadata=metadata
        )
        
        # 使用温度场专用样式
        style = style_override or PresetStyles.thermal_field_style()
        
        # 创建可视化配置
        config = VisualizationConfig(
            scene_width=scene_width,
            scene_height=scene_height,
            layout_domain=layout_domain,
            style=style
        )
        
        return FieldVisualizer.create_field_visualization(result, config)
    
    @staticmethod
    def create_thermal_with_isotherms(temperature_data: np.ndarray,
                                    scene_width: float,
                                    scene_height: float,
                                    layout_domain: Tuple[float, float] = (0.2, 0.2),
                                    isotherm_temps: Optional[List[float]] = None) -> QPixmap:
        """创建带等温线的温度场可视化
        
        Args:
            temperature_data: 温度数据
            scene_width: 场景宽度
            scene_height: 场景高度
            layout_domain: 布局域
            isotherm_temps: 等温线温度值列表
            
        Returns:
            QPixmap: 带等温线的温度场图像
        """
        # 自动计算等温线温度
        if isotherm_temps is None:
            temp_min, temp_max = np.min(temperature_data), np.max(temperature_data)
            isotherm_temps = np.linspace(temp_min, temp_max, 10).tolist()
        
        # 创建带等温线的样式
        style = PresetStyles.thermal_field_style()
        style.contour_lines = True
        style.title = f"温度场分布（等温线: {len(isotherm_temps)}条）"
        
        # 元数据中包含等温线信息
        metadata = {
            "unit": "K",
            "description": "带等温线的温度场",
            "isotherm_temperatures": isotherm_temps,
            "temperature_range": (float(np.min(temperature_data)), float(np.max(temperature_data)))
        }
        
        return ThermalFieldPlotter.create_thermal_visualization(
            temperature_data, scene_width, scene_height, layout_domain, metadata, style
        )
    
    @staticmethod
    def create_thermal_gradient_visualization(temperature_data: np.ndarray,
                                            scene_width: float,
                                            scene_height: float,
                                            layout_domain: Tuple[float, float] = (0.2, 0.2),
                                            show_vectors: bool = False) -> QPixmap:
        """创建温度梯度可视化
        
        Args:
            temperature_data: 温度数据
            scene_width: 场景宽度
            scene_height: 场景高度  
            layout_domain: 布局域
            show_vectors: 是否显示梯度矢量
            
        Returns:
            QPixmap: 温度梯度图像
        """
        # 计算温度梯度
        grad_y, grad_x = np.gradient(temperature_data)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 创建梯度结果
        metadata = {
            "unit": "K/m",
            "description": "温度梯度幅值",
            "gradient_components": {
                "grad_x": grad_x.tolist(),
                "grad_y": grad_y.tolist()
            }
        }
        
        result = ComputationResult(
            field_data=gradient_magnitude,
            field_type=FieldType.TEMPERATURE,  # 仍然是温度相关的场
            metadata=metadata,
            auxiliary_data={"grad_x": grad_x, "grad_y": grad_y}
        )
        
        # 使用专门的梯度样式
        style = PresetStyles.thermal_field_style()
        style.colormap = "viridis"  # 梯度更适合用viridis
        style.colorbar_label = "温度梯度 (K/m)"
        style.title = "温度梯度分布"
        
        config = VisualizationConfig(
            scene_width=scene_width,
            scene_height=scene_height,
            layout_domain=layout_domain,
            style=style
        )
        
        return FieldVisualizer.create_field_visualization(result, config)
    
    @staticmethod
    def create_temperature_profile(temperature_data: np.ndarray,
                                 layout_domain: Tuple[float, float] = (0.2, 0.2),
                                 profile_type: str = "horizontal",
                                 position: float = 0.5) -> QPixmap:
        """创建温度剖面图
        
        Args:
            temperature_data: 温度数据
            layout_domain: 布局域
            profile_type: 剖面类型 ("horizontal" 或 "vertical")
            position: 剖面位置 (0-1之间的归一化位置)
            
        Returns:
            QPixmap: 温度剖面图
        """
        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
        
        H, W = temperature_data.shape
        
        if profile_type == "horizontal":
            # 水平剖面
            row_idx = int(position * H)
            profile = temperature_data[row_idx, :]
            x_coords = np.linspace(0, layout_domain[0], W)
            ax.plot(x_coords, profile, 'b-', linewidth=2)
            ax.set_xlabel("X 坐标 (m)")
            ax.set_title(f"水平剖面 (Y = {position * layout_domain[1]:.3f} m)")
        else:
            # 垂直剖面
            col_idx = int(position * W)
            profile = temperature_data[:, col_idx]
            y_coords = np.linspace(0, layout_domain[1], H)
            ax.plot(y_coords, profile, 'r-', linewidth=2)
            ax.set_xlabel("Y 坐标 (m)")
            ax.set_title(f"垂直剖面 (X = {position * layout_domain[0]:.3f} m)")
        
        ax.set_ylabel("温度 (K)")
        ax.grid(True, alpha=0.3)
        
        # 转换为QPixmap
        pixmap = FieldVisualizer._fig_to_pixmap(fig)
        plt.close(fig)
        
        return pixmap


# 向后兼容函数
def create_temperature_background(temp_array: np.ndarray, 
                                scene_width: float, 
                                scene_height: float) -> QPixmap:
    """创建温度场背景图像（向后兼容函数）"""
    return ThermalFieldPlotter.create_thermal_visualization(
        temperature_data=temp_array,
        scene_width=scene_width,
        scene_height=scene_height,
        layout_domain=(0.2, 0.2)  # 默认布局域
    )
