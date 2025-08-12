"""
泰森多边形计算后端
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import Voronoi, voronoi_plot_2d
from typing import List, Dict, Any, Tuple
from PyQt6.QtGui import QPixmap
import io

from .base_backend import ComputationBackendV2, FieldType, ComputationResult


class VoronoiBackend(ComputationBackendV2):
    """泰森多边形计算后端"""
    
    def __init__(self):
        super().__init__()
        self.name = "Voronoi Diagram Backend"
        self.supported_fields = [FieldType.VORONOI]
        self.layout_size = (0.1, 0.1)  # 默认布局尺寸（米）
    
    def validate_input_data(self, input_data: Dict[str, Any]) -> Tuple[bool, str]:
        """验证输入数据"""
        if not isinstance(input_data, dict):
            return False, "输入数据必须是字典格式"
        
        if 'components' not in input_data:
            return False, "缺少必需的'components'字段"
        
        components = input_data['components']
        if not isinstance(components, list):
            return False, "'components'必须是列表格式"
        
        if len(components) < 2:
            return False, "泰森多边形计算至少需要2个组件"
        
        if len(components) == 2:
            # 对于只有2个点的情况，给出特殊提示但仍允许计算
            print("[VoronoiBackend] 警告: 只有2个组件，将生成简化的泰森图")
        
        # 验证每个组件
        for i, comp in enumerate(components):
            if not isinstance(comp, dict):
                return False, f"组件 {i} 必须是字典格式"
            
            if 'center' not in comp:
                return False, f"组件 {i} 缺少'center'字段"
            
            center = comp['center']
            if not isinstance(center, (list, tuple)) or len(center) != 2:
                return False, f"组件 {i} 的'center'必须是包含2个坐标的列表或元组"
            
            try:
                float(center[0])
                float(center[1])
            except (ValueError, TypeError):
                return False, f"组件 {i} 的坐标必须是数值类型"
        
        return True, "输入数据验证通过"
    
    def compute_field(self, input_data: Dict[str, Any], field_type: FieldType) -> ComputationResult:
        """计算泰森多边形"""
        try:
            # 验证输入
            is_valid, error_msg = self.validate_input_data(input_data)
            if not is_valid:
                return ComputationResult(
                    field_data=None,
                    field_type=FieldType.VORONOI,
                    metadata={},
                    error_info=error_msg
                )
            
            if field_type != FieldType.VORONOI:
                return ComputationResult(
                    field_data=None,
                    field_type=FieldType.VORONOI,
                    metadata={},
                    error_info=f"不支持的字段类型: {field_type}"
                )
            
            # 提取组件中心点
            components = input_data['components']
            points = []
            
            for comp in components:
                center = comp['center']
                # 确保坐标是浮点数
                x, y = float(center[0]), float(center[1])
                points.append([x, y])
            
            points = np.array(points)
            print(f"[VoronoiBackend] 处理 {len(points)} 个点")
            
            # 获取布局尺寸
            self.layout_size = input_data.get('layout_size', (0.1, 0.1))
            
            # 计算泰森多边形
            voronoi_image = self._compute_voronoi_diagram(points, self.layout_size)
            
            return ComputationResult(
                field_data=voronoi_image,
                field_type=FieldType.VORONOI,
                metadata={
                    'field_type': 'voronoi',
                    'num_points': len(points),
                    'layout_size': self.layout_size,
                    'points': points.tolist()
                }
            )
            
        except Exception as e:
            return ComputationResult(
                field_data=None,
                field_type=FieldType.VORONOI,
                metadata={},
                error_info=f"泰森多边形计算失败: {str(e)}"
            )
    
    def _compute_voronoi_diagram(self, points: np.ndarray, layout_size: Tuple[float, float]) -> QPixmap:
        """计算并绘制泰森多边形图"""
        try:
            # 创建matplotlib图形
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.set_aspect('equal')
            
            # 设置坐标范围
            width, height = layout_size
            ax.set_xlim(0, width)
            ax.set_ylim(0, height)
            
            # 为了确保边界的泰森多边形闭合，添加边界外的辅助点
            boundary_points = self._add_boundary_points(points, layout_size)
            all_points = np.vstack([points, boundary_points])
            
            # 计算Voronoi图
            vor = Voronoi(all_points)
            
            # 绘制泰森多边形
            self._plot_voronoi_polygons(vor, ax, layout_size, len(points))
            
            # 绘制原始点
            ax.scatter(points[:, 0], points[:, 1], c='red', s=50, zorder=10, alpha=0.8)
            
            # 添加点的标号
            for i, point in enumerate(points):
                ax.annotate(f'{i+1}', (point[0], point[1]), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, color='darkred', weight='bold')
            
            # 设置标题和标签
            ax.set_title('泰森多边形 (Voronoi Diagram)', fontsize=14, pad=20)
            ax.set_xlabel('X (m)', fontsize=12)
            ax.set_ylabel('Y (m)', fontsize=12)
            
            # 添加网格
            ax.grid(True, alpha=0.3)
            
            # 移除多余的空白
            plt.tight_layout()
            
            # 转换为QPixmap
            pixmap = self._matplotlib_to_pixmap(fig)
            
            # 清理
            plt.close(fig)
            
            return pixmap
            
        except Exception as e:
            print(f"[VoronoiBackend] 绘制失败: {e}")
            # 创建错误图像
            return self._create_error_image(f"泰森多边形计算失败: {str(e)}")
    
    def _add_boundary_points(self, points: np.ndarray, layout_size: Tuple[float, float]) -> np.ndarray:
        """添加边界外的辅助点，确保泰森多边形闭合"""
        width, height = layout_size
        margin = max(width, height) * 2  # 足够大的边距
        
        # 基础边界点
        boundary_points = [
            [-margin, -margin],
            [width + margin, -margin],
            [width + margin, height + margin],
            [-margin, height + margin],
            [width/2, -margin],
            [width + margin, height/2],
            [width/2, height + margin],
            [-margin, height/2]
        ]
        
        # 如果原始点少于3个，添加额外的虚拟点确保Voronoi计算稳定
        if len(points) < 3:
            # 在布局中心添加一个虚拟点
            boundary_points.extend([
                [width/2, height/2],
                [width/4, height/4],
                [3*width/4, 3*height/4]
            ])
        
        return np.array(boundary_points)
    
    def _plot_voronoi_polygons(self, vor: Voronoi, ax, layout_size: Tuple[float, float], num_original_points: int):
        """绘制泰森多边形"""
        width, height = layout_size
        
        # 生成颜色映射
        colors = plt.cm.Set3(np.linspace(0, 1, num_original_points))
        
        # 只处理原始点对应的区域（前num_original_points个点）
        for i in range(num_original_points):
            region_index = vor.point_region[i]
            region = vor.regions[region_index]
            
            if not region or -1 in region:
                continue
            
            # 获取多边形顶点
            polygon_vertices = vor.vertices[region]
            
            # 裁剪到布局边界内
            clipped_vertices = self._clip_polygon_to_bounds(polygon_vertices, (0, 0, width, height))
            
            if len(clipped_vertices) >= 3:
                # 绘制多边形
                polygon = Polygon(clipped_vertices, alpha=0.6, facecolor=colors[i], 
                                edgecolor='black', linewidth=1.5)
                ax.add_patch(polygon)
    
    def _clip_polygon_to_bounds(self, vertices: np.ndarray, bounds: Tuple[float, float, float, float]) -> np.ndarray:
        """将多边形裁剪到指定边界内"""
        x_min, y_min, x_max, y_max = bounds
        
        # 简单的边界裁剪：只保留在边界内的顶点，并添加边界交点
        clipped = []
        
        for vertex in vertices:
            x, y = vertex
            # 裁剪到边界
            x = max(x_min, min(x_max, x))
            y = max(y_min, min(y_max, y))
            clipped.append([x, y])
        
        if len(clipped) >= 3:
            return np.array(clipped)
        else:
            # 如果裁剪后顶点太少，返回边界矩形的一部分
            return np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
    
    def _matplotlib_to_pixmap(self, fig) -> QPixmap:
        """将matplotlib图形转换为QPixmap"""
        from PyQt6.QtGui import QImage
        
        # 保存到内存
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        
        # 读取为QImage
        img_data = buf.getvalue()
        q_image = QImage.fromData(img_data)
        
        return QPixmap.fromImage(q_image)
    
    def _create_error_image(self, error_msg: str) -> QPixmap:
        """创建错误提示图像"""
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, f"错误:\n{error_msg}", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='red',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        pixmap = self._matplotlib_to_pixmap(fig)
        plt.close(fig)
        return pixmap
    
    def get_supported_field_types(self) -> List[FieldType]:
        """获取支持的字段类型"""
        return [FieldType.VORONOI]
