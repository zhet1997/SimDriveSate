"""
Qt与Matplotlib图像集成统一接口
解决两个绘图系统的坐标系统冲突和比例失调问题
"""

import io
from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QPointF


class QtMatplotlibIntegration:
    """Qt与Matplotlib图像集成统一接口"""
    
    # 默认配置
    DEFAULT_DPI = 100
    DEFAULT_ALPHA = 0.8
    DEFAULT_INTERPOLATION = 'nearest'
    
    @staticmethod
    def create_qt_compatible_image(data_array: np.ndarray, 
                                  scene_width: float, 
                                  scene_height: float,
                                  colormap: str = 'viridis',
                                  alpha: float = DEFAULT_ALPHA,
                                  interpolation: str = DEFAULT_INTERPOLATION,
                                  show_colorbar: bool = False,
                                  colorbar_label: str = 'Value') -> QPixmap:
        """
        创建与Qt坐标系兼容的matplotlib图像
        
        Args:
            data_array: 要可视化的数据数组
            scene_width: Qt场景宽度（像素）
            scene_height: Qt场景高度（像素）
            colormap: matplotlib颜色映射
            alpha: 透明度 (0.0-1.0)
            interpolation: 插值方法
            show_colorbar: 是否显示色条（注意：会影响图像尺寸）
            colorbar_label: 色条标签
            
        Returns:
            QPixmap: 与Qt坐标系兼容的图像
        """
        # 创建matplotlib图形，精确匹配Qt场景尺寸
        dpi = QtMatplotlibIntegration.DEFAULT_DPI
        fig_width = scene_width / dpi
        fig_height = scene_height / dpi
        
        # 如果显示colorbar，需要为其预留空间
        if show_colorbar:
            # 为colorbar预留15%的宽度
            actual_width = fig_width * 0.85
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        else:
            actual_width = fig_width
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        
        # 设置透明背景
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        
        # 关键修复：使用origin='upper'匹配Qt坐标系（左上角原点，Y轴向下）
        im = ax.imshow(data_array, 
                      cmap=colormap, 
                      origin='upper',  # 匹配Qt坐标系
                      extent=[0, scene_width, scene_height, 0],  # 注意：Y轴范围颠倒
                      alpha=alpha, 
                      interpolation=interpolation)
        
        # 隐藏坐标轴
        ax.axis('off')
        
        # 可选：添加色条
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.15, pad=0.02, shrink=0.8)
            cbar.set_label(colorbar_label, fontsize=10)
            # 设置边距以容纳colorbar
            plt.subplots_adjust(left=0, right=0.85, top=1, bottom=0, wspace=0, hspace=0)
        else:
            # 零边距，精确匹配场景尺寸
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        
        # 设置精确的坐标限制
        ax.set_xlim(0, scene_width)
        ax.set_ylim(scene_height, 0)  # Y轴颠倒以匹配Qt
        
        # 转换为QPixmap
        pixmap = QtMatplotlibIntegration._fig_to_pixmap(fig, dpi)
        
        # 清理matplotlib资源
        plt.close(fig)
        
        return pixmap
    
    @staticmethod
    def create_no_colorbar_image(data_array: np.ndarray,
                                scene_width: float,
                                scene_height: float,
                                colormap: str = 'viridis',
                                alpha: float = DEFAULT_ALPHA,
                                interpolation: str = DEFAULT_INTERPOLATION) -> QPixmap:
        """
        创建无色条的精确尺寸图像（推荐用于背景）
        
        Args:
            data_array: 数据数组
            scene_width: 场景宽度
            scene_height: 场景高度
            colormap: 颜色映射
            alpha: 透明度
            interpolation: 插值方法
            
        Returns:
            QPixmap: 精确匹配场景尺寸的图像
        """
        return QtMatplotlibIntegration.create_qt_compatible_image(
            data_array, scene_width, scene_height,
            colormap=colormap, alpha=alpha, interpolation=interpolation,
            show_colorbar=False
        )
    
    @staticmethod
    def add_image_to_scene(scene: QGraphicsScene, 
                          pixmap: QPixmap, 
                          position: Tuple[float, float] = (0, 0),
                          z_value: float = -1,
                          replace_existing: Optional[QGraphicsPixmapItem] = None) -> QGraphicsPixmapItem:
        """
        将图像添加到Qt场景的标准方法
        
        Args:
            scene: Qt图形场景
            pixmap: 要添加的图像
            position: 放置位置 (x, y)
            z_value: 层级值（负值在底层）
            replace_existing: 如果提供，将替换现有的图像项
            
        Returns:
            QGraphicsPixmapItem: 创建的图像项
        """
        # 移除现有项（如果指定）
        if replace_existing is not None:
            scene.removeItem(replace_existing)
        
        # 创建新的图像项
        pixmap_item = QGraphicsPixmapItem(pixmap)
        pixmap_item.setPos(QPointF(position[0], position[1]))
        pixmap_item.setZValue(z_value)
        
        # 添加到场景
        scene.addItem(pixmap_item)
        
        return pixmap_item
    
    @staticmethod
    def coordinate_transform_matplotlib_to_qt(matplotlib_y: float, scene_height: float) -> float:
        """
        Matplotlib坐标到Qt坐标的Y轴转换
        
        Args:
            matplotlib_y: Matplotlib的Y坐标（底部原点）
            scene_height: 场景高度
            
        Returns:
            float: Qt的Y坐标（顶部原点）
        """
        return scene_height - matplotlib_y
    
    @staticmethod
    def coordinate_transform_qt_to_matplotlib(qt_y: float, scene_height: float) -> float:
        """
        Qt坐标到Matplotlib坐标的Y轴转换
        
        Args:
            qt_y: Qt的Y坐标（顶部原点）
            scene_height: 场景高度
            
        Returns:
            float: Matplotlib的Y坐标（底部原点）
        """
        return scene_height - qt_y
    
    @staticmethod
    def _fig_to_pixmap(fig: plt.Figure, dpi: int) -> QPixmap:
        """
        将matplotlib图形转换为QPixmap
        
        Args:
            fig: matplotlib图形对象
            dpi: 图像DPI
            
        Returns:
            QPixmap: 转换后的Qt图像
        """
        # 保存为字节流
        buf = io.BytesIO()
        fig.savefig(buf, 
                   format='png', 
                   dpi=dpi, 
                   bbox_inches=None,  # 不使用tight，保持精确尺寸
                   pad_inches=0,      # 零边距
                   transparent=True,  # 透明背景
                   facecolor='none')
        buf.seek(0)
        
        # 转换为QPixmap
        qimg = QImage()
        if not qimg.loadFromData(buf.getvalue()):
            raise RuntimeError("Failed to convert matplotlib figure to QImage")
        
        pixmap = QPixmap.fromImage(qimg)
        return pixmap
    
    @staticmethod
    def verify_image_dimensions(pixmap: QPixmap, 
                               expected_width: float, 
                               expected_height: float,
                               tolerance: int = 5) -> bool:
        """
        验证图像尺寸是否符合预期
        
        Args:
            pixmap: 要验证的图像
            expected_width: 期望宽度
            expected_height: 期望高度
            tolerance: 允许的像素误差
            
        Returns:
            bool: 尺寸是否符合预期
        """
        actual_width = pixmap.width()
        actual_height = pixmap.height()
        
        width_ok = abs(actual_width - expected_width) <= tolerance
        height_ok = abs(actual_height - expected_height) <= tolerance
        
        return width_ok and height_ok


class ImageIntegrationConfig:
    """图像集成配置常量"""
    
    # 默认DPI设置
    DEFAULT_DPI = 100
    HIGH_QUALITY_DPI = 150
    
    # 默认透明度
    SDF_ALPHA = 0.6
    TEMPERATURE_ALPHA = 0.8
    
    # 默认插值方法
    SDF_INTERPOLATION = 'nearest'
    TEMPERATURE_INTERPOLATION = 'nearest'
    
    # 默认色彩映射
    SDF_COLORMAP = 'coolwarm'
    TEMPERATURE_COLORMAP = 'plasma'
    
    # 层级值
    BACKGROUND_Z_VALUE = -1
    OVERLAY_Z_VALUE = 1
    
    # ColorBar配置
    COLORBAR_FRACTION = 0.15  # ColorBar宽度比例
    COLORBAR_PAD = 0.02       # ColorBar间距
    COLORBAR_SHRINK = 0.8     # ColorBar高度比例


# 便利函数
def create_sdf_background(sdf_array: np.ndarray, 
                         scene_width: float, 
                         scene_height: float) -> QPixmap:
    """创建SDF背景图像（便利函数）"""
    return QtMatplotlibIntegration.create_no_colorbar_image(
        sdf_array, scene_width, scene_height,
        colormap=ImageIntegrationConfig.SDF_COLORMAP,
        alpha=ImageIntegrationConfig.SDF_ALPHA,
        interpolation=ImageIntegrationConfig.SDF_INTERPOLATION
    )


def create_temperature_background(temp_array: np.ndarray,
                                 scene_width: float,
                                 scene_height: float) -> QPixmap:
    """创建温度场背景图像（便利函数）"""
    return QtMatplotlibIntegration.create_no_colorbar_image(
        temp_array, scene_width, scene_height,
        colormap=ImageIntegrationConfig.TEMPERATURE_COLORMAP,
        alpha=ImageIntegrationConfig.TEMPERATURE_ALPHA,
        interpolation=ImageIntegrationConfig.TEMPERATURE_INTERPOLATION
    )
