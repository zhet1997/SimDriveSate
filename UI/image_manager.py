"""
图像管理器 - 统一管理所有图像的计算、缓存和显示
"""

from typing import Dict, Optional, Any, Callable
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QPixmap
import numpy as np


class ImageManager(QObject):
    """图像管理器 - 计算与显示分离"""
    
    # 信号
    image_computed = pyqtSignal(str)  # image_type
    image_displayed = pyqtSignal(str)  # image_type
    display_cleared = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.image_cache: Dict[str, QPixmap] = {}  # 缓存计算结果
        self.current_display: Optional[str] = None  # 当前显示的图像类型
        self.compute_callbacks: Dict[str, Callable] = {}  # 计算回调函数
        self.scene = None  # 主场景引用
        self.current_background_item = None  # 当前背景图像项
        
        # 支持的图像类型
        self.supported_images = {
            'sdf': 'SDF场',
            'temperature': '温度场', 
            'voronoi': '泰森多边形'
        }
    
    def set_scene(self, scene):
        """设置主场景引用"""
        self.scene = scene
    
    def register_compute_callback(self, image_type: str, callback: Callable):
        """注册图像计算回调函数"""
        self.compute_callbacks[image_type] = callback
        print(f"[ImageManager] 注册 {image_type} 计算回调")
    
    def compute_image(self, image_type: str, component_data: Any = None) -> bool:
        """计算图像并缓存"""
        if image_type not in self.compute_callbacks:
            print(f"[ImageManager] 错误: 未找到 {image_type} 的计算回调")
            return False
        
        try:
            print(f"[ImageManager] 开始计算 {image_type} 图像")
            
            # 调用对应的计算回调
            callback = self.compute_callbacks[image_type]
            if component_data is not None:
                result = callback(component_data)
            else:
                result = callback()
            
            # 检查返回结果
            if isinstance(result, QPixmap):
                pixmap = result
            elif isinstance(result, np.ndarray):
                # 将numpy数组转换为QPixmap
                pixmap = self._numpy_to_pixmap(result)
            else:
                print(f"[ImageManager] 错误: {image_type} 计算结果格式不支持")
                return False
            
            # 缓存结果
            self.image_cache[image_type] = pixmap
            self.image_computed.emit(image_type)
            print(f"[ImageManager] {image_type} 图像计算完成并缓存")
            return True
            
        except Exception as e:
            print(f"[ImageManager] {image_type} 计算失败: {e}")
            return False
    
    def display_image(self, image_type: str) -> bool:
        """显示指定类型的图像"""
        if image_type not in self.image_cache:
            print(f"[ImageManager] 错误: {image_type} 图像未计算或缓存")
            return False
        
        if not self.scene:
            print(f"[ImageManager] 错误: 场景未设置")
            return False
        
        try:
            # 清除当前显示
            self._clear_current_display()
            
            # 显示新图像
            pixmap = self.image_cache[image_type]
            from image_integration_interface import QtMatplotlibIntegration
            
            self.current_background_item = QtMatplotlibIntegration.add_image_to_scene(
                self.scene, pixmap, 
                position=(0, 0),
                z_value=-10,  # 背景层
                replace_existing=self.current_background_item
            )
            
            self.current_display = image_type
            self.image_displayed.emit(image_type)
            print(f"[ImageManager] 显示 {image_type} 图像")
            return True
            
        except Exception as e:
            print(f"[ImageManager] 显示 {image_type} 失败: {e}")
            return False
    
    def clear_display(self):
        """清除当前显示"""
        self._clear_current_display()
        self.current_display = None
        self.display_cleared.emit()
        print(f"[ImageManager] 清除图像显示")
    
    def _clear_current_display(self):
        """内部方法：清除当前显示的背景图像"""
        if self.current_background_item and self.scene:
            try:
                if self.current_background_item.scene() is not None:
                    self.scene.removeItem(self.current_background_item)
                self.current_background_item = None
            except RuntimeError:
                # C++对象已删除
                self.current_background_item = None
    
    def _numpy_to_pixmap(self, array: np.ndarray) -> QPixmap:
        """将numpy数组转换为QPixmap"""
        from PyQt6.QtGui import QImage
        
        # 假设array是灰度图像，范围0-1
        if array.dtype != np.uint8:
            array = (array * 255).astype(np.uint8)
        
        if len(array.shape) == 2:
            # 灰度图像
            height, width = array.shape
            bytes_per_line = width
            q_image = QImage(array.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        else:
            # RGB图像
            height, width, channels = array.shape
            bytes_per_line = width * channels
            if channels == 3:
                q_image = QImage(array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            elif channels == 4:
                q_image = QImage(array.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888)
            else:
                raise ValueError(f"不支持的通道数: {channels}")
        
        return QPixmap.fromImage(q_image)
    
    def get_cached_images(self) -> list:
        """获取已缓存的图像类型列表"""
        return list(self.image_cache.keys())
    
    def is_cached(self, image_type: str) -> bool:
        """检查图像是否已缓存"""
        return image_type in self.image_cache
    
    def get_current_display(self) -> Optional[str]:
        """获取当前显示的图像类型"""
        return self.current_display
    
    def clear_cache(self, image_type: str = None):
        """清除缓存"""
        if image_type is None:
            self.image_cache.clear()
            print(f"[ImageManager] 清除所有图像缓存")
        elif image_type in self.image_cache:
            del self.image_cache[image_type]
            print(f"[ImageManager] 清除 {image_type} 图像缓存")


# 全局单例
_image_manager_instance = None

def get_image_manager() -> ImageManager:
    """获取图像管理器单例"""
    global _image_manager_instance
    if _image_manager_instance is None:
        _image_manager_instance = ImageManager()
    return _image_manager_instance
