"""
温度场重构模型接口
定义温度重构算法的标准接口，支持不同算法的插件化实现
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import numpy as np


class TemperatureReconstructionModel(ABC):
    """温度场重构模型抽象基类"""
    
    @abstractmethod
    def reconstruct_temperature_field(self, sensor_data: List[Dict], grid_shape: Tuple[int, int]) -> np.ndarray:
        """
        重构温度场
        
        Args:
            sensor_data: 传感器数据列表，每个元素包含 {'position': (x, y), 'temperature': float}
            grid_shape: 输出网格形状 (height, width)
            
        Returns:
            np.ndarray: 重构的温度场数组，形状为 grid_shape
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """返回模型名称"""
        pass
    
    @abstractmethod
    def get_model_description(self) -> str:
        """返回模型描述"""
        pass
    
    def validate_sensor_data(self, sensor_data: List[Dict]) -> bool:
        """
        验证传感器数据的有效性
        
        Args:
            sensor_data: 传感器数据列表
            
        Returns:
            bool: 数据是否有效
        """
        if not sensor_data:
            return False
            
        for sensor in sensor_data:
            # 检查必要字段
            if 'position' not in sensor or 'temperature' not in sensor:
                return False
            
            # 检查位置数据
            position = sensor['position']
            if not isinstance(position, (list, tuple)) or len(position) != 2:
                return False
            
            # 检查温度数据
            temperature = sensor['temperature']
            if not isinstance(temperature, (int, float)) or temperature < 0:  # 温度不能低于0K
                return False
                
        return True


class VoronoiTemperatureModel(TemperatureReconstructionModel):
    """基于泰森多边形的温度场重构模型"""
    
    def __init__(self):
        self.model_name = "Voronoi Temperature Reconstruction"
        self.model_description = "使用泰森多边形算法进行温度场重构，每个区域的温度等于最近传感器的温度值"
    
    def reconstruct_temperature_field(self, sensor_data: List[Dict], grid_shape: Tuple[int, int]) -> np.ndarray:
        """
        使用泰森多边形算法重构温度场
        """
        from scipy.spatial import Voronoi, voronoi_plot_2d
        from scipy.spatial.distance import cdist
        
        if not self.validate_sensor_data(sensor_data):
            raise ValueError("Invalid sensor data")
        
        height, width = grid_shape
        
        # 提取传感器位置和温度
        sensor_positions = np.array([sensor['position'] for sensor in sensor_data])
        sensor_temperatures = np.array([sensor['temperature'] for sensor in sensor_data])
        
        # 如果只有一个传感器，整个区域使用该温度
        if len(sensor_positions) == 1:
            return np.full(grid_shape, sensor_temperatures[0])
        
        # 创建网格点
        x = np.linspace(0, width, width)
        y = np.linspace(0, height, height)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        
        # 计算每个网格点到各传感器的距离
        distances = cdist(grid_points, sensor_positions)
        
        # 找到每个网格点最近的传感器
        nearest_sensor_indices = np.argmin(distances, axis=1)
        
        # 分配温度值
        temperature_field = sensor_temperatures[nearest_sensor_indices]
        
        # 重塑为网格形状
        temperature_field = temperature_field.reshape(grid_shape)
        
        return temperature_field
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def get_model_description(self) -> str:
        return self.model_description


class IDWTemperatureModel(TemperatureReconstructionModel):
    """基于反距离加权插值的温度场重构模型（备用实现）"""
    
    def __init__(self, power: float = 2.0):
        self.power = power
        self.model_name = "IDW Temperature Reconstruction"
        self.model_description = f"使用反距离加权插值算法进行温度场重构，幂参数={power}"
    
    def reconstruct_temperature_field(self, sensor_data: List[Dict], grid_shape: Tuple[int, int]) -> np.ndarray:
        """
        使用IDW算法重构温度场
        """
        if not self.validate_sensor_data(sensor_data):
            raise ValueError("Invalid sensor data")
        
        height, width = grid_shape
        
        # 提取传感器位置和温度
        sensor_positions = np.array([sensor['position'] for sensor in sensor_data])
        sensor_temperatures = np.array([sensor['temperature'] for sensor in sensor_data])
        
        # 如果只有一个传感器，整个区域使用该温度
        if len(sensor_positions) == 1:
            return np.full(grid_shape, sensor_temperatures[0])
        
        # 创建网格
        x = np.linspace(0, width, width)
        y = np.linspace(0, height, height)
        xx, yy = np.meshgrid(x, y)
        
        temperature_field = np.zeros((height, width))
        
        for i in range(height):
            for j in range(width):
                point = np.array([xx[i, j], yy[i, j]])
                
                # 计算到各传感器的距离
                distances = np.linalg.norm(sensor_positions - point, axis=1)
                
                # 避免除零错误：如果某个点恰好与传感器重合
                if np.any(distances == 0):
                    idx = np.where(distances == 0)[0][0]
                    temperature_field[i, j] = sensor_temperatures[idx]
                else:
                    # IDW插值
                    weights = 1.0 / (distances ** self.power)
                    temperature_field[i, j] = np.sum(weights * sensor_temperatures) / np.sum(weights)
        
        return temperature_field
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def get_model_description(self) -> str:
        return self.model_description


# 模型工厂函数
def create_temperature_model(model_type: str = "voronoi") -> TemperatureReconstructionModel:
    """
    创建温度重构模型实例
    
    Args:
        model_type: 模型类型 ("voronoi", "idw")
        
    Returns:
        TemperatureReconstructionModel: 模型实例
    """
    model_type = model_type.lower()
    
    if model_type == "voronoi":
        return VoronoiTemperatureModel()
    elif model_type == "idw":
        return IDWTemperatureModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# 可用模型列表
AVAILABLE_MODELS = {
    "voronoi": "泰森多边形算法",
    "idw": "反距离加权插值算法"
}
