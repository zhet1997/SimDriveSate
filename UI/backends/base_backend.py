"""
基础计算后端接口
定义所有计算后端的统一接口规范，支持多种物理场计算
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from dataclasses import dataclass


class FieldType(Enum):
    """物理场类型枚举"""
    TEMPERATURE = "temperature"  # 温度场
    STRESS = "stress"            # 应力场
    SDF = "sdf"                  # 有符号距离场
    DISPLACEMENT = "displacement" # 位移场
    VORONOI = "voronoi"          # 泰森多边形
    CUSTOM = "custom"            # 自定义场


@dataclass
class ComputationResult:
    """计算结果数据结构"""
    field_data: np.ndarray           # 主要场数据 (H, W)
    field_type: FieldType            # 场类型
    metadata: Dict[str, Any]         # 元数据（单位、范围等）
    auxiliary_data: Optional[Dict[str, np.ndarray]] = None  # 辅助数据（梯度、通量等）
    computation_time: Optional[float] = None  # 计算时间（秒）
    error_info: Optional[str] = None          # 错误信息
    
    def is_valid(self) -> bool:
        """检查结果是否有效"""
        return self.field_data is not None and self.error_info is None


class ComputationBackendV2(ABC):
    """计算后端抽象基类 - 第二版本，更加通用和可扩展"""
    
    def __init__(self, name: str = "UnknownBackend"):
        self.name = name
        self._initialized = False
        
    @abstractmethod
    def get_supported_field_types(self) -> List[FieldType]:
        """返回此后端支持的物理场类型列表"""
        pass
    
    @abstractmethod
    def validate_input_data(self, input_data: Dict[str, Any]) -> Tuple[bool, str]:
        """验证输入数据的有效性
        
        Args:
            input_data: 输入数据字典，格式由具体后端定义
            
        Returns:
            (is_valid, error_message): 验证结果和错误信息
        """
        pass
    
    @abstractmethod
    def compute_field(self, 
                     input_data: Dict[str, Any], 
                     field_type: FieldType,
                     grid_shape: Tuple[int, int],
                     **kwargs) -> ComputationResult:
        """计算指定类型的物理场
        
        Args:
            input_data: 输入数据（组件布局、边界条件等）
            field_type: 要计算的场类型
            grid_shape: 网格形状 (height, width)
            **kwargs: 后端特定的计算参数
            
        Returns:
            ComputationResult: 计算结果
        """
        pass
    
    def get_backend_info(self) -> Dict[str, Any]:
        """获取后端信息"""
        return {
            "name": self.name,
            "supported_fields": [ft.value for ft in self.get_supported_field_types()],
            "initialized": self._initialized
        }
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化后端（可选）"""
        self._initialized = True
        return True
    
    def cleanup(self) -> None:
        """清理资源（可选）"""
        pass


class InputDataFormat:
    """标准输入数据格式定义"""
    
    @staticmethod
    def create_component_data(components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """创建标准的组件数据格式
        
        标准组件格式：
        {
            "id": int,
            "shape": str,  # "rect", "circle", "capsule"
            "center": [float, float],  # 中心坐标（米）
            "power": float,  # 功率（W）
            "properties": Dict[str, Any]  # 形状特定属性
        }
        """
        return {
            "components": components,
            "layout_domain": (0.2, 0.2),  # 默认布局域（米）
            "coordinate_system": "meters"  # 坐标系统
        }
    
    @staticmethod
    def create_sensor_data(sensor_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """创建标准的传感器数据格式
        
        传感器格式：
        {
            "position": [float, float],  # 位置（米）
            "value": float,              # 测量值
            "sensor_type": str,          # 传感器类型
            "uncertainty": float         # 测量不确定度（可选）
        }
        """
        return {
            "sensors": sensor_points,
            "measurement_type": "temperature",  # 默认测量类型
            "coordinate_system": "meters"
        }
    
    @staticmethod
    def create_boundary_conditions(boundary_specs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """创建边界条件数据格式"""
        return {
            "boundary_conditions": boundary_specs,
            "default_boundary_type": "dirichlet"
        }


# 向后兼容：保持原有的ComputationBackend接口
class ComputationBackend(ComputationBackendV2):
    """向后兼容的计算后端接口"""
    
    def compute(self, scene_data: list[dict], grid_shape: tuple[int, int]) -> np.ndarray:
        """原有的计算接口，为了向后兼容"""
        # 转换数据格式
        input_data = InputDataFormat.create_component_data(scene_data)
        
        # 调用新接口，默认计算SDF
        result = self.compute_field(input_data, FieldType.SDF, grid_shape)
        
        if result.is_valid():
            return result.field_data
        else:
            # 返回默认的SDF数据
            return np.full(grid_shape, float('inf'))
    
    @abstractmethod
    def get_supported_field_types(self) -> List[FieldType]:
        """子类必须实现"""
        pass
    
    @abstractmethod
    def validate_input_data(self, input_data: Dict[str, Any]) -> Tuple[bool, str]:
        """子类必须实现"""
        pass
    
    @abstractmethod
    def compute_field(self, input_data: Dict[str, Any], field_type: FieldType, 
                     grid_shape: Tuple[int, int], **kwargs) -> ComputationResult:
        """子类必须实现"""
        pass
