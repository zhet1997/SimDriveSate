"""
计算后端模块
提供统一的计算接口，支持热仿真、应力分析、神经网络等多种计算方式
"""

from .base_backend import ComputationBackendV2, FieldType, ComputationResult
from .thermal_backend import ThermalSimulationBackend

__all__ = [
    'ComputationBackendV2',
    'FieldType', 
    'ComputationResult',
    'ThermalSimulationBackend'
]
