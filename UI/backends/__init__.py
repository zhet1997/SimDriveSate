"""
计算后端模块
提供统一的计算接口，支持热仿真、应力分析、神经网络等多种计算方式
"""

from .base_backend import ComputationBackendV2, FieldType, ComputationResult
from .thermal_backend import ThermalSimulationBackend

# 尝试导入POD后端
try:
    from .pod_temperature_backend import PODTemperatureBackend
    POD_BACKEND_AVAILABLE = True
except ImportError:
    PODTemperatureBackend = None
    POD_BACKEND_AVAILABLE = False

__all__ = [
    'ComputationBackendV2',
    'FieldType', 
    'ComputationResult',
    'ThermalSimulationBackend'
]

if POD_BACKEND_AVAILABLE:
    __all__.append('PODTemperatureBackend')
