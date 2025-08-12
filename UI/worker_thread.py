"""
后台计算线程
处理SDF计算等耗时操作，避免阻塞UI线程
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from interfaces import ComputationBackend
from sdf_backend import SDFBackend
from temperature_model_interface import TemperatureReconstructionModel, create_temperature_model


class Worker(QObject):
    """后台计算工作线程"""
    
    # 定义信号
    computation_complete = pyqtSignal(np.ndarray)  # 发送SDF计算结果
    temperature_reconstruction_complete = pyqtSignal(np.ndarray)  # 发送温度重构结果
    
    def __init__(self):
        super().__init__()
        # 初始化后端计算模块
        self.backend: Optional[ComputationBackend] = SDFBackend()  # 默认使用SDF后端
        # 初始化温度重构模型
        self.temperature_model: TemperatureReconstructionModel = create_temperature_model("voronoi")
        
    def set_backend(self, backend: ComputationBackend):
        """设置计算后端"""
        self.backend = backend
    
    def set_temperature_model(self, model_type: str):
        """设置温度重构模型"""
        self.temperature_model = create_temperature_model(model_type)
        
    def compute(self, scene_data: List[Dict], grid_shape: Tuple[int, int]):
        """执行SDF计算并发送结果"""
        if self.backend is not None:
            result = self.backend.compute(scene_data, grid_shape)
            self.computation_complete.emit(result)
    
    def compute_temperature_reconstruction(self, sensor_data: List[Dict], grid_shape: Tuple[int, int]):
        """执行温度场重构计算"""
        try:
            print(f"Using {self.temperature_model.get_model_name()} for temperature reconstruction")
            print(f"Sensor data: {len(sensor_data)} sensors")
            
            # 使用配置的温度重构模型
            result = self.temperature_model.reconstruct_temperature_field(sensor_data, grid_shape)
            self.temperature_reconstruction_complete.emit(result)
        except Exception as e:
            print(f"Temperature reconstruction failed: {e}")
            print(f"Error details: {type(e).__name__}: {str(e)}")
            print(f"Sensor data received: {sensor_data}")
            print(f"Grid shape: {grid_shape}")
            import traceback
            traceback.print_exc()
            # 返回一个默认的温度场
            default_result = np.full(grid_shape, 0.0)  # 0K的均匀温度场
            self.temperature_reconstruction_complete.emit(default_result)
    
