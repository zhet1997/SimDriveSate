"""
后台计算线程
处理SDF计算等耗时操作，避免阻塞UI线程
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from interfaces import ComputationBackend
from sdf_backend import SDFBackend


class Worker(QObject):
    """后台计算工作线程"""
    
    # 定义信号
    computation_complete = pyqtSignal(np.ndarray)  # 发送计算结果
    
    def __init__(self):
        super().__init__()
        # 初始化后端计算模块
        self.backend: Optional[ComputationBackend] = SDFBackend()  # 默认使用SDF后端
        
    def set_backend(self, backend: ComputationBackend):
        """设置计算后端"""
        self.backend = backend
        
    def compute(self, scene_data: List[Dict], grid_shape: Tuple[int, int]):
        """执行计算并发送结果"""
        if self.backend is not None:
            result = self.backend.compute(scene_data, grid_shape)
            self.computation_complete.emit(result)