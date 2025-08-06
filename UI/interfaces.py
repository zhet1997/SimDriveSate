from abc import ABC, abstractmethod
import numpy as np


class ComputationBackend(ABC):
    """定义所有计算后端的通用接口"""
    
    @abstractmethod
    def compute(self, scene_data: list[dict], grid_shape: tuple[int, int]) -> np.ndarray:
        """
        根据场景数据，在指定形状的网格上进行计算。
        :param scene_data: 描述场景中所有元件的字典列表。
        :param grid_shape: 计算网格的形状 (height, width)。
        :return: 一个形状为 (height, width) 的浮点型NumPy数组。
        """
        pass