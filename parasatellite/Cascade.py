import numpy as np
from abc import ABC, abstractmethod
from typing import Union


__all__ = [
    'LinearCascade',
    'QuadraticCascade'
]


class GenericCascade(ABC):
    # 连带设计变量的常用积叠函数类
    @abstractmethod
    def __call__(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_dict(self) -> dict:
        pass


class LinearCascade(GenericCascade):
    r'''
    线性积叠，输入范围取0-1，输出按`Row`叶高归一化
    '''
    def __init__(self, tip_deviation: float, root_deviation: float) -> None:
        super().__init__()
        self.tip_deviation = tip_deviation
        self.root_deviation = root_deviation

    def __call__(self, h_norm: Union[float, list[float], np.ndarray]) -> Union[float, list[float], np.ndarray]:
        return self.root_deviation * (1 - h_norm) + self.tip_deviation * h_norm
    
    def get_dict(self) -> dict:
        d = {
            'cascade_method': 'linear',
            'tip_deviation': self.tip_deviation,
            'root_deviation': self.root_deviation
        }
        return d


class QuadraticCascade(GenericCascade):
    r'''
    二次曲线积叠（常用弯曲模式）
    '''
    def __init__(self, tip_deviation: float, mid_deviation: float, mid_location: float) -> None:
        super().__init__()
        self.tip_deviation = tip_deviation
        self.mid_deviation = mid_deviation
        self.mid_location = mid_location
        assert 0 < self.mid_location and self.mid_location < 1

    def __call__(self, h_norm: Union[float, list[float], np.ndarray]) -> Union[float, list[float], np.ndarray]:
        h0 = np.array([0, self.mid_location, 1])
        x0 = np.array([0, self.mid_deviation, self.tip_deviation])
        A = np.array([[h0[0]**2, h0[0], 1],
                      [h0[1]**2, h0[1], 1],
                      [h0[2]**2, h0[2], 1]])
        z = np.linalg.solve(A, x0)
        return np.polyval(z, h_norm)
    
    def get_dict(self) -> dict:
        d = {
            'cascade_method': 'quadratic',
            'tip_deviation': self.tip_deviation,
            'mid_deviation': self.mid_deviation,
            'mid_location': self.mid_location
        }
        return d