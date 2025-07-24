import numpy as np
from scipy.interpolate import BSpline
from abc import ABC, abstractmethod
from typing import Union
from copy import deepcopy

from ._bspline_methods.approx_non_periodic import approx_2d_non_periodic

__all__ = [
    'RawFlowpath',
    'BSplineFlowpath'
]

class GenericFlowpath(ABC):
    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def get_dict(self) -> dict:
        pass


class RawFlowpath(GenericFlowpath):
    def __init__(self, 
                 fp_ref: Union[list, np.ndarray, None]):
        super().__init__()
        self.fp_ref = fp_ref
        if type(self.fp_ref) == list:
            self.fp_ref = np.array(self.fp_ref, dtype=float)

    def __call__(self) -> np.ndarray:
        return self.fp_ref

    def get_dict(self) -> dict:
        return {'flowpath_method': 'raw_flowpath'}


class BSplineFlowpath(GenericFlowpath):
    r'''B样条轴对称流道

    定义在Z-R坐标系的三次B样条，端点满足 clamped 特性
    
    参数
    --------
    fp_ref : ArrayLike, shape (n, 2), optional
        参考流道
    fp_bspl : BSpline, optional
        对参考流道拟合的B样条
    n_interp : int = 128
        生成时插值点个数
    n_control_points: int = 6
        除首末点外的控制点个数
    param_method : 'centripetal or 'chordal' or 'uniform'
        B样条参数排布方式：向心/弦长/均匀参数化，默认为向心参数化

    方法
    --------
    __call__
    fit
    export

    备注
    --------
    fp_bspl 允许在定义时直接注入，该特性允许强行定义高阶流道、非均匀节点流道等操作。
    但程序不会对其维度、节点范围、clamped 特性等进行检查，
    也不保证拟合时能对其他型线得到类似产物；
    n_control_points 和 param_method 在此时与 fp_bspl 亦无关。
    '''

    def __init__(self, 
                 fp_ref: Union[list, np.ndarray, None] = None,
                 fp_bspl: Union[BSpline, None] = None,
                 n_interp: int = 128,
                 n_control_points: int = 6, 
                 param_method: str = 'centripetal',
                 deviation: Union[list, np.ndarray, None] = None):
        super().__init__()
        self.fp_ref = fp_ref
        if type(self.fp_ref) == list:
            self.fp_ref = np.array(self.fp_ref, dtype=float)

        self.n_interp = n_interp
        self.n_control_points = n_control_points
        self.param_method = param_method

        # fill items
        self.fp_bspl = fp_bspl
        if (self.fp_ref is None) and (self.fp_bspl is not None):
            self.fp_ref = self.export()
        if (self.fp_bspl is None) and (self.fp_ref is not None):
            self.fit(self.fp_ref)

        # 流道控制变量
        self.deviation = deviation
        if self.deviation is None:
            self.deviation = np.zeros(self.n_control_points)
        elif type(self.deviation) == list:
            self.deviation = np.array(self.deviation, dtype=float)
        assert len(self.deviation) == self.n_control_points


    def get_dict(self) -> dict:
        d = {
            'flowpath_method': 'bspline_flowpath',
            'n_interp': self.n_interp,
            'n_control_points': self.n_control_points,
            'param_method': self.param_method,
            'deviation': self.deviation.tolist()
        }
        return d


    def export(self) -> np.ndarray:
        r'''
        按 fp_bspl 及控制点偏移量采样
        '''
        u_interp = np.linspace(0, 1, self.n_interp)
        
        # 控制点偏移
        cv = self.deviation * np.hypot(self.fp_bspl.c[0, 0] - self.fp_bspl.c[-1, 0],
                                       self.fp_bspl.c[0, 1] - self.fp_bspl.c[-1, 1])
        d_c_spl = self.fp_bspl.c[2:] - self.fp_bspl.c[:-2]
        mag_d_c_spl = np.hypot(d_c_spl[:, 0], d_c_spl[:, 1])
        tmp_bspl = deepcopy(self.fp_bspl)
        tmp_bspl.c[1:-1, 0] += cv * d_c_spl[:, 1] * -1 / mag_d_c_spl
        tmp_bspl.c[1:-1, 1] += cv * d_c_spl[:, 0] / mag_d_c_spl

        return tmp_bspl(u_interp)


    def fit(self,
            fp_ref: Union[list, np.ndarray],
            n_control_points: int = None, 
            param_method: str = None
            ) -> None:
        r'''
        对输入的参考曲线进行三次 clamped B样条拟合
        '''
        if n_control_points is not None:
            self.n_control_points = n_control_points
        if param_method is not None:
            self.param_method = param_method

        self.fp_ref = fp_ref
        if type(self.fp_ref) == list:
            self.fp_ref = np.array(self.fp_ref, dtype=float)
        
        self.fp_bspl = approx_2d_non_periodic(self.fp_ref,
                                              self.n_control_points,
                                              self.param_method)


    def __call__(self, mode='fit', n_interp: int = None) -> np.ndarray:
        r'''
        导出参考曲线或按 fp_bspl 采样
        '''
        if n_interp is not None:
            self.n_interp = n_interp
        assert mode == 'fit' or mode == 'raw'
        if mode == 'raw':
            return self.fp_ref
        else:
            return self.export()
        