import numpy as np
from abc import ABC, abstractmethod
from typing import Union

from ._param_methods.axial_turbine import legacy_11_params

DEG = np.pi / 180

__all__ = [
    'Raw2DSection',
    'Raw3DSection',
    'TurbineLegacyParamsSection'
]

class Generic2DSection(ABC):
    r'''广义二维截面抽象类
    必须可获取径向位置（截面半径、截面高度、相对叶高等）
    '''
    @abstractmethod
    def __call__(self) -> tuple[np.ndarray, np.ndarray]:
        pass
    
    @abstractmethod
    def get_spanwise_location(self) -> float:
        pass

    @abstractmethod
    def get_dict(self) -> dict:
        pass


class Generic3DSection(ABC):
    r'''广义三维截面抽象类
    纯粹的闭合空间曲线
    '''
    @abstractmethod
    def __call__(self) -> tuple[np.ndarray, np.ndarray]:
        pass
    
    @abstractmethod
    def get_dict(self) -> dict:
        pass


class Raw2DSection(Generic2DSection):
    r'''任意散点二维截面

    定义在Z-S面上，吸/压力面首尾相交，径向位置由半径绝对值表示
    '''
    def __init__(self, 
                 ss_ref: Union[list, np.ndarray],
                 ps_ref: Union[list, np.ndarray],
                 radius: float):
        super().__init__()
        self.ss_ref = ss_ref
        self.ps_ref = ps_ref
        self.radius = radius
        if type(self.ss_ref) == list:
            self.ss_ref = np.array(self.ss_ref, dtype=float)
        if type(self.ps_ref) == list:
            self.ps_ref = np.array(self.ps_ref, dtype=float)

    def __call__(self) -> tuple[np.ndarray, np.ndarray]:
        return self.ss_ref, self.ps_ref
    
    def get_spanwise_location(self) -> float:
        return self.radius
    
    def get_dict(self) -> dict:
        raise NotImplementedError


class Raw3DSection(Generic3DSection):
    r'''任意散点三维截面

    按S-R-Z格式定义，吸/压力面首尾相交
    '''
    def __init__(self, 
                 ss_ref: Union[list, np.ndarray],
                 ps_ref: Union[list, np.ndarray]):
        super().__init__()
        self.ss_ref = ss_ref
        self.ps_ref = ps_ref
        if type(self.ss_ref) == list:
            self.ss_ref = np.array(self.ss_ref, dtype=float)
        if type(self.ps_ref) == list:
            self.ps_ref = np.array(self.ps_ref, dtype=float)

    def __call__(self) -> tuple[np.ndarray, np.ndarray]:
        return self.ss_ref, self.ps_ref
    
    def get_dict(self) -> dict:
        raise NotImplementedError


class TurbineLegacyParamsSection(Generic2DSection):
    r'''通用轴流式涡轮几何参数化方法截面

    定义在Z-S面上，吸/压力面首尾相交，径向位置由半径绝对值表示
    '''
    def __init__(self, 
                 radius: float, repetition: int, 
                 cx: float, r_le: float, r_te: float,
                 gamma: float, beta_in: float, wedge_is: float, wedge_ip: float, s_factor: float,
                 eff_beta_out: float, deflect_out: float, corr: float, wedge_out: float,
                 sl1: float, sl2: float, st1: float, st2: float, p1: float, p2: float,
                 is_deg: bool = True,
                 **kwargs):
        r'''
        定义几何参数；**kwargs不做处理
        '''
        self.radius = radius
        self.repetition = repetition
        self.is_deg = is_deg

        self.cx = cx        # 轴向弦长
        self.r_le = r_le    # 前缘半径
        self.r_te = r_te    # 尾缘半径
        self.gamma = gamma  # 圆心连线角
        self.beta_in = beta_in # 进口气流角
        self.wedge_is = wedge_is  # 进口上楔角
        self.wedge_ip = wedge_ip # 进口下楔角
        self.s_factor = s_factor # 前缘修正，不支持椭前缘
        self.eff_beta_out = eff_beta_out    # 有效出气角
        self.deflect_out = deflect_out      # 出口偏转角
        self.corr = corr    # 关联系数
        self.wedge_out = wedge_out  # 出口楔角
        self.sl1 = sl1      # 贝塞尔曲线控制点
        self.sl2 = sl2
        self.st1 = st1
        self.st2 = st2
        self.p1 = p1
        self.p2 = p2


    def __call__(self) -> tuple[np.ndarray, np.ndarray]:
        # 输出SS/PS相对坐标
        t = 2 * np.pi * self.radius / self.repetition
        if self.is_deg:
            return legacy_11_params(self.cx, t, self.r_le, self.r_te, 
                                    self.gamma * DEG, 
                                    self.beta_in * DEG, 
                                    self.wedge_is * DEG, 
                                    self.wedge_ip * DEG, 
                                    self.s_factor,
                                    self.eff_beta_out * DEG, 
                                    self.deflect_out * DEG, 
                                    self.corr, 
                                    self.wedge_out * DEG,
                                    self.sl1, self.sl2, self.st1, self.st2, self.p1, self.p2)
        else:
            return legacy_11_params(self.cx, t, self.r_le, self.r_te, 
                                    self.gamma, self.beta_in, self.wedge_is, self.wedge_ip, self.s_factor,
                                    self.eff_beta_out, self.deflect_out, self.corr, self.wedge_out,
                                    self.sl1, self.sl2, self.st1, self.st2, self.p1, self.p2)


    def get_spanwise_location(self) -> float:
        return self.radius


    def get_dict(self) -> dict:
        d = {
            'section_method': 'axial_turbine',
            'is_deg': self.is_deg,
            'repetition': self.repetition,
            'radius': self.radius,
            'cx': self.cx,
            'r_le': self.r_le,
            'r_te': self.r_te,
            'gamma': self.gamma,
            'beta_in': self.beta_in,
            'wedge_is': self.wedge_is,
            'wedge_ip': self.wedge_ip,
            's_factor': self.s_factor,
            'eff_beta_out': self.eff_beta_out,
            'deflect_out': self.deflect_out,
            'corr': self.corr,
            'wedge_out': self.wedge_out,
            'sl1': self.sl1,
            'sl2': self.sl2,
            'st1': self.st1,
            'st2': self.st2,
            'p1': self.p1,
            'p2': self.p2
        }
        return d