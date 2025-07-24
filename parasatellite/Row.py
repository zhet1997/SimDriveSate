import numpy as np
from abc import ABC, abstractmethod
from typing import Union

from .Section import *
from .Cascade import GenericCascade
from . import _utils 
from ._bspline_methods.approx_periodic import *

DEG = np.pi / 180


__all__ = [
    'TurbineLegacyParamsRow'
]


class GenericRow(ABC):
    @abstractmethod
    def __call__(self) -> tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def get_dict(self) -> dict:
        pass


class TurbineLegacyParamsRow(GenericRow):
    r'''轴流式涡轮几何参数化方法叶栅
    '''
    def __init__(self,
                 repetition: int,
                 sections: list[TurbineLegacyParamsSection],
                 reverse: bool,
                 alignment: str,
                 axial_location: float,
                 range: tuple[float, float],
                 pitch_cascade: GenericCascade,
                 axial_cascade: GenericCascade,
                 radial_interp: int = 11):

        super().__init__()
        self.repetition = repetition
        self.sections = sections
        self.reverse = reverse
        self.alignment = alignment
        self.axial_location = axial_location
        self.range = range
        self.radial_interp = radial_interp
        self.pitch_cascade = pitch_cascade
        self.axial_cascade = axial_cascade

        assert self.range[0] < self.range[1]
        assert self.alignment in ['le', 'te', 'centroid']
        self._sort_sections()


    def _sort_sections(self):
        r'''对各截面按半径排序'''
        r = np.array([i.get_spanwise_location() for i in self.sections])
        idx = np.argsort(r)
        self.sections = [self.sections[i] for i in idx]


    def get_dict(self) -> dict:
        d = {
            'reverse': self.reverse,
            'alignment': self.alignment,
            'axial_location': self.axial_location,
            'repetition': self.repetition,
            'range_min': self.range[0],
            'range_max': self.range[1],
            'radial_interp': self.radial_interp,
            'pitch_cascade': self.pitch_cascade.get_dict(),
            'axial_cascade': self.axial_cascade.get_dict(),
            'sections': [i.get_dict() for i in self.sections]
        }
        return d


    def __call__(self):
        
        assert len(self.sections) > 1 and len(self.sections) < 4

        target_v = np.linspace(0, 1, self.radial_interp)
        delta_z = self.axial_cascade(target_v) * (self.range[1] - self.range[0])
        delta_s = self.pitch_cascade(target_v) * (self.range[1] - self.range[0])

        # 参考型线
        x_ss = np.zeros((len(self.sections), 112, 3))
        x_ps = np.zeros((len(self.sections), 112, 3))
        for i in range(len(self.sections)):
            _ss, _ps = self.sections[i]()
            x_ss[i] = np.pad(_ss, [[0, 0], [0, 1]], mode='constant', constant_values=self.sections[i].radius)
            x_ps[i] = np.pad(_ps, [[0, 0], [0, 1]], mode='constant', constant_values=self.sections[i].radius)


        # 叠加积叠位置基准与轴向位置
        if self.alignment == 'te':
            _, centroid_s0 = _utils.centroid(np.concatenate((np.flip(x_ss[0], axis=0), x_ps[0]), axis=0))
            for i in range(len(self.sections)):
                x_ss[i, :, 0] += self.axial_location
                x_ps[i, :, 0] += self.axial_location
                x_ss[i, :, 1] -= centroid_s0
                x_ps[i, :, 1] -= centroid_s0

        elif self.alignment == 'centroid':
            centroid_z = np.zeros(len(self.sections))
            centroid_s = np.zeros(len(self.sections))
            for i in range(len(self.sections)):
                centroid_z[i], centroid_s[i] = _utils.centroid(np.concatenate((np.flip(x_ss[i], axis=0), x_ps[i]), axis=0))
            for i in range(len(self.sections)):
                x_ss[i, :, 0] += self.axial_location - centroid_z[i]
                x_ps[i, :, 0] += self.axial_location - centroid_z[i]
                x_ss[i, :, 1] -= centroid_s[i]
                x_ps[i, :, 1] -= centroid_s[i]
        
        else: # le
            le_z = np.zeros(len(self.sections))
            le_s = np.zeros(len(self.sections))
            for i in range(len(self.sections)):
                le_z[i] = - (self.sections[i].cx - self.sections[i].r_le - self.sections[i].r_te)
                le_s[i] = - le_z[i] * (np.tan(self.sections[i].gamma * DEG) if self.sections[i].is_deg else np.tan(self.sections[i].gamma))
            for i in range(len(self.sections)):
                x_ss[i, :, 0] += self.axial_location - le_z[i]
                x_ps[i, :, 0] += self.axial_location - le_z[i]
                x_ss[i, :, 1] -= le_s[i]
                x_ps[i, :, 1] -= le_s[i]

        # 蒙皮法
        pbs_list = [None] * len(self.sections)
        for i in range(len(self.sections)):
            pbs_list[i] = make_pbs_from_turbine_legacy_params_section(
                x_ss[i], x_ps[i]
            )
        
        rlist = np.array([i.radius for i in self.sections])
        tv = (rlist - self.range[0]) / (self.range[1] - self.range[0])
        bsurf = make_rbbs_from_pbs_list(pbs_list, tv)

        # 取点并补充积叠线偏置

        u_probe_ss = np.r_[
            np.linspace(0, bsurf.avg_s_ratio_ss[0], 16),
            np.linspace(bsurf.avg_s_ratio_ss[0], bsurf.avg_s_ratio_ss[1], 82)[1:-1],
            np.linspace(bsurf.avg_s_ratio_ss[1], 1, 16)
        ]
        u_probe_ss = np.flip(u_probe_ss)

        u_probe_ps = np.r_[
            np.linspace(1, 1 + bsurf.avg_s_ratio_ps[0], 16),
            np.linspace(1 + bsurf.avg_s_ratio_ps[0], 1 + bsurf.avg_s_ratio_ps[1], 82)[1:-1],
            np.linspace(1 + bsurf.avg_s_ratio_ps[1], 2, 16)
        ]

        # 周向偏置以吸力面方向为正，与叶片摆放方式无关
        x_smooth_ss = np.zeros((self.radial_interp, 112, 3))
        x_smooth_ps = np.zeros((self.radial_interp, 112, 3))
        for i in range(self.radial_interp):
            x_smooth_ss[i] = bsurf.eval_u(target_v[i], u_probe_ss)
            x_smooth_ps[i] = bsurf.eval_u(target_v[i], u_probe_ps)
            x_smooth_ss[i, :, 0] += delta_z[i]
            x_smooth_ps[i, :, 0] += delta_z[i]
            x_smooth_ss[i, :, 1] += delta_s[i]
            x_smooth_ps[i, :, 1] += delta_s[i]

        # 反向
        x_smooth_ss[:, :, 1] *= (-1 if self.reverse else 1)
        x_smooth_ps[:, :, 1] *= (-1 if self.reverse else 1)

        return x_smooth_ss, x_smooth_ps

