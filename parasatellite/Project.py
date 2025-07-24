import numpy as np
import yaml
import os, shutil

from .Row import GenericRow
from .Flowpath import GenericFlowpath
from ._io_methods import geomturbo


class Project():
    def __init__(self,
                 rows: list[GenericRow],
                 row_name: list[str],
                 hub_fp: GenericFlowpath,
                 shroud_fp: GenericFlowpath) -> None:
        self.rows = rows
        self.row_name = row_name
        self.hub_fp = hub_fp
        self.shroud_fp = shroud_fp
        assert len(self.rows) > 0 and len(self.rows) == len(self.row_name)


    def export(self, path: str, fmt='geomturbo') -> None:
        if fmt=='geomturbo':
            f = open(path, 'w')
            f.write(geomturbo.header())
            f.write(geomturbo.flowpath(self.hub_fp(), self.shroud_fp()))
            for i in range(len(self.rows)):
                f.write(geomturbo.row_header(self.row_name[i], self.rows[i].repetition))
                X_ss, X_ps = self.rows[i]()
                f.write(geomturbo.row_data(X_ss, 'suction'))
                f.write(geomturbo.row_data(X_ps, 'pressure'))
                f.write(geomturbo.row_tail())
            f.close()
        else:
            raise NotImplementedError
        
    
    def save(self, path: str, fmt='yaml_ext') -> None:
        if fmt=='yaml_ext':
            d = {
                'hub_flowpath': self.hub_fp.get_dict(),
                'shroud_flowpath': self.shroud_fp.get_dict(),
                'row_name': self.row_name,
                'rows': [i.get_dict() for i in self.rows]
            }
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
            # 除端壁参考线外数据
            with open(path + '/project.yaml', 'w') as f:
                yaml.dump(d, f, sort_keys=False, default_flow_style=None)
            # 端壁参考线
            np.savetxt(path + '/fp_hub.dat', self.hub_fp.fp_ref)
            np.savetxt(path + '/fp_shroud.dat', self.shroud_fp.fp_ref)
        else:
            raise NotImplementedError