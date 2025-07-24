import paraturbo.Section
import yaml
import numpy as np
from paraturbo._io_methods.tecplot import line_zsr
from paraturbo._bspline_methods.approx_periodic import make_pbs_from_turbine_legacy_params_section

def test_11params():
    with open('paraturbo/tests/resources/project.yaml', 'r') as f:
        p = yaml.safe_load(f)
    s = paraturbo.Section.TurbineLegacyParamsSection(**p['rows'][1]['sections'][2])
    x_raw_ss, x_raw_ps = s()

    x_aug_ss = np.pad(x_raw_ss, [[0, 0], [0, 1]], mode='constant', constant_values=s.radius)
    x_aug_ps = np.pad(x_raw_ps, [[0, 0], [0, 1]], mode='constant', constant_values=s.radius)
    
    bspl = make_pbs_from_turbine_legacy_params_section(
        x_aug_ss, x_aug_ps
    )

    u_probe_ss = np.r_[
        np.linspace(0, bspl.s_ratio_ss[0], 16),
        np.linspace(bspl.s_ratio_ss[0], bspl.s_ratio_ss[1], 82)[1:-1],
        np.linspace(bspl.s_ratio_ss[1], 1, 16)
    ]
    u_probe_ss = np.flip(u_probe_ss, axis=0)
    u_probe_ps = np.r_[
        np.linspace(1, 1 + bspl.s_ratio_ps[0], 16),
        np.linspace(1 + bspl.s_ratio_ps[0], 1 + bspl.s_ratio_ps[1], 82)[1:-1],
        np.linspace(1 + bspl.s_ratio_ps[1], 2, 16)
    ]
    x_smooth_ss = bspl(u_probe_ss)
    x_smooth_ps = bspl(u_probe_ps)

    x = np.concatenate(([x_smooth_ss], [x_smooth_ps]), axis=0)
    with open('profile_2.dat', 'w') as f:
        f.write(line_zsr(x))
