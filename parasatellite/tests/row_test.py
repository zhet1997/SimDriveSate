import paraturbo
import yaml
import numpy as np
from paraturbo._io_methods.tecplot import line_zsr


def test_11params():
    with open('paraturbo/tests/resources/project.yaml', 'r') as f:
        p = yaml.safe_load(f)
    s1c0 = paraturbo.TurbineLegacyParamsSection(**p['rows'][0]['sections'][0])
    s1c1 = paraturbo.TurbineLegacyParamsSection(**p['rows'][0]['sections'][1])
    s1c2 = paraturbo.TurbineLegacyParamsSection(**p['rows'][0]['sections'][2])

    dz = paraturbo.LinearCascade(tip_deviation=p['rows'][0]['axial_cascade']['tip_deviation'], 
                                 root_deviation=p['rows'][0]['axial_cascade']['root_deviation'])
    ds = paraturbo.LinearCascade(tip_deviation=p['rows'][0]['pitch_cascade']['tip_deviation'], 
                                 root_deviation=p['rows'][0]['pitch_cascade']['root_deviation'])

    stator_1 = paraturbo.TurbineLegacyParamsRow(
        p['rows'][0]['repetition'], 
        [s1c0, s1c1, s1c2], 
        reverse=p['rows'][0]['reverse'], 
        alignment=p['rows'][0]['alignment'], 
        axial_location=p['rows'][0]['axial_location'], 
        range=(p['rows'][0]['range_min'], p['rows'][0]['range_max']), 
        pitch_cascade=ds, axial_cascade=dz
    )

    x_ss, x_ps = stator_1()
    with open('profile_row.dat', 'w') as f:
        f.write(line_zsr(x_ss))
        f.write(line_zsr(x_ps))

