import paraturbo
import yaml
import numpy as np

def test_assemble():
    with open('paraturbo/tests/resources/project.yaml', 'r') as f:
        p = yaml.safe_load(f)
    s1c0 = paraturbo.TurbineLegacyParamsSection(**p['rows'][0]['sections'][0])
    s1c1 = paraturbo.TurbineLegacyParamsSection(**p['rows'][0]['sections'][1])
    s1c2 = paraturbo.TurbineLegacyParamsSection(**p['rows'][0]['sections'][2])

    dz1 = paraturbo.LinearCascade(tip_deviation=p['rows'][0]['axial_cascade']['tip_deviation'], 
                                 root_deviation=p['rows'][0]['axial_cascade']['root_deviation'])
    ds1 = paraturbo.LinearCascade(tip_deviation=p['rows'][0]['pitch_cascade']['tip_deviation'], 
                                 root_deviation=p['rows'][0]['pitch_cascade']['root_deviation'])

    stator_1 = paraturbo.TurbineLegacyParamsRow(
        p['rows'][0]['repetition'], 
        [s1c0, s1c1, s1c2], 
        reverse=p['rows'][0]['reverse'], 
        alignment=p['rows'][0]['alignment'], 
        axial_location=p['rows'][0]['axial_location'], 
        range=(p['rows'][0]['range_min'], p['rows'][0]['range_max']), 
        pitch_cascade=ds1, axial_cascade=dz1
    )

    r1c0 = paraturbo.TurbineLegacyParamsSection(**p['rows'][1]['sections'][0])
    r1c1 = paraturbo.TurbineLegacyParamsSection(**p['rows'][1]['sections'][1])
    r1c2 = paraturbo.TurbineLegacyParamsSection(**p['rows'][1]['sections'][2])

    dz2 = paraturbo.LinearCascade(tip_deviation=p['rows'][1]['axial_cascade']['tip_deviation'], 
                                 root_deviation=p['rows'][1]['axial_cascade']['root_deviation'])
    ds2 = paraturbo.LinearCascade(tip_deviation=p['rows'][1]['pitch_cascade']['tip_deviation'], 
                                 root_deviation=p['rows'][1]['pitch_cascade']['root_deviation'])

    rotor_1 = paraturbo.TurbineLegacyParamsRow(
        p['rows'][1]['repetition'], 
        [r1c0, r1c1, r1c2], 
        reverse=p['rows'][1]['reverse'], 
        alignment=p['rows'][1]['alignment'], 
        axial_location=p['rows'][1]['axial_location'], 
        range=(p['rows'][1]['range_min'], p['rows'][1]['range_max']), 
        pitch_cascade=ds2, axial_cascade=dz2
    )

    ref_fp_hub = np.loadtxt('paraturbo/tests/resources/Flowpath.curve', skiprows=11, max_rows=101)
    ref_fp_shroud = np.loadtxt('paraturbo/tests/resources/Flowpath.curve', skiprows=123, max_rows=101)

    fp_hub = paraturbo.BSplineFlowpath(ref_fp_hub, n_control_points=6)
    fp_shroud = paraturbo.BSplineFlowpath(ref_fp_shroud, n_control_points=12)

    prj = paraturbo.Project([stator_1, rotor_1], ['stator_1', 'rotor_1'], fp_hub, fp_shroud)
    prj.export('stage.geomturbo', fmt='geomturbo')

    prj.save('project_export', fmt='yaml_ext')


def test_load():
    prj = paraturbo.load_yaml_ext_project('project_export')
    prj.export('stage.geomturbo', fmt='geomturbo')