import yaml
import os
import numpy as np
import paraturbo


def set_dict_cascade(d: dict) -> paraturbo.Satellite_2d.GenericCascade:
    if d['cascade_method'] == 'linear':
        return paraturbo.LinearCascade(tip_deviation=d['tip_deviation'], root_deviation=d['root_deviation'])
    if d['cascade_method'] == 'quadratic':
        return paraturbo.QuadraticCascade(tip_deviation=d['tip_deviation'],
                                          mid_deviation=d['mid_deviation'],
                                          mid_location=d['mid_location'])
    else:
        raise ValueError


def set_dict_flowpath(d: dict, fp_ref: np.ndarray) -> paraturbo.Flowpath.GenericFlowpath:
    if d['flowpath_method'] == 'raw_flowpath':
        return paraturbo.RawFlowpath(fp_ref)
    if d['flowpath_method'] == 'bspline_flowpath':
        return paraturbo.BSplineFlowpath(
            fp_ref,
            n_interp=d['n_interp'],
            n_control_points=d['n_control_points'],
            param_method=d['param_method'],
            deviation=d['deviation']                          
        )
    else:
        raise ValueError


def load_yaml_ext_project(path: str) -> paraturbo.Project:
    assert os.path.exists(os.path.join(path, 'project.yaml'))
    assert os.path.exists(os.path.join(path, 'fp_hub.dat'))
    assert os.path.exists(os.path.join(path, 'fp_shroud.dat'))

    with open(path + '/project.yaml', 'r') as f:
        p = yaml.safe_load(f)

    assert len(p['rows']) == len(p['row_name'])

    rows = [None] * len(p['rows'])
    for i in range(len(rows)):
        dz = set_dict_cascade(p['rows'][i]['axial_cascade'])
        ds = set_dict_cascade(p['rows'][i]['pitch_cascade'])
        sections = [None] * len(p['rows'][i]['sections'])

        for j in range(len(sections)):
            if p['rows'][i]['sections'][j]['section_method'] != 'axial_turbine':
                raise NotImplementedError
            sections[j] = paraturbo.TurbineLegacyParamsSection(**p['rows'][i]['sections'][j])

        rows[i] = paraturbo.TurbineLegacyParamsRow(
            p['rows'][i]['repetition'], 
            sections, 
            reverse=p['rows'][i]['reverse'], 
            alignment=p['rows'][i]['alignment'], 
            axial_location=p['rows'][i]['axial_location'], 
            range=(p['rows'][i]['range_min'], p['rows'][i]['range_max']), 
            pitch_cascade=ds, axial_cascade=dz
        )

    ref_fp_hub = np.loadtxt(path + '/fp_hub.dat')
    ref_fp_shroud = np.loadtxt(path + '/fp_shroud.dat')
    fp_hub = set_dict_flowpath(p['hub_flowpath'], ref_fp_hub)
    fp_shroud = set_dict_flowpath(p['shroud_flowpath'], ref_fp_shroud)

    return paraturbo.Project(rows, p['row_name'], fp_hub, fp_shroud)
