import numpy as np


def header() -> str:
    return 'GEOMETRY TURBO\nVERSION\t5.3\nTOLERANCE\t0.00005000\n'


def flowpath(hub: np.ndarray, shroud: np.ndarray) -> str:
    s = ''
    s += 'NI_BEGIN CHANNEL\n\tNI_BEGIN basic_curve\n\t\tNAME hub\n\t\tDISCRETISATION 10\n\t\tNI_BEGIN zrcurve\n\t\t\tZR polyline\n'
    s += '\t\t\t%d\n' % hub.shape[0]
    for i in range(hub.shape[0]):
        s += '%16.8f\t%16.8f\n' % (hub[i, 0], hub[i, 1])
    s += '\t\tNI_END zrcurve\n\tNI_END basic_curve\n'
    s += '\tNI_BEGIN basic_curve\n\t\tNAME shroud\n\t\tDISCRETISATION 10\n\t\tNI_BEGIN zrcurve\n\t\t\tZR polyline\n'
    s += '\t\t\t%d\n' % shroud.shape[0]
    for i in range(shroud.shape[0]):
        s += '%16.8f\t%16.8f\n' % (shroud[i, 0], shroud[i, 1])
    s += '\t\tNI_END zrcurve\n\tNI_END basic_curve\n'
    s += '\tNI_BEGIN channel_curve hub\n\t\tNAME hub\n\t\tVERTEX CURVE_P hub 0\n\t\tVERTEX CURVE_P hub 1\n\tNI_END channel_curve hub\n'
    s += '\tNI_BEGIN channel_curve shroud\n\t\tNAME shroud\n\t\tVERTEX CURVE_P shroud 0\n\t\tVERTEX CURVE_P shroud 1\n\tNI_END channel_curve shroud\n'
    s += 'NI_END CHANNEL\n'
    return s


def row_header(name: str, periodicity: int) -> str:
    s = ''
    s += 'NI_BEGIN nirow\n\tNAME %s\n\tTYPE normal\n' % (name)
    s += '\tNI_BEGIN NINonAxiSurfaces hub\n\t\tNAME\tnon axisymmetric hub\n\t\tREPETITION 0\n\tNI_END NINonAxiSurfaces hub\n'
    s += '\tNI_BEGIN NINonAxiSurfaces shroud\n\t\tNAME\tnon axisymmetric shroud\n\t\tREPETITION 0\n\tNI_END NINonAxiSurfaces shroud\n'
    s += '\tPERIODICITY %d\n' % (periodicity)
    s += '\tNI_BEGIN NIBlade\n\t\tNAME\tMain Blade\n\t\tNI_BEGIN nibladegeometry\n\t\t\tTYPE GEOMTURBO\n\t\t\tGEOMETRY_MODIFIED 0\n\t\t\tGEOMETRY TURBO VERSION 5\n'
    s += '\t\t\tblade_expansion_factor_hub 0.1\n\t\t\tblade_expansion_factor_shroud 0.1\n'
    s += '\t\t\tintersection_npts 10\n\t\t\tintersection_control 0\n\t\t\tunits 0.001\n\t\t\tnumber_of_blades %d\n' % (periodicity)
    return s


def row_data(X: np.ndarray, type: str) -> str:
    s = ''
    assert type=='suction' or type=='pressure'
    s += '\t\t\t%s\n' % (type)
    s += '\t\t\tSECTIONAL\n'
    s += '\t\t\t%d\n' % (X.shape[0])
    for i in range(X.shape[0]):
        s += '#section %d\nXYZ\n%d\n' % (i + 1, X.shape[1])
        for j in range(X.shape[1]):
            # z-s-r -> x-y-z
            s += '%16.8f\t%16.8f\t%16.8f\n' % (X[i, j, 1], X[i, j, 2], X[i, j, 0])
    return s


def row_tail() -> str:
    return '\t\tNI_END nibladegeometry\n\tNI_END NIBlade\nNI_END nirow\n'