import numpy as np


def line_zsr(x: np.ndarray) -> str:
    r'''
    按tecplot格式输出Z-S-R曲线；
    输入格式 N * L * 3
    '''
    assert x.shape[2] == 3 and len(x.shape) == 3
    s = ''
    s += 'variables=\"x\",\"s\",\"r\"\n'
    for n in range(x.shape[0]):
        s += 'zone i= %d j= 1 k= 1 f=point\n' % (x.shape[1])
        for l in range(x.shape[1]):
            # z-s-r
            s += '%16.8f\t%16.8f\t%16.8f\n' % (x[n, l, 0], x[n, l, 1], x[n, l, 2])
    return s