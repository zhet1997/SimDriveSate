import numpy as np
import matplotlib.pyplot as plt

from paraturbo.Flowpath import RawFlowpath, BSplineFlowpath

def test_raw_flowpath():
    df = np.loadtxt('paraturbo/tests/resources/ref_fp.dat')
    fp_z = df[0::2]
    fp_r = df[1::2]
    fp_ref = np.vstack((fp_z, fp_r)).T

    rfp = RawFlowpath(fp_ref)
    x = rfp()

    plt.subplots()
    plt.plot(x[:, 0], x[:, 1])
    plt.axis('equal')
    plt.show()



def test_bspline_flowpath():
    df = np.loadtxt('paraturbo/tests/resources/Flowpath.curve', skiprows=123, max_rows=101)
    fp_ref = df

    # Step 1
    bfp = BSplineFlowpath(fp_ref, n_control_points=12)
    x = bfp()

    plt.subplots()
    plt.plot(fp_ref[:, 0], fp_ref[:, 1], ls=':', color='k')
    plt.plot(x[:, 0], x[:, 1], lw=1, color='b')
    plt.plot(bfp.fp_bspl.c[:, 0], bfp.fp_bspl.c[:, 1], color='r', marker='s', lw=1)
    plt.axis('equal')
    plt.show()
