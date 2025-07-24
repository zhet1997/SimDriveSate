import numpy as np

def centroid(points):
    x = points[:, 0]
    y = points[:, 1]
    x1 = np.roll(x, -1)
    y1 = np.roll(y, -1)
    CoGxi = (x + x1) / 3
    CoGyi = (y + y1) / 3
    Si = 0.5 * (x * y1 - x1 * y)
    CoGx = np.sum(Si * CoGxi) / np.sum(Si)
    CoGy = np.sum(Si * CoGyi) / np.sum(Si)
    return CoGx, CoGy