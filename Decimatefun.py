import numpy as np


def Decimatefun(x, scale):
    # dc = x[:, range(0, x.shape[1], scale)]
    # drdc = dc[range(0, x.shape[0], scale), :]
    # y = drdc
    if scale%2 == 0:
        dc = x[:, range(0 + int(scale/2), x.shape[1], scale)]
        drdc = dc[range(0 + int(scale/2), x.shape[0], scale), :]
    else:
        dc = x[:, range(0 + int((scale-1)/2), x.shape[1], scale)]
        drdc = dc[range(0 + int((scale-1)/2), x.shape[0], scale), :]
    y = drdc
    return  y