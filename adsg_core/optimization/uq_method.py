import numpy as np
from adsg_core.graph.adsg import *


class UQMethod:
    def __init__(self, dsg: DSGType, method, func, **kwargs):
        self.dsg = dsg
        self.method = method
        self.func = func
        if self.method == "MC":
            self.mean, self.std = self.mc(func, **kwargs)

    @staticmethod
    def run(dsg, method, func, **kwargs):
        if method == "MC":
            mean, std = mc(dsg, func, **kwargs)
            return mean, std

    @staticmethod
    def mc(dsg, func, n: int):
        res = np.full(n, np.nan)
        for i in range(n):
            sample = dsg.sample_parameters()
            res[i] = func(dsg, sample)

        mean = np.mean(res)
        std = np.std(res)

        return mean, std