import numpy as np
from adsg_core.graph.adsg import *


class UQMethod:
    def __init__(self, dsg: DSGType, method, func, **kwargs):
        self.dsg = dsg
        self.method = method
        self.func = func
        if self.method == "MC":
            self.mean, self.std = self.mc(**kwargs)


    def mc(self, n: int, func):
        res = np.full(n, np.nan)
        for i in range(n):
            sample = self.dsg.sample_parameters()
            res[i] = func(self.dsg, sample)

        mean = np.mean(res)
        std = np.std(res)

        return mean, std