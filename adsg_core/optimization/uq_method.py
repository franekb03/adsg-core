import numpy as np
from adsg_core.graph.adsg import *


class UQMethod:

    @staticmethod
    def run(dsg, method, func, **kwargs):
        if method == "MC":
            mean, std = UQMethod.mc(dsg, func, **kwargs)
            return mean, std
        else:
            raise ValueError("Method {} not implemented".format(method))

    @staticmethod
    def mc(dsg, func, n: int):
        res = np.full(n, np.nan)
        for i in range(n):
            # Sample parameters on DSG instance and return a dictionary sample: Dict[UncertainParameterNode, float]
            sample = dsg.sample_parameters()
            res[i] = func(dsg, sample)

        mean = np.mean(res)
        std = np.std(res)

        return mean, std