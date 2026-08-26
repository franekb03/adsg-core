import chaospy as cp
import numpy as np
from adsg_core.graph.parameter_node import ParameterNode
from adsg_core.graph.adsg import *


class UQMethod:
    def __init__(self, dsg: DSGType, parameter_nodes: List[ParameterNode], kind, **kwargs):
        self.dsg = dsg
        self.parameter_nodes = parameter_nodes
        self.kind = kind
        if type == "MC":
            self.mean, self.std = self.mc(**kwargs)


    def mc(self, N: int, func):
        joint_dist = cp.J(*[node.distribution for node in self.parameter_nodes])
        res = np.full(N, np.nan)
        samples = joint_dist.sample(N, rule="sobol")
        for i in range(N):
            parameters_sample = joint_dist.sample()
            res[i] = func(self.dsg, samples[:, i])

        mean = np.mean(res)
        std = np.std(res)

        return mean, std