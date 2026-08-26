from adsg_core.graph.adsg_nodes import DSGNode
from enum import Enum
import chaospy as cp
import numpy as np

class Distribution(Enum):
    NORMAL = "normal"

class ParameterNode(DSGNode):

    def __init__(self, name, distribution: cp.Distribution, idx=None):

        self.name = name
        self.idx = idx
        self.distribution = distribution
        super(ParameterNode, self).__init__()

    def sample(self, n:int) -> np.ndarray:
        return self.distribution.sample(n)


if __name__ == "__main__":
    n = cp.Normal(mu=0, sigma=1)
    p = ParameterNode("Normal", n)
    print(p)