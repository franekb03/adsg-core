"""
MIT License

Copyright: (c) 2024, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import logging
import math
from typing import *
import warnings

import numpy as np

from adsg_core import DSGType
from adsg_core.graph.adsg_nodes import MetricNode, InputParameterNode
from adsg_core.optimization.dv_output_defs import *
from adsg_core.optimization.graph_processor import *

__all__ = ['StochasticDSGEvaluator', 'StochasticADSGEvaluator', 'HAS_SB_ARCH_OPT', 'check_dependency']
try:
    from sb_arch_opt.stochastic_problem import StochasticArchOptProblem
    from sb_arch_opt.uncertainty import RobustMeasure, StochasticResults, UQMethod
    from sb_arch_opt.sampling import TrailRepairWarning

    warnings.simplefilter("ignore", category=TrailRepairWarning)

    HAS_SB_ARCH_OPT = True

except ImportError:
    HAS_SB_ARCH_OPT = False

    class RobustMeasure:
        pass

    class StochasticResults:
        pass

    class UQMethod:
        pass

log = logging.getLogger('adsg.opt')


def check_dependency():
    if not HAS_SB_ARCH_OPT:
        raise ImportError('Looks like SBArchOpt is not installed! Run: pip install sb-arch-opt')

class StochasticDSGEvaluator(GraphProcessor):
    """
    Base class for implementing an evaluator for stochastic problem that directly evaluates DSG instances.
    Override _evaluate to implement the evaluation.

    Extends `GraphProcessor`, so all its functions are also available.
    """

    def get_problem(self,
                    uq_method: UQMethod,
                    obj_measure: List[RobustMeasure] = None,
                    constr_measure: List[RobustMeasure] = None,
                    nan_policy = 'propagate',
                    n_parallel=None, parallel_processes=True):

        from adsg_core.optimization.stochastic_problem import DSGStochasticArchOptProblem
        return DSGStochasticArchOptProblem(self,
                uq_method,
                obj_measure,
                constr_measure,
                nan_policy,
                n_parallel=n_parallel, parallel_processes=parallel_processes)

    def _choose_metric_type(self, objective: Objective, constraint: Constraint) -> Union[Objective, Constraint]:
        raise RuntimeError(f'Metric {objective.name} can either be an objective or a constraint! '
                           f'Specify the metric type using node.type = MetricType.x')

    def evaluate(self, dsg: DSGType, samples: np.ndarray, uq_method: UQMethod) -> StochasticResults:
        """
        Evaluate a DSG instance for all samples. Returns StochasticResult for given DSG.
        """

        # Evaluate the DSG instance
        metric_nodes = dsg.metric_nodes
        n_s = samples.shape[0]
        n_obj = len(self.objectives)
        n_constr = len(self.constraints)

        f_s = np.zeros((n_s, n_obj)) * np.nan
        g_s = np.zeros((n_s, n_constr)) * np.nan

        # Evaluate DSG instance for all samples
        for i in range(n_s):
            parameter_values = uq_method.param_space.include_deterministic_values(samples)
            realization = {node: float(value) for node, value in zip(list(dsg.input_parameter_values.keys()), parameter_values[i, :]) if node is not None}
            value_map = self._evaluate(dsg, metric_nodes, realization)
            f_s[i, :] = [value_map.get(objective.node, math.nan) for objective in self.objectives]
            g_s[i, :] = [value_map.get(constraint.node, math.nan) if constraint.node in metric_nodes else constraint.ref for constraint in self.constraints]

        # Apply UQ method to process the results and return StochasticResult for this DSG instance
        result = uq_method.process_results(np.concatenate([f_s, g_s], axis=1))

        # Set metric nodes with the corresponding StochasticOutput
        for i, objective in enumerate(self.objectives):
            if objective.node in metric_nodes:
                dsg.set_metric_value(objective.node, result.outputs[i])

        for i, constraint in enumerate(self.constraints):
            if constraint.node in metric_nodes:
                dsg.set_metric_value(constraint.node, result.outputs[n_obj+i])
        #TODO Consider rewriting StochasticResult class to return obj, constr tuple of [StochasticOutput], [StochasticOutput].
        # So far this option is chosen because StochasticResult.metric_results is retained this way
        return result

    def _evaluate(self, dsg: DSGType, metric_nodes: List[MetricNode], parameters: Dict[InputParameterNode, float]) -> Dict[MetricNode, float]:
        """
        Implement this function to provide DSG evaluation for ONE realization of the uncertain parameters.

        `parameters` maps every parameter node present in this architecture to its value for this realization.
        Should return a mapping from metric node to float (NaN is allowed).
        """
        raise NotImplementedError

StochasticADSGEvaluator = StochasticDSGEvaluator