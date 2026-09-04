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
import math
from typing import *

import numpy as np

from adsg_core.graph.adsg import DSGType
from adsg_core.graph.adsg_nodes import MetricNode, StochasticMetricType
from adsg_core.optimization.dv_output_defs import *
from adsg_core.optimization.graph_processor import *
from adsg_core.optimization.uq_method import *
from sb_arch_opt.uncertainty import *

__all__ = ['DSGEvaluator', 'ADSGEvaluator', 'StochasticDSGEvaluator', 'StochasticADSGEvaluator']


class DSGEvaluator(GraphProcessor):
    """
    Base class for implementing an evaluator that directly evaluates DSG instances.
    Override _evaluate to implement the evaluation.

    Extends `GraphProcessor`, so all its functions are also available.
    """

    def get_problem(self, n_parallel=None, parallel_processes=True):
        """Get an SBArchOpt problem instance."""
        from adsg_core.optimization.problem import DSGArchOptProblem
        return DSGArchOptProblem(self, n_parallel=n_parallel, parallel_processes=parallel_processes)

    def _choose_metric_type(self, objective: Objective, constraint: Constraint) -> Union[Objective, Constraint]:
        raise RuntimeError(f'Metric {objective.name} can either be an objective or a constraint! '
                           f'Specify the metric type using node.type = MetricType.x')

    def evaluate(self, dsg: DSGType) -> Tuple[List[float], List[float]]:
        """
        Evaluate a DSG instance. Returns a list of objective values and a list of constraint values.
        """

        # Evaluate the DSG instance
        metric_nodes = dsg.metric_nodes
        value_map = self._evaluate(dsg, metric_nodes)

        # Set metric values
        for metric_node in metric_nodes:
            dsg.set_metric_value(metric_node, value_map.get(metric_node, math.nan))

        # Associate values to objectives
        objective_values = [value_map.get(objective.node, math.nan) for objective in self.objectives]

        # Associate values to constraints: if the constraint does not exist in this DSG instance, set value to ref
        constraint_values = [value_map.get(constraint.node, math.nan)
                             if constraint.node in metric_nodes else constraint.ref
                             for constraint in self.constraints]

        return objective_values, constraint_values

    def _evaluate(self, dsg: DSGType, metric_nodes: List[MetricNode]) -> Dict[MetricNode, float]:
        """
        Implement this function to provide DSG evaluation.
        Should return a mapping from metric node to float (NaN is allowed).
        """
        raise NotImplementedError


class StochasticDSGEvaluator(GraphProcessor):
    """
    Base class for implementing an evaluator that directly evaluates DSG instances.
    Override _evaluate to implement the evaluation.

    Extends `GraphProcessor`, so all its functions are also available.
    """

    def get_problem(self,
                    uq_method: UQMethod,
                    obj_measure: List[RobustMeasure] = None,
                    constr_measure: List[RobustMeasure] = None,
                    nan_policy = None,
                    n_parallel=None, parallel_processes=True):

        from adsg_core.optimization.robust import DSGStochasticArchOptProblem
        return DSGStochasticArchOptProblem(self,
                uq_method,
                obj_measure,
                constr_measure,
                nan_policy,
                n_parallel=n_parallel, parallel_processes=parallel_processes)

    def evaluate(self, dsg: DSGType, samples: np.ndarray, uq_method: UQMethod = None) -> Tuple[List[float], List[float]]:
        """
        Evaluate a DSG instance. Returns a list of objective values and a list of constraint values.
        """

        # Evaluate the DSG instance
        metric_nodes = dsg.metric_nodes
        parameters_space = dsg.stochastic_space()

        n_s = samples.shape[0]

        f_s = np.zeros((n_s, len(self.objectives))) * np.nan
        g_s = np.zeros((n_s, len(self.constraints))) * np.nan

        # Evaluate all design vectors for each realization of the uncertain parameters: the loop is over samples,
        # not over design points, so that the evaluation function stays vectorized over design points
        for sample_i in range(n_s):
            metric_values = list(self._evaluate(dsg, metric_nodes, sample_i).values())
            f_s[sample_i, :] = metric_values[:len(self.objectives)]
            g_s[sample_i, :] = metric_values[len(self.objectives):]

        # Reduce the sampled responses of each design point to the values the optimizer sees
        stochastic_results = uq_method.process_results(np.concatenate([f_s, g_s], axis=1), parameters_space)

        for i, metric_node in enumerate(metric_nodes):
            dsg.set_metric_value(metric_node, stochastic_results[i])

        value_map = dsg.metric_values
        # Associate values to objectives
        objective_values = [value_map.get(objective.node, math.nan) for objective in self.objectives]

        # Associate values to constraints: if the constraint does not exist in this DSG instance, set value to ref
        constraint_values = [value_map.get(constraint.node, math.nan)
                             if constraint.node in metric_nodes else constraint.ref
                             for constraint in self.constraints]

        return objective_values, constraint_values

    def _evaluate(self, dsg: DSGType, metric_nodes: List[MetricNode], parameters) -> Dict[MetricNode, float]:
        """
        Implement this function to provide DSG evaluation.
        Should return a mapping from metric node to float (NaN is allowed).
        """
        raise NotImplementedError



ADSGEvaluator = DSGEvaluator  # Backward compatibility
StochasticADSGEvaluator = StochasticDSGEvaluator
