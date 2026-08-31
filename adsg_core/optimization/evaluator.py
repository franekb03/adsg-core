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
from adsg_core.graph.adsg import DSGType
from adsg_core.graph.adsg_nodes import MetricNode, StochasticMetricType, UncertainParameterNode
from adsg_core.optimization.dv_output_defs import *
from adsg_core.optimization.graph_processor import *
from adsg_core.optimization.uq_method import *

__all__ = ['DSGEvaluator', 'ADSGEvaluator']


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

    def process_stochastic_qoi(self, metric_node: MetricNode, mean: float, std: float) -> float:
        """
        Reduce statistics of stochastic metric to the single value the optimizer sees.
        """
        if metric_node.stochastic in (None, StochasticMetricType.NONE, StochasticMetricType.MEAN):
            return mean

        if metric_node.stochastic == StochasticMetricType.MARGIN:
            if metric_node.dir is None:
                raise ValueError(f'A direction is needed for a margin metric: {metric_node.name}')
            if metric_node.k is None or metric_node.k < 0:
                raise ValueError(f'A positive k is needed for a margin metric: {metric_node.name}')
            return mean - np.sign(metric_node.dir) * metric_node.k * std

        if metric_node.stochastic == StochasticMetricType.QUANTILE:
            raise NotImplementedError

        raise ValueError(f'Unknown stochastic metric type: {metric_node.stochastic}')


    def propagate_uncertainty(self, uq_method: str, func, dsg: DSGType, metric_node: MetricNode, **kwargs) -> Dict[MetricNode, float]:
        """
        This function propagates uncertainty using a UQ method for a single design vector.
        This function should be called inside _evaluate() per each objective
        Input:
            - uq_method: UQMethod instance
            - func: Objective/Constraint function with the following format func(dsg: DSGType, param_sample: Dict[UncertainParameterNode, float]) -> float
            - dsg: DSG instance
            - metric_node: MetricNode instance
            **kwargs: N samples for Monte Carlo evaluation or any other relevant parameters for the chosen UQ method.
        Output:
            - Returns a mapping from metric node to float
        """
        stochastic_metric_node = {}
        mean, std = UQMethod.run(dsg, uq_method, func, **kwargs)
        dsg.set_metric_statistics(metric_node, mean=mean, std=std, method=uq_method, n_samples=kwargs.get('n'))
        stochastic_metric_node[metric_node] = self.process_stochastic_qoi(metric_node=metric_node, mean=mean, std=std)
        return stochastic_metric_node

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


ADSGEvaluator = DSGEvaluator  # Backward compatibility
