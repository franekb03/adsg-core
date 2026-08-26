from adsg_core.graph.parameter_node import ParameterNode
from adsg_core.optimization.evaluator import DSGEvaluator
import math
from typing import *
from adsg_core.graph.adsg import DSGType
from adsg_core.graph.adsg_nodes import MetricNode, StochasticMetricType
from adsg_core.optimization.dv_output_defs import *
from adsg_core.optimization.graph_processor import *
import chaospy as cp
from adsg_core.optimization.uq_method import UQMethod


class DSGStochasticEvaluator(DSGEvaluator):
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
        parameter_nodes = dsg.parameter_nodes
        value_map = self._evaluate(dsg, metric_nodes, parameter_nodes)

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

    def process_stochastic_qoi(self, metric_node: MetricNode, mean, std) -> float:
        """
        Process stochastic qoi metric and return a float stored on the MetricNode that can be used by pymoo.
        Input:
            - metric_node: MetricNode to evaluate.
            - mean, std: returned by UQ method
        Output:
            - float
        TODO Impelent QUANTILE handling and storing distribution
        """
        if metric_node.stochastic == StochasticMetricType.MEAN:
            return mean
        elif metric_node.stochastic == StochasticMetricType.MARGIN:
            return mean + metric_node.k * std
        elif metric_node.quantile == StochasticMetricType.QUANTILE:
            raise NotImplementedError


    def propagate_uncertainty(self, uq_method: str, func, dsg: DSGType, metric_nodes:List[MetricNode], parameter_nodes: List[ParameterNode], **kwargs) -> Dict[MetricNode, float]:
        """
        This function propagates uncertainty propagation using a UQ method for a single design vector.
        This function should be called inside _evaluate() per each objective
        Input:
            - uq_method: UQMethod instance
            - func: Objective/Constraint function with the following format func(dsg: DSGType, param_sample: List[float]) -> float
            **kwargs: N samples for Monte Carlo evaluation or any other relevant parameters for the chosen UQ method.
        Output:
            - Returns a mapping from metric node to float in the same format as _evaluate
        """
        dict = {}
        for metric_node in metric_nodes:
            uq_analysis = UQMethod(uq_method, metric_node=metric_node, **kwargs)
            dict[metric_node] = self.process_stochastic_qoi(metric_node=metric_node, mean=uq_analysis.mean, std=uq_analysis.std)
        return dict


    def _evaluate(self, dsg: DSGType, metric_nodes: List[MetricNode], parameter_nodes: List[ParameterNode]) -> Dict[MetricNode, float]:
        """
        Implement this function to provide DSG evaluation.
        Should return a mapping from metric node to float (NaN is allowed).
        """
        raise NotImplementedError


# ADSGEvaluator = DSGEvaluator  # Backward compatibility
