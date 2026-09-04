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
from adsg_core.graph.adsg_nodes import MetricNode, InputParameter
from adsg_core.optimization.dv_output_defs import *
from adsg_core.optimization.graph_processor import *
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
    Base class for implementing an evaluator that evaluates DSG instances **under uncertainty**.
    Override `_evaluate` to implement the evaluation of one realization of the uncertain parameters.

    Extends `GraphProcessor`, so all its functions are also available.

    The uncertain parameters themselves live on the graph: a parameter node (`InputParameter`) carries no value,
    the DSG instance holds its *distribution* (`DSG.set_input_parameter_value`). Realizations are never stored on
    the graph - they are handed to `_evaluate` and discarded. What is stored per evaluated architecture is the
    `StochasticOutput` of every metric present in it: the full sampled column, in physical units.

    `evaluate` owns the loop over realizations, so it can be used on its own:

    ```python
    uq_method = MonteCarlo(n_evaluations=100, seed=42)
    dsg, _, _ = evaluator.get_graph(x)
    result = evaluator.evaluate(dsg, uq_method.get_samples(dsg.stochastic_space), uq_method)

    result.outputs[0].mean()              # Statistics of the first objective
    dsg.metric_value(some_metric_node)    # ... also stored on the graph, per metric node
    ```
    """

    def get_problem(self,
                    uq_method: UQMethod,
                    obj_measure: List[RobustMeasure] = None,
                    constr_measure: List[RobustMeasure] = None,
                    nan_policy: str = 'propagate',
                    n_parallel=None, parallel_processes=True):
        """Get an SBArchOpt problem instance for robust optimization of this evaluator."""
        from adsg_core.optimization.robust import DSGStochasticArchOptProblem
        return DSGStochasticArchOptProblem(
            self, uq_method=uq_method, obj_measure=obj_measure, constr_measure=constr_measure,
            nan_policy=nan_policy, n_parallel=n_parallel, parallel_processes=parallel_processes)

    def _choose_metric_type(self, objective: Objective, constraint: Constraint) -> Union[Objective, Constraint]:
        raise RuntimeError(f'Metric {objective.name} can either be an objective or a constraint! '
                           f'Specify the metric type using node.type = MetricType.x')

    def evaluate(self, dsg: DSGType, samples: np.ndarray, uq_method: UQMethod,
                 param_space: StochasticParameterSpace = None) -> StochasticResult:
        """
        Propagate `samples` (an n_samples x n_parameters matrix of realizations of the uncertain parameters)
        through one DSG instance.

        Stores the `StochasticOutput` of every metric present in the instance on the instance itself, in physical
        units (no objective/constraint sign conventions are applied here - that is the problem's business), and
        returns the `StochasticResult` whose `outputs` are ordered objectives-then-constraints.

        `param_space` describes what the columns of `samples` are; it defaults to this instance's own parameter
        space. An optimization problem passes the *union* space of the template graph instead, so that one sample
        matrix serves every architecture (see `DSGStochasticArchOptProblem`).
        """
        if param_space is None:
            param_space = dsg.stochastic_space

        samples = np.atleast_2d(np.asarray(samples, dtype=float))
        if samples.shape[1] != param_space.n_parameters:
            raise ValueError(f'Realizations have {samples.shape[1]} columns, but the parameter space has '
                             f'{param_space.n_parameters} parameters')

        metric_nodes = dsg.metric_nodes
        n_obj, n_con = len(self.objectives), len(self.constraints)

        # The columns of samples follow the parameter space's order; DSG.input_parameter_nodes and
        # GraphProcessor.uncertain_parameter_nodes are both sorted by name so that the two line up. Parameters that
        # do not exist in this architecture are simply not looked up.
        param_nodes = self._get_sample_nodes(dsg, param_space)

        f_s = np.full((samples.shape[0], n_obj), np.nan)
        g_s = np.full((samples.shape[0], n_con), np.nan)

        for sample_i, sample in enumerate(samples):
            realization = {node: float(value) for node, value in zip(param_nodes, sample) if node is not None}
            value_map = self._evaluate(dsg, metric_nodes, realization)

            # Look up by node, not by position: the order of metric_nodes is not the order of objectives and
            # constraints, and a constraint that does not exist in this instance is substituted with its reference
            f_s[sample_i, :] = [value_map.get(objective.node, math.nan) for objective in self.objectives]
            g_s[sample_i, :] = [value_map.get(constraint.node, math.nan)
                                if constraint.node in metric_nodes else constraint.ref
                                for constraint in self.constraints]

        result = uq_method.process_results(np.concatenate([f_s, g_s], axis=1), param_space)

        # Store the sampled outputs on the instance, keyed by node
        for i, objective in enumerate(self.objectives):
            if objective.node in metric_nodes:
                dsg.set_metric_value(objective.node, result.outputs[i])
        for i, constraint in enumerate(self.constraints):
            if constraint.node in metric_nodes:
                dsg.set_metric_value(constraint.node, result.outputs[n_obj+i])

        return result

    @staticmethod
    def _get_sample_nodes(dsg: DSGType, param_space: StochasticParameterSpace) -> List[Optional[InputParameter]]:
        """Map each column of a realization to the parameter node it belongs to, or None if that parameter does
        not exist in this architecture"""
        nodes_by_name = {node.name: node for node in dsg.input_parameter_nodes}
        return [nodes_by_name.get(name) for name in param_space.parameter_names]

    def _evaluate(self, dsg: DSGType, metric_nodes: List[MetricNode],
                  parameters: Dict[InputParameter, float]) -> Dict[MetricNode, float]:
        """
        Implement this function to provide DSG evaluation for ONE realization of the uncertain parameters.

        `parameters` maps every parameter node present in this architecture to its value for this realization.
        Should return a mapping from metric node to float (NaN is allowed).
        """
        raise NotImplementedError


ADSGEvaluator = DSGEvaluator  # Backward compatibility
StochasticADSGEvaluator = StochasticDSGEvaluator
