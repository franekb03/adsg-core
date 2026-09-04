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
import warnings
import numpy as np
from typing import *
from concurrent.futures import wait, ProcessPoolExecutor, ThreadPoolExecutor

from adsg_core.optimization.evaluator import StochasticDSGEvaluator
from adsg_core.optimization.problem import DSGDesignSpace

try:
    from sb_arch_opt.problem import ArchOptProblemBase
    from sb_arch_opt.robust import StochasticArchOptProblem
    from sb_arch_opt.uncertainty import *
    from sb_arch_opt.design_space import ArchDesignSpace
    from pymoo.core.variable import Variable, Real, Integer, Choice

    from sb_arch_opt.sampling import TrailRepairWarning
    warnings.simplefilter("ignore", category=TrailRepairWarning)

    HAS_SB_ARCH_OPT = True

except ImportError:
    HAS_SB_ARCH_OPT = False

    class ArchDesignSpace:
        pass

    class ArchOptProblemBase:
        pass

__all__ = ['check_dependency', 'DSGStochasticArchOptProblem', 'HAS_SB_ARCH_OPT',
           'ADSGDesignSpace', 'ADSGStochasticArchOptProblem']

log = logging.getLogger('adsg.opt')


def check_dependency():
    if not HAS_SB_ARCH_OPT:
        raise ImportError('Looks like SBArchOpt is not installed! Run: pip install sb-arch-opt')


class DSGStochasticArchOptProblem(StochasticArchOptProblem):
    """
    [SBArchOpt](https://sbarchopt.readthedocs.io/) wrapper for a stochastic (robust) DSG optimization problem.
    Note that under the hood, SBArchOpt uses [pymoo](https://pymoo.org/).

    The connection is made between `StochasticArchOptProblem` (which defines what a robust architecture
    optimization problem looks like to the optimizer) and `StochasticDSGEvaluator` (which evaluates DSG instances
    under uncertainty).

    The split of work is: the evaluator propagates the uncertainty and stores the resulting `StochasticOutput` of
    every metric on its DSG instance in **physical** units; this problem then reduces those outputs with the
    robust measures and applies the optimizer's conventions (minimization, `g(x) <= 0`). Because the graph is
    written before any sign flipping, it always holds actual quantities of interest.

    All architectures in one evaluation batch see the same realizations of the uncertain parameters (common random
    numbers), which is what makes design points comparable to each other and to a surrogate fitted through them.

    Note on measures for a *maximized* objective: the measure is applied to the physical samples, so `Margin(k)`
    gives `mean + k*std`. The conservative value of a maximized quantity is `mean - k*std`, so such an objective
    wants `Margin(k=-k)`; likewise the conservative side of a maximized metric is the *lower* tail, so a
    `Quantile(q)` becomes `Quantile(1-q)`.

    Parallel processing is possible by setting `n_parallel` to a number higher than 1.
    By default, assumes parallel processing is done within the thread and therefore starts a multiprocessing pool
    to run the parallel evaluations.

    Ensure SBArchOpt is installed: `pip install sb-arch-opt`

    Example usage:

    ```python
    from pymoo.optimize import minimize

    evaluator = ...  # Instance of StochasticDSGEvaluator
    problem = evaluator.get_problem(uq_method=MonteCarlo(n_evaluations=100, seed=42),
                                    obj_measure=[Margin(k=-2.)])

    result = minimize(problem, algorithm, termination=('n_eval', 500))
    ```
    """

    def __init__(self, evaluator: StochasticDSGEvaluator,
                 uq_method: UQMethod,
                 obj_measure: List[RobustMeasure] = None,
                 constr_measure: List[RobustMeasure] = None,
                 nan_policy: str = 'propagate',
                 n_parallel=None, parallel_processes=True):
        check_dependency()

        self.evaluator = evaluator
        self.n_parallel = n_parallel
        self.parallel_processes = parallel_processes

        n_obj = len(evaluator.objectives)
        n_constr = len(evaluator.constraints)

        design_space = DSGDesignSpace(evaluator)
        parameter_space = self.get_parameter_space(evaluator)

        super().__init__(design_space, param_space=parameter_space, uq_method=uq_method, n_obj=n_obj,
                         n_ieq_constr=n_constr, obj_measure=obj_measure, ieq_constr_measure=constr_measure,
                         nan_policy=nan_policy)

        self.obj_is_max = [obj.dir.value > 0 for obj in evaluator.objectives]
        self.con_ref = [(con.dir.value > 0, con.ref) for con in evaluator.constraints]

    @staticmethod
    def get_parameter_space(evaluator: StochasticDSGEvaluator) -> StochasticParameterSpace:
        """
        The *union* parameter space of the problem: every parameter node of the template graph, sorted by name.

        A hierarchical problem has parameters that only exist in some architectures (a parameter hanging off a
        selection-choice option), so per-instance spaces differ in width. Sampling the union instead keeps one
        fixed-width realization matrix for the whole batch, which is what makes common random numbers across
        architectures - and a constant input dimension for a polynomial chaos expansion - possible at all. Each
        instance then reads only the parameters it actually has.
        """
        graph = evaluator.graph
        space = StochasticParameterSpace()
        for param_node in evaluator.uncertain_parameter_nodes:
            distribution = graph.get_parameter_distribution(graph.input_parameter_value(param_node))
            space.add_parameter(StochasticParameter(param_node.name, distribution))
        return space

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        # Correct integer design variables
        self.design_space.round_x_discrete(x)

        # Generate architectures: once per design point, not once per realization, since the architecture does not
        # depend on the uncertain parameters
        is_discrete_mask = self.is_discrete_mask
        dsg_instances = []
        for i, xi in enumerate(x):
            x_arch = [int(val) if is_discrete_mask[j] else float(val) for j, val in enumerate(xi)]
            dsg_instance, x_imputed, is_active_arch = self.evaluator.get_graph(x_arch)
            dsg_instances.append(dsg_instance)
            x[i, :] = x_imputed
            is_active_out[i, :] = is_active_arch

        # One draw for the whole batch: common random numbers across design points
        samples = self.uq_method.get_samples(self.param_space)

        # Evaluate architectures: the evaluator loops the realizations and stores the physical stochastic outputs
        # on each instance
        if self.n_parallel is not None and self.n_parallel > 1:
            executor_class = ProcessPoolExecutor if self.parallel_processes else ThreadPoolExecutor
            with executor_class(max_workers=self.n_parallel) as executor:
                futures = [executor.submit(self.evaluator.evaluate, dsg, samples, self.uq_method, self.param_space)
                           for dsg in dsg_instances]

                wait(futures)
                results = [fut.result() for fut in futures]

        else:
            results = [self.evaluator.evaluate(dsg, samples, self.uq_method, self.param_space)
                       for dsg in dsg_instances]

        # Reduce the sampled responses with this problem's robust measures, then apply the optimizer's conventions.
        # Keeping the results also makes them available per individual through out['stochastic'].
        self.stochastic_results = results
        n_obj = self.n_obj
        for i, result in enumerate(results):
            for j in range(n_obj):
                value = result.outputs[j].reduce(self.obj_measure[j], nan_policy=self.nan_policy)

                # Correct directions of objectives to represent minimization
                f_out[i, j] = -value if self.obj_is_max[j] else value

            for j in range(self.n_ieq_constr):
                value = result.outputs[n_obj+j].reduce(self.ieq_constr_measure[j], nan_policy=self.nan_policy)

                # Correct directions and offset constraints to represent g(x) <= 0
                flip, ref = self.con_ref[j]
                g_out[i, j] = (value-ref)*(-1 if flip else 1)

    def _print_extra_stats(self):
        self.get_discrete_rates(show=True)
        self.evaluator.print_stats()

    def get_n_batch_evaluate(self) -> Optional[int]:
        return self.n_parallel

    def __repr__(self):
        return f'{self.__class__.__name__}({self.evaluator!r})'


ADSGDesignSpace = DSGDesignSpace  # Backward compatibility
ADSGStochasticArchOptProblem = DSGStochasticArchOptProblem  # Backward compatibility
