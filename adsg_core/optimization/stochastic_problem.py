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

from adsg_core.optimization.stochastic_evaluator import  StochasticDSGEvaluator
from adsg_core.optimization.problem import DSGDesignSpace


try:
    from sb_arch_opt.stochastic_problem import StochasticArchOptProblem
    from sb_arch_opt.uncertainty import *
    from pymoo.core.variable import Variable, Real, Integer, Choice

    from sb_arch_opt.sampling import TrailRepairWarning
    warnings.simplefilter("ignore", category=TrailRepairWarning)

    HAS_SB_ARCH_OPT = True

except ImportError:
    HAS_SB_ARCH_OPT = False

    class StochasticArchOptProblem:
        pass

__all__ = ['check_dependency', 'DSGStochasticArchOptProblem', 'HAS_SB_ARCH_OPT',
           'ADSGDesignSpace', 'ADSGStochasticArchOptProblem']

log = logging.getLogger('adsg.opt')


def check_dependency():
    if not HAS_SB_ARCH_OPT:
        raise ImportError('Looks like SBArchOpt is not installed! Run: pip install sb-arch-opt')


class DSGStochasticArchOptProblem(StochasticArchOptProblem):
    """
    [SBArchOpt](https://sbarchopt.readthedocs.io/) wrapper for a DSG stochastic optimization problem. Note that under the
    hood, SBArchOpt uses [pymoo](https://pymoo.org/).
    The connection is made between the `StochasticArchOptProblem` class (which specifies all information needed to optimize an
    architecture optimization problem), and the `StochasticDSGEvaluator` class, which contains all information for
    running a stochastic DSG architecture optimization problem.

    Parallel processing is possible by setting `n_parallel` to a number higher than 1.
    By default, assumes parallel processing is done within the thread and therefore starts a multiprocessing pool to
    run the parallel evaluations.

    Ensure SBArchOpt is installed: `pip install sb-arch-opt`

    Example usage:

    ```python
    from pymoo.optimize import minimize
    from sb_arch_opt.algo.pymoo_interface import get_nsga2

    evaluator = ...  # Instance of StochasticDSGEvaluator

    algorithm = get_nsga2(pop_size=100)
    problem = DSGStochasticArchOptProblem(evaluator, uq_method)

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


        super().__init__(design_space, uq_method=uq_method, n_obj=n_obj, n_ieq_constr=n_constr,
                         obj_measure=obj_measure, ieq_constr_measure=constr_measure, nan_policy=nan_policy)

        self.obj_is_max = [obj.dir.value > 0 for obj in evaluator.objectives]
        self.con_ref = [(con.dir > 0, con.ref) for con in evaluator.constraints]


    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        """
        Overrides parent _arch_evaluate class to integrate it with StochasticDSGEvaluator, but maintains the same functionality.
        """
        # Correct integer design variables
        self.design_space.round_x_discrete(x)

        # Generate architectures
        is_discrete_mask = self.is_discrete_mask
        dsg_instances = []
        for i, xi in enumerate(x):
            x_arch = [int(val) if is_discrete_mask[j] else float(val) for j, val in enumerate(xi)]
            dsg_instance, x_imputed, is_active_arch = self.evaluator.get_graph(x_arch)
            dsg_instances.append(dsg_instance)
            x[i, :] = x_imputed
            is_active_out[i, :] = is_active_arch

        # Sample parameter space

        samples = self.uq_method.get_samples()

        # Evaluate architectures
        if self.n_parallel is not None and self.n_parallel > 1:
            executor_class = ProcessPoolExecutor if self.parallel_processes else ThreadPoolExecutor
            with executor_class(max_workers=self.n_parallel) as executor:
                futures = [executor.submit(self.evaluator.evaluate, dsg, samples, self.uq_method) for dsg in dsg_instances]

                wait(futures)
                results = [fut.result() for fut in futures]

        else:
            results = [self.evaluator.evaluate(dsg, samples, self.uq_method) for dsg in dsg_instances]

        # Process results
        self.stochastic_results = results
        n_obj = self.n_obj
        n_constr = self.ieq_constr_measure
        for i, result in enumerate(results):
            for j in range(n_obj):
                value = result.outputs[j].reduce(self.obj_measure[j], nan_policy=self.nan_policy)

                # Correct directions of objectives to represent minimization
                f_out[i, j] = -value if self.obj_is_max[j] else value

            for j in range(self.n_ieq_constr):
                value = result.outputs[n_obj+j].reduce(n_constr[j], nan_policy=self.nan_policy)

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
