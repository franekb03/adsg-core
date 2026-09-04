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
import functools
import numpy as np
from typing import *
from concurrent.futures import wait, ProcessPoolExecutor, ThreadPoolExecutor

from adsg_core import InputParameter
from adsg_core.optimization.evaluator import DSGEvaluator, StochasticDSGEvaluator
from adsg_core.optimization.problem import DSGDesignSpace
from adsg_core.optimization.dv_output_defs import DesVar
from adsg_core.optimization.graph_processor import GraphProcessor
from adsg_core.optimization.assign_enc.time_limiter import run_timeout

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
    [SBArchOpt](https://sbarchopt.readthedocs.io/) wrapper for a DSG optimization problem. Note that under the
    hood, SBArchOpt uses [pymoo](https://pymoo.org/).
    The connection is made between the `ArchOptProblemBase` class (which specifies all information needed to optimize an
    architecture optimization problem), and the `DSGEvaluator` class, which contains all information for
    running a DSG architecture optimization problem.

    Parallel processing is possible by setting `n_parallel` to a number higher than 1.
    By default, assumes parallel processing is done within the thread and therefore starts a multiprocessing pool to
    run the parallel evaluations.

    Ensure SBArchOpt is installed: `pip install sb-arch-opt`

    Example usage:

    ```python
    from pymoo.optimize import minimize
    from sb_arch_opt.algo.pymoo_interface import get_nsga2

    evaluator = ...  # Instance of DSGEvaluator

    algorithm = get_nsga2(pop_size=100)
    problem = DSGArchOptProblem(evaluator)

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

        n_objs = len(evaluator.objectives)
        n_constr = len(evaluator.constraints)

        design_space = DSGDesignSpace(evaluator)

        parameter_nodes = self.evaluator.uncertain_parameter_nodes
        parameter_space = self.get_parameter_space(parameter_nodes)


        super().__init__(design_space, param_space=parameter_space, uq_method=uq_method, n_objs=n_objs, n_ieq_constr=n_constr,
                         obj_measure=obj_measure, ieq_constr_measure=constr_measure, nan_policy=nan_policy)

        self.obj_is_max = [obj.dir.value > 0 for obj in evaluator.objectives]
        self.con_ref = [(con.dir > 0, con.ref) for con in evaluator.constraints]

    @staticmethod
    def get_parameter_space(input_parameters_nodes: List[InputParameter]) -> StochasticParameterSpace:
        space = StochasticParameterSpace()
        for param_node in input_parameters_nodes:
            # TODO handle deterministic parameters in sbarchopt
            param = StochasticParameter(param_node.name, param_node.distribution)
            space.add_parameter(param)
        return space

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):

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

        samples = self.uq_method.get_samples(self.param_space)

        # Evaluate architectures
        if self.n_parallel is not None and self.n_parallel > 1:
            executor_class = ProcessPoolExecutor if self.parallel_processes else ThreadPoolExecutor
            with executor_class(max_workers=self.n_parallel) as executor:
                futures = [executor.submit(self.evaluator.evaluate, dsg, samples) for dsg in dsg_instances]

                wait(futures)
                results = [fut.result() for fut in futures]

        else:
            results = [self.evaluator.evaluate(dsg) for dsg in dsg_instances]

        # Process results
        for i, (obj_values, con_values) in enumerate(results):
            # Correct directions of objectives to represent minimization
            f_out[i, :] = [-val if self.obj_is_max[j] else val for j, val in enumerate(obj_values)]

            # Correct directions and offset constraints to represent g(x) <= 0
            g_out[i, :] = [(val-self.con_ref[j][1])*(-1 if self.con_ref[j][0] else 1)
                           for j, val in enumerate(con_values)]

    def _print_extra_stats(self):
        self.get_discrete_rates(show=True)
        self.evaluator.print_stats()

    def get_n_batch_evaluate(self) -> Optional[int]:
        return self.n_parallel

    def __repr__(self):
        return f'{self.__class__.__name__}({self.evaluator!r})'


ADSGDesignSpace = DSGDesignSpace  # Backward compatibility
ADSGStochasticArchOptProblem = DSGStochasticArchOptProblem  # Backward compatibility
