"""
Tests for InputParameterNode and the uncertainty-propagation machinery built on top of it.

Ordered from simple to general:
1.  InputParameterNode itself (it is an identity; the value lives on the graph)
2.  Storing parameter distributions on the DSG
3.  Parameter nodes under graph derivation (conditional existence)
4.  GraphProcessor: parameters are not design variables, and the union parameter space
5.  StochasticDSGEvaluator.evaluate: one realization per sample, StochasticOutput per metric on the graph
6.  A full stochastic optimization problem through SBArchOpt
"""
import math
import pytest
import numpy as np
import openturns as ot
from typing import *

from adsg_core.graph.adsg import DSGType
from adsg_core.graph.adsg_basic import *
from adsg_core.graph.adsg_nodes import *
from adsg_core.optimization.evaluator import *
from adsg_core.optimization.stochastic_evaluator import *
from adsg_core.optimization.graph_processor import *
from sb_arch_opt.uncertainty import (MonteCarlo, PolynomialChaos, StochasticOutput, StochasticResults,
                                     StochasticParameterSpace, Mean, Margin, Quantile)


"""#################################
### 1. InputParameterNode        ###
#################################"""


def test_input_parameter_node_basics():
    par_node = InputParameterNode('E')

    assert par_node.name == 'E'
    assert par_node.idx is None
    assert str(par_node) == 'PARAM[E]'
    assert repr(par_node)
    assert par_node.get_export_color()
    assert par_node.str_context() == 'PARAM.E'


def test_input_parameter_node_export_title():
    """The node holds no value, so it only shows one once the graph assigned it for export"""
    par_node = InputParameterNode('E')
    assert par_node.get_export_title() == 'E'

    par_node.assigned_value = ot.Normal(10., 2.)
    assert par_node.get_export_title().startswith('E = ')


def test_input_parameter_nodes_are_distinct():
    """Nodes are identity-based, so two parameters with the same name are still different nodes"""
    par_a, par_b = InputParameterNode('E'), InputParameterNode('E')

    assert par_a != par_b
    assert len({par_a, par_b}) == 2
    assert par_a == par_a


"""#################################
### 2. Value storage on the DSG  ###
#################################"""


def _dsg_with_parameters(n, par_nodes):
    dsg = BasicDSG()
    dsg.add_edges([(n[0], par_node) for par_node in par_nodes])
    return dsg.set_start_nodes({n[0]})


def test_set_get_input_parameter_value(n):
    par_a, par_b = InputParameterNode('A'), InputParameterNode('B')
    dsg = _dsg_with_parameters(n, [par_a, par_b])

    assert dsg.feasible
    assert set(dsg.input_parameter_nodes) == {par_a, par_b}

    assert dsg.input_parameter_value(par_a) is None
    dist = ot.Normal(0., 1.)
    dsg.set_input_parameter_value(par_a, dist)
    assert dsg.input_parameter_value(par_a) is dist

    # A deterministic parameter is stored as a plain value
    dsg.set_input_parameter_value(par_b, 1.225)
    assert dsg.input_parameter_value(par_b) == 1.225

    dsg.reset_input_parameter_values()
    assert dsg.input_parameter_values == {}
    assert dsg.input_parameter_value(par_a) is None


def test_input_parameter_values_is_a_copy(n):
    par_a = InputParameterNode('A')
    dsg = _dsg_with_parameters(n, [par_a])

    dsg.set_input_parameter_value(par_a, 1.)
    values = dsg.input_parameter_values
    values[par_a] = 99.

    assert dsg.input_parameter_value(par_a) == 1.


def test_input_parameter_values_survive_copy(n):
    """Regression: the copy constructors once passed a pre-rename keyword, silently dropping every value"""
    par_a = InputParameterNode('A')
    dsg = _dsg_with_parameters(n, [par_a])

    dsg.set_input_parameter_value(par_a, ot.Normal(0., 1.))
    dsg_copy = dsg.copy()

    assert dsg_copy.input_parameter_value(par_a) is not None
    assert dsg_copy.input_parameter_value(par_a).getMean()[0] == 0.


def test_input_parameter_values_keep_insertion_order(n):
    """A realization is an array indexed by position, and the column order comes from this dict (see
    GraphProcessor.param_space), so its order is part of the contract"""
    pars = [InputParameterNode(name) for name in ['c', 'a', 'b']]
    dsg = _dsg_with_parameters(n, pars)

    for par in pars:
        dsg.set_input_parameter_value(par, ot.Normal(0., 1.))

    assert [par.name for par in dsg.input_parameter_values] == ['c', 'a', 'b']


"""#################################
### 3. Derivation / existence    ###
#################################"""


def test_parameter_node_conditional_existence(n):
    """A parameter node hung off a selection-choice option only exists if that option is selected"""
    par_common = InputParameterNode('common')
    par_opt_a, par_opt_b = InputParameterNode('only_a'), InputParameterNode('only_b')

    dsg = BasicDSG()
    dsg.add_edges([(n[0], par_common), (n[1], par_opt_a), (n[2], par_opt_b)])
    dsg.add_selection_choice('C1', n[0], [n[1], n[2]])
    dsg = dsg.set_start_nodes({n[0]})
    for par in [par_common, par_opt_a, par_opt_b]:
        dsg.set_input_parameter_value(par, ot.Normal(0., 1.))

    processor = GraphProcessor(dsg)
    assert len(processor.des_vars) == 1

    seen = set()
    for opt_idx in range(2):
        graph, _, _ = processor.get_graph([opt_idx])

        par_nodes = set(graph.input_parameter_nodes)
        assert par_common in par_nodes
        seen |= par_nodes

        # Only the parameter belonging to the selected option is present in the graph
        assert (par_opt_b not in par_nodes) if par_opt_a in par_nodes else (par_opt_b in par_nodes)
        assert len(par_nodes) == 2

        # The distributions survived derivation
        assert graph.input_parameter_value(par_common) is not None

    assert seen == {par_common, par_opt_a, par_opt_b}


def test_derived_instance_keeps_all_parameter_values(n):
    """Note the *values* are not pruned along with the nodes: an instance carries the whole template dict, which
    is what keeps its column order aligned with the problem's union parameter space"""
    par_common, par_opt_a = InputParameterNode('common'), InputParameterNode('only_a')

    dsg = BasicDSG()
    dsg.add_edges([(n[0], par_common), (n[1], par_opt_a)])
    dsg.add_selection_choice('C1', n[0], [n[1], n[2]])
    dsg = dsg.set_start_nodes({n[0]})
    for par in [par_common, par_opt_a]:
        dsg.set_input_parameter_value(par, ot.Normal(0., 1.))

    graph, _, _ = GraphProcessor(dsg).get_graph([1])  # option without only_a

    assert par_opt_a not in graph.input_parameter_nodes  # pruned from the graph
    assert par_opt_a in graph.input_parameter_values  # but its value is still carried


def test_parameter_values_isolated_between_instances(n):
    """Each derived instance carries its own values"""
    par_a = InputParameterNode('A')

    dsg = BasicDSG()
    dsg.add_edges([(n[0], par_a)])
    dsg.add_selection_choice('C1', n[0], [n[1], n[2]])
    dsg = dsg.set_start_nodes({n[0]})

    processor = GraphProcessor(dsg)
    graph_a, _, _ = processor.get_graph([0])
    graph_b, _, _ = processor.get_graph([1])

    graph_a.set_input_parameter_value(par_a, 1.)
    graph_b.set_input_parameter_value(par_a, 2.)

    assert graph_a.input_parameter_value(par_a) == 1.
    assert graph_b.input_parameter_value(par_a) == 2.

    # The template graph is never touched by evaluating instances
    assert dsg.input_parameter_value(par_a) is None


"""#################################
### 4. GraphProcessor            ###
#################################"""


def test_parameters_are_not_design_variables(n):
    """Input parameters must never show up as design variables: the optimizer does not choose them"""
    par_a = InputParameterNode('A')
    dv_node = DesignVariableNode('DV', bounds=(0., 1.))

    dsg = BasicDSG()
    dsg.add_edges([(n[0], par_a), (n[0], dv_node)])
    dsg.add_selection_choice('C1', n[0], [n[1], n[2]])
    dsg = dsg.set_start_nodes({n[0]})

    processor = GraphProcessor(dsg)

    assert [dv.name for dv in processor.des_vars] == ['C1', 'DV']
    assert all(dv.node is not par_a for dv in processor.des_vars)
    assert par_a not in processor.design_variable_nodes


def test_processor_input_parameter_nodes_sorted(n):
    """The node listing is sorted by name. Note this is NOT the parameter space's column order, which follows the
    order the values were assigned in - see test_param_space_follows_value_insertion_order"""
    pars = [InputParameterNode(name) for name in ['C', 'A', 'B']]

    dsg = BasicDSG()
    dsg.add_edges([(n[0], par) for par in pars])
    dsg.add_selection_choice('C1', n[0], [n[1], n[2]])
    dsg = dsg.set_start_nodes({n[0]})

    assert [par.name for par in GraphProcessor(dsg).input_parameter_nodes] == ['A', 'B', 'C']


def test_param_space_follows_value_insertion_order(n):
    """The union space is built by iterating the value dict, so its columns are in assignment order. This is the
    order a realization is handed out in, so it has to be the one the evaluator zips against"""
    pars = [InputParameterNode(name) for name in ['c', 'a', 'b']]

    dsg = BasicDSG()
    dsg.add_edges([(n[0], par) for par in pars])
    dsg = dsg.set_start_nodes({n[0]})
    for par in pars:
        dsg.set_input_parameter_value(par, ot.Normal(0., 1.))

    space = GraphProcessor(dsg).param_space
    assert space.parameter_names == ['c', 'a', 'b']
    assert space.n_parameters == 3


def test_param_space_is_the_union_of_all_parameters(n):
    """Every parameter of the template graph gets a column, including branch-local ones: one fixed-width
    realization matrix then serves every architecture"""
    par_common, par_opt_a, par_opt_b = (InputParameterNode('common'), InputParameterNode('only_a'),
                                        InputParameterNode('only_b'))

    dsg = BasicDSG()
    dsg.add_edges([(n[0], par_common), (n[1], par_opt_a), (n[2], par_opt_b)])
    dsg.add_selection_choice('C1', n[0], [n[1], n[2]])
    dsg = dsg.set_start_nodes({n[0]})
    for par in [par_common, par_opt_a, par_opt_b]:
        dsg.set_input_parameter_value(par, ot.Normal(0., 1.))

    assert GraphProcessor(dsg).param_space.n_parameters == 3


@pytest.mark.parametrize('value,expected_mean', [(1.225, 1.225), (7, 7.)])
def test_deterministic_parameter_becomes_a_dirac(n, value, expected_mean):
    """A deterministic parameter keeps its column in the realization matrix, with zero variance"""
    par_a, par_b = InputParameterNode('A'), InputParameterNode('B')

    dsg = BasicDSG()
    dsg.add_edges([(n[0], par_a), (n[0], par_b)])
    dsg = dsg.set_start_nodes({n[0]})
    dsg.set_input_parameter_value(par_a, ot.Normal(0., 1.))
    dsg.set_input_parameter_value(par_b, value)

    space = GraphProcessor(dsg).param_space
    assert space.n_parameters == 2
    assert space.parameters[1].mean() == pytest.approx(expected_mean)
    assert space.parameters[1].std() == pytest.approx(0.)

    samples = space.get_random_samples(10)
    assert samples.shape == (10, 2)
    assert np.all(samples[:, 1] == expected_mean)
    assert samples[:, 0].std() > 0.


"""#####################################
### 5. StochasticDSGEvaluator.evaluate #
#####################################"""


class StochasticBeamEvaluator(StochasticDSGEvaluator):
    """
    Small stochastic problem: pick a material and a thickness for a beam under an uncertain load.

    - `material`: selection choice between 'steel' (stiff, heavy) and 'alu' (compliant, light)
    - `t`: continuous thickness
    - `load`: uncertain applied load; `E_steel` / `E_alu`: stiffness, only present for the selected material
    - `mass` is deterministic; `deflection` is stochastic
    - `stress` is an optional constraint, present only in the steel branch
    """

    stiffness = {'steel': 210., 'alu': 70.}
    density = {'steel': 7.8, 'alu': 2.7}

    def __init__(self, stress_ref: float = None):
        self.seen_parameters: List[Dict] = []

        self.par_load = InputParameterNode('load')
        self.par_e = {'steel': InputParameterNode('E_steel'), 'alu': InputParameterNode('E_alu')}

        self.mass_node = MetricNode('mass', direction=-1, type_=MetricType.OBJECTIVE)
        self.deflection_node = MetricNode('deflection', direction=-1, type_=MetricType.OBJECTIVE)

        self.stress_ref = stress_ref
        self.stress_node = None
        if stress_ref is not None:
            self.stress_node = MetricNode('stress', direction=-1, ref=stress_ref, type_=MetricType.CONSTRAINT)

        self.material_nodes = {}
        super().__init__(self._build_dsg())

    def _build_dsg(self):
        dsg = BasicDSG()

        beam = NamedNode('beam')
        thickness = DesignVariableNode('t', bounds=(1., 5.))
        dsg.add_edges([(beam, self.mass_node), (beam, self.deflection_node),
                       (beam, self.par_load), (beam, thickness)])

        for name in ['steel', 'alu']:
            material_node = NamedNode(name)
            self.material_nodes[name] = material_node
            dsg.add_edge(material_node, self.par_e[name])

        if self.stress_node is not None:
            dsg.add_edge(self.material_nodes['steel'], self.stress_node)

        dsg.add_selection_choice('material', beam, [self.material_nodes['steel'], self.material_nodes['alu']])
        dsg = dsg.set_start_nodes({beam})

        dsg.set_input_parameter_value(self.par_load, ot.Normal(100., 20.))
        dsg.set_input_parameter_value(self.par_e['steel'], ot.Normal(210., 10.))
        dsg.set_input_parameter_value(self.par_e['alu'], ot.Normal(70., 10.))
        return dsg

    def _selected_material(self, dsg: DSGType) -> str:
        for name, material_node in self.material_nodes.items():
            if material_node in dsg.graph.nodes:
                return name
        raise RuntimeError('No material selected!')

    def _thickness(self, dsg: DSGType) -> float:
        for des_var_node, value in dsg.des_var_values.items():
            if des_var_node.name == 't':
                return value
        raise RuntimeError('No thickness!')

    def _evaluate(self, dsg, metric_nodes, parameters):
        self.seen_parameters.append(parameters)

        material = self._selected_material(dsg)
        thickness = self._thickness(dsg)
        load = parameters[self.par_load]
        e_modulus = parameters[self.par_e[material]]

        values = {
            self.mass_node: self.density[material]*thickness,
            self.deflection_node: load / (e_modulus*thickness**3),
        }
        if self.stress_node is not None and self.stress_node in metric_nodes:
            values[self.stress_node] = load / thickness**2
        return values


@pytest.fixture
def beam():
    return StochasticBeamEvaluator()


def _evaluate(evaluator, x, n_evaluations=20, seed=42, method_class=MonteCarlo, **kwargs):
    """Evaluate one architecture standalone, the way `evaluate` is meant to be called"""
    dsg, _, _ = evaluator.get_graph(x)
    uq_method = method_class(evaluator.param_space, n_evaluations=n_evaluations, seed=seed, **kwargs)
    return dsg, evaluator.evaluate(dsg, uq_method.get_samples(), uq_method)


def test_evaluate_passes_a_realization_not_an_index(beam):
    """Regression: the sample *index* was once handed to _evaluate, so every sample came out identical"""
    _evaluate(beam, [0, 3.], n_evaluations=15)

    assert len(beam.seen_parameters) == 15
    loads = [parameters[beam.par_load] for parameters in beam.seen_parameters]
    assert all(isinstance(load, float) for load in loads)
    assert len(set(loads)) == 15
    assert np.std(loads) > 0.


def test_evaluate_realizations_follow_the_parameter_space_order(beam):
    """A realization is an array indexed by position: the node each column belongs to has to be the one the
    parameter space names, or every parameter silently takes another's value"""
    dsg, _ = _evaluate(beam, [0, 3.], n_evaluations=5)

    space = beam.param_space
    samples = MonteCarlo(space, n_evaluations=5, seed=42).get_samples()
    by_name = {node.name: value for node, value in beam.seen_parameters[0].items()}

    for i, name in enumerate(space.parameter_names):
        assert by_name[name] == pytest.approx(samples[0, i])


def test_evaluate_returns_a_stochastic_results(beam):
    _, result = _evaluate(beam, [0, 3.])

    assert isinstance(result, StochasticResults)
    assert len(result.outputs) == len(beam.objectives) + len(beam.constraints)
    assert len(result.outputs[0].to_numpy()) == 20


def test_evaluate_stores_a_stochastic_output_per_metric(beam):
    dsg, _ = _evaluate(beam, [0, 3.])

    for metric_node in dsg.metric_nodes:
        value = dsg.metric_value(metric_node)
        assert isinstance(value, StochasticOutput)
        assert len(value.to_numpy()) == 20

    # The deflection scatters (it depends on the parameters), the mass does not
    assert dsg.metric_value(beam.deflection_node).std() > 0.
    assert dsg.metric_value(beam.mass_node).std() == pytest.approx(0.)


def test_evaluate_stores_physical_values(beam):
    """No objective/constraint sign conventions are applied by the evaluator"""
    dsg, _ = _evaluate(beam, [0, 3.])

    assert dsg.metric_value(beam.mass_node).mean() == pytest.approx(7.8*3.)
    assert np.all(dsg.metric_value(beam.deflection_node).to_numpy() > 0.)


def test_evaluated_instances_keep_their_own_outputs(beam):
    """Every evaluated architecture stores its own stochastic outputs"""
    steel, _ = _evaluate(beam, [0, 3.])
    alu, _ = _evaluate(beam, [1, 3.])

    means = [dsg.metric_value(beam.deflection_node).mean() for dsg in (steel, alu)]
    assert means[0] != means[1]
    assert means[0] < means[1]  # steel is stiffer, so it deflects less


def test_evaluate_writes_each_constraint_its_own_output():
    """Regression: constraints were written back with the objective's column index, so a constraint node silently
    ended up holding an objective's samples"""
    evaluator = StochasticBeamEvaluator(stress_ref=100.)
    n_obj = len(evaluator.objectives)
    assert len(evaluator.constraints) == 1

    steel, result = _evaluate(evaluator, [0, 3.])
    assert evaluator.stress_node in steel.metric_nodes

    stored = steel.metric_value(evaluator.stress_node)
    assert isinstance(stored, StochasticOutput)
    assert np.allclose(stored.to_numpy(), result.outputs[n_obj].to_numpy())

    # ... and that is the stress, not one of the objectives
    for i in range(n_obj):
        assert not np.allclose(stored.to_numpy(), result.outputs[i].to_numpy())


def test_evaluate_substitutes_absent_constraints_with_their_reference():
    """A constraint node that does not exist in an architecture takes its reference value, so it is satisfied"""
    evaluator = StochasticBeamEvaluator(stress_ref=100.)
    n_obj = len(evaluator.objectives)

    alu, result = _evaluate(evaluator, [1, 3.])  # no stress node in the alu branch
    assert evaluator.stress_node not in alu.metric_nodes

    assert np.all(result.outputs[n_obj].to_numpy() == 100.)
    assert alu.metric_value(evaluator.stress_node) is None


def test_export_handles_stochastic_outputs(beam):
    """Regression: the export once read a removed `assigned_statistics`, and formatted the value as a float"""
    dsg, _ = _evaluate(beam, [0, 3.], n_evaluations=10)

    dsg._get_graph_for_export()
    title = beam.deflection_node.get_export_title()
    assert 'μ=' in title and 'σ=' in title


"""#################################
### 6. Stochastic optimization   ###
#################################"""


def _problem(evaluator, n_evaluations=50, seed=42, **kwargs):
    uq_method = MonteCarlo(evaluator.param_space, n_evaluations=n_evaluations, seed=seed)
    return evaluator.get_problem(uq_method=uq_method, **kwargs)


def test_problem_shape_matches_the_evaluator():
    """Regression: n_obj and the constraint measures were once passed under keywords the base class does not
    take, so they landed in **kwargs and were silently ignored"""
    evaluator = StochasticBeamEvaluator(stress_ref=100.)
    problem = _problem(evaluator, n_evaluations=10)

    assert problem.n_obj == len(evaluator.objectives) == 2
    assert problem.n_ieq_constr == len(evaluator.constraints) == 1
    assert problem.n_var == len(evaluator.des_vars)


def test_problem_with_constraints_can_be_built():
    """Regression: con.dir is a Direction enum, so comparing it to an int raised and no constrained stochastic
    problem could be constructed at all"""
    evaluator = StochasticBeamEvaluator(stress_ref=100.)
    problem = _problem(evaluator, n_evaluations=10)

    assert problem.con_ref == [(False, 100.)]  # minimized constraint, so not flipped


def test_problem_uses_the_methods_parameter_space():
    """The problem has no parameter space of its own: it reads the one its UQ method propagates"""
    evaluator = StochasticBeamEvaluator()
    problem = _problem(evaluator, n_evaluations=10)

    assert problem.param_space is problem.uq_method.param_space
    assert problem.param_space.parameter_names == ['load', 'E_steel', 'E_alu']


def test_problem_measures_take_effect():
    """Regression: the constraint measure was once dropped, so everything reduced with Mean()"""
    evaluator = StochasticBeamEvaluator(stress_ref=100.)
    problem = _problem(evaluator, obj_measure=[Margin(k=2.), Mean()], constr_measure=[Quantile(q=.9)])

    assert [type(m).__name__ for m in problem.obj_measure] == ['Margin', 'Mean']
    assert [type(m).__name__ for m in problem.ieq_constr_measure] == ['Quantile']

    out = problem.evaluate(np.array([[0, 3.]]), return_as_dictionary=True)
    result = out['stochastic'][0]

    # Objectives are ordered by name, so deflection is first
    assert out['F'][0, 0] == pytest.approx(result.outputs[0].reduce(Margin(k=2.)))
    assert out['F'][0, 0] > result.outputs[0].mean()
    assert out['G'][0, 0] == pytest.approx(result.outputs[2].reduce(Quantile(q=.9)) - 100.)


def test_problem_evaluation():
    evaluator = StochasticBeamEvaluator()
    problem = _problem(evaluator)

    out = problem.evaluate(np.array([[0, 2.], [1, 4.]]), return_as_dictionary=True)

    assert out['F'].shape == (2, 2)
    assert np.all(np.isfinite(out['F']))

    # Objectives are ordered by name, so mass is the second one; it is minimized and therefore stored as-is
    assert [objective.name for objective in evaluator.objectives] == ['deflection', 'mass']
    assert out['F'][0, 1] == pytest.approx(7.8*2.)
    assert out['F'][1, 1] == pytest.approx(2.7*4.)


def test_problem_publishes_the_statistics():
    evaluator = StochasticBeamEvaluator()
    problem = _problem(evaluator)

    out = problem.evaluate(np.array([[0, 2.], [1, 4.]]), return_as_dictionary=True)

    assert len(out['stochastic']) == 2
    for result in out['stochastic']:
        assert isinstance(result, StochasticResults)
        assert len(result.outputs) == 2

    # Regression for the original blocker: realizations must actually reach the model
    deflection = out['stochastic'][0].outputs[0]
    assert len(deflection.to_numpy()) == 50
    assert len(set(deflection.to_numpy().tolist())) == 50
    assert deflection.std() > 0.


def test_problem_reported_statistics_reproduce_the_objectives():
    evaluator = StochasticBeamEvaluator()
    problem = _problem(evaluator, obj_measure=[Margin(k=1.5), Mean()])

    out = problem.evaluate(np.array([[0, 2.]]), return_as_dictionary=True)
    result = out['stochastic'][0]

    for j, measure in enumerate(problem.obj_measure):
        assert out['F'][0, j] == pytest.approx(result.outputs[j].reduce(measure))


def test_problem_uses_common_random_numbers():
    """All design points in a batch see the same realizations"""
    evaluator = StochasticBeamEvaluator()
    problem = _problem(evaluator, n_evaluations=20)

    problem.evaluate(np.array([[0, 2.], [0, 2.]]), return_as_dictionary=True)

    first, second = evaluator.seen_parameters[:20], evaluator.seen_parameters[20:]
    assert [p[evaluator.par_load] for p in first] == [p[evaluator.par_load] for p in second]


def test_problem_derives_each_architecture_once_per_evaluation():
    """The architecture does not depend on the realization, so it must not be rebuilt per sample"""
    evaluator = StochasticBeamEvaluator()
    problem = _problem(evaluator, n_evaluations=30)

    n_calls, original = [0], evaluator.get_graph

    def _counting(*args, **kwargs):
        n_calls[0] += 1
        return original(*args, **kwargs)

    evaluator.get_graph = _counting
    problem.evaluate(np.array([[0, 2.], [1, 4.]]), return_as_dictionary=True)

    assert n_calls[0] == 2


def test_problem_with_polynomial_chaos():
    """The UQ method is a plain parameter, so a different one needs no change anywhere else"""
    evaluator = StochasticBeamEvaluator()
    uq_method = PolynomialChaos(evaluator.param_space, n_evaluations=40, seed=42, degree=2)
    problem = evaluator.get_problem(uq_method=uq_method)

    out = problem.evaluate(np.array([[0, 2.]]), return_as_dictionary=True)

    assert np.all(np.isfinite(out['F']))
    result = out['stochastic'][0]

    # PCE takes its statistics from the fitted expansion, not from the 40 expensive evaluations
    assert len(result.outputs[0].to_numpy()) > 40
    assert result.method_result is not None


def test_uav_example_end_to_end():
    """The example is the only hierarchical end-to-end user of the feature"""
    from adsg_core.examples.robust_uav import RobustUAVEvaluator

    evaluator = RobustUAVEvaluator(k=2.)
    uq_method = MonteCarlo(evaluator.param_space, n_evaluations=25, seed=42)
    problem = evaluator.get_problem(uq_method=uq_method, obj_measure=evaluator.obj_measure)

    assert problem.n_obj == 2
    assert set(problem.param_space.parameter_names) == {'payload', 'headwind', 'drag_factor', 'eta_bat', 'bsfc'}

    x = np.array([evaluator.get_random_design_vector() for _ in range(4)], dtype=float)
    out = problem.evaluate(x, return_as_dictionary=True)

    assert out['F'].shape == (4, 2)
    assert np.all(np.isfinite(out['F']))

    # Endurance is maximized, so it is stored negated; the graph keeps the physical value
    assert np.all(out['F'][:, 0] < 0.)
    endurance = out['stochastic'][0].outputs[0]
    assert endurance.mean() > 0.
    assert endurance.std() > 0.


def test_uav_example_statistics_helper():
    from adsg_core.examples.robust_uav import RobustUAVEvaluator

    evaluator = RobustUAVEvaluator(k=2.)
    dsg, _, _ = evaluator.get_graph(evaluator.get_random_design_vector())

    statistics = evaluator.evaluate_statistics(dsg, MonteCarlo(evaluator.param_space, n_evaluations=25, seed=1))

    assert statistics['endurance_mean'] > 0.
    assert statistics['endurance_std'] > 0.
    assert statistics['endurance_robust'] < statistics['endurance_mean']
    assert statistics['mass'] > 0.