"""
Tests for InputParameter and the uncertainty-propagation machinery built on top of it.

Ordered from simple to general:
1.  InputParameter itself (it is an identity; the value lives on the graph)
2.  Storing parameter distributions on the DSG, and the per-instance stochastic space
3.  Parameter nodes under graph derivation (conditional existence)
4.  GraphProcessor integration (parameters are not design variables)
5.  StochasticDSGEvaluator.evaluate: one realization per sample, StochasticOutput per metric on the graph
6.  A full robust optimization problem through SBArchOpt
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
from adsg_core.optimization.graph_processor import *
from sb_arch_opt.uncertainty import (MonteCarlo, PolynomialChaos, StochasticOutput, StochasticResult,
                                     Mean, Margin, Quantile)


"""#################################
### 1. InputParameter            ###
#################################"""


def test_input_parameter_node_basics():
    par_node = InputParameter('E')

    assert par_node.name == 'E'
    assert par_node.idx is None
    assert str(par_node) == 'PARAM[E]'
    assert repr(par_node)
    assert par_node.get_export_color()
    assert par_node.str_context() == 'PARAM.E'


def test_input_parameter_node_export_title():
    """The node holds no value, so it only shows one once the graph assigned it for export"""
    par_node = InputParameter('E')
    assert par_node.get_export_title() == 'E'

    par_node.assigned_value = ot.Normal(10., 2.)
    assert 'E = ' in par_node.get_export_title()


def test_input_parameter_nodes_are_distinct():
    """Nodes are identity-based, so two parameters with the same name are still different nodes"""
    par_a = InputParameter('E')
    par_b = InputParameter('E')

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
    par_a, par_b = InputParameter('A'), InputParameter('B')
    dsg = _dsg_with_parameters(n, [par_a, par_b])

    assert dsg.feasible
    assert set(dsg.input_parameter_nodes) == {par_a, par_b}

    assert dsg.input_parameter_value(par_a) is None
    dist = ot.Normal(0., 1.)
    dsg.set_input_parameter_value(par_a, dist)
    assert dsg.input_parameter_value(par_a) is dist
    assert dsg.input_parameter_values == {par_a: dist}

    # A deterministic parameter is stored as a plain value
    dsg.set_input_parameter_value(par_b, 1.225)
    assert dsg.input_parameter_value(par_b) == 1.225

    dsg.reset_input_parameter_values()
    assert dsg.input_parameter_values == {}
    assert dsg.input_parameter_value(par_a) is None


def test_input_parameter_values_is_a_copy(n):
    par_a = InputParameter('A')
    dsg = _dsg_with_parameters(n, [par_a])

    dsg.set_input_parameter_value(par_a, 1.)
    values = dsg.input_parameter_values
    values[par_a] = 99.

    assert dsg.input_parameter_value(par_a) == 1.


def test_input_parameter_values_survive_copy(n):
    """Regression: the copy constructors used to pass the pre-rename keyword, silently dropping all values"""
    par_a = InputParameter('A')
    dsg = _dsg_with_parameters(n, [par_a])

    dsg.set_input_parameter_value(par_a, ot.Normal(0., 1.))
    dsg_copy = dsg.copy()

    assert dsg_copy.input_parameter_value(par_a) is not None
    assert dsg_copy.input_parameter_value(par_a).getMean()[0] == 0.


def test_input_parameter_nodes_sorted_by_name(n):
    """Realizations are arrays indexed by position, so the node order must be deterministic"""
    par_c, par_a, par_b = InputParameter('C'), InputParameter('A'), InputParameter('B')
    dsg = _dsg_with_parameters(n, [par_c, par_a, par_b])

    assert [par.name for par in dsg.input_parameter_nodes] == ['A', 'B', 'C']


def test_stochastic_space(n):
    par_a, par_b = InputParameter('A'), InputParameter('B')
    dsg = _dsg_with_parameters(n, [par_a, par_b])
    dsg.set_input_parameter_value(par_a, ot.Normal(10., 2.))
    dsg.set_input_parameter_value(par_b, ot.Uniform(0., 1.))

    space = dsg.stochastic_space
    assert space.n_parameters == 2
    assert space.parameter_names == ['A', 'B']
    assert space.parameters[0].mean() == pytest.approx(10.)
    assert space.parameters[0].std() == pytest.approx(2.)


def test_stochastic_space_is_not_cached(n):
    """Values are assigned after construction, so a cached space would freeze the wrong one"""
    par_a = InputParameter('A')
    dsg = _dsg_with_parameters(n, [par_a])

    assert dsg.stochastic_space.parameters[0].std() == pytest.approx(0.)  # Unassigned --> Dirac

    dsg.set_input_parameter_value(par_a, ot.Normal(0., 3.))
    assert dsg.stochastic_space.parameters[0].std() == pytest.approx(3.)


@pytest.mark.parametrize('value,expected_mean', [(None, 0.), (1.225, 1.225), (7, 7.)])
def test_deterministic_parameter_becomes_dirac(n, value, expected_mean):
    """A deterministic parameter keeps its column in the realization matrix, with zero variance"""
    par_a = InputParameter('A')
    dsg = _dsg_with_parameters(n, [par_a])
    if value is not None:
        dsg.set_input_parameter_value(par_a, value)

    space = dsg.stochastic_space
    assert space.n_parameters == 1
    assert space.parameters[0].mean() == pytest.approx(expected_mean)
    assert space.parameters[0].std() == pytest.approx(0.)

    # It joins a joint distribution without complaint, and its samples are constant
    samples = space.get_samples(10)
    assert samples.shape == (10, 1)
    assert np.all(samples == expected_mean)


"""#################################
### 3. Derivation / existence    ###
#################################"""


def test_parameter_node_conditional_existence(n):
    """A parameter node hung off a selection-choice option only exists if that option is selected"""
    par_common, par_opt_a, par_opt_b = InputParameter('common'), InputParameter('only_a'), InputParameter('only_b')

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

        # Only the parameter belonging to the selected option is present, and only it is in the space
        assert (par_opt_b not in par_nodes) if par_opt_a in par_nodes else (par_opt_b in par_nodes)
        assert graph.stochastic_space.n_parameters == 2

        # The distributions survived derivation
        assert graph.input_parameter_value(par_common) is not None

    assert seen == {par_common, par_opt_a, par_opt_b}


def test_parameter_values_isolated_between_instances(n):
    """Each derived instance carries its own values"""
    par_a = InputParameter('A')

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
    """Uncertain parameters must never show up as design variables: the optimizer does not choose them"""
    par_a = InputParameter('A')
    dv_node = DesignVariableNode('DV', bounds=(0., 1.))

    dsg = BasicDSG()
    dsg.add_edges([(n[0], par_a), (n[0], dv_node)])
    dsg.add_selection_choice('C1', n[0], [n[1], n[2]])
    dsg = dsg.set_start_nodes({n[0]})

    processor = GraphProcessor(dsg)

    assert [dv.name for dv in processor.des_vars] == ['C1', 'DV']
    assert all(dv.node is not par_a for dv in processor.des_vars)
    assert par_a not in processor.design_variable_nodes


def test_processor_uncertain_parameter_nodes_sorted(n):
    """The processor and the graph must agree on the parameter order (both sorted by name)"""
    par_c, par_a, par_b = InputParameter('C'), InputParameter('A'), InputParameter('B')

    dsg = BasicDSG()
    dsg.add_edges([(n[0], par_c), (n[0], par_a), (n[0], par_b)])
    dsg.add_selection_choice('C1', n[0], [n[1], n[2]])
    dsg = dsg.set_start_nodes({n[0]})

    processor = GraphProcessor(dsg)
    assert [par.name for par in processor.uncertain_parameter_nodes] == ['A', 'B', 'C']
    assert [par.name for par in dsg.input_parameter_nodes] == ['A', 'B', 'C']


"""#####################################
### 5. StochasticDSGEvaluator.evaluate #
#####################################"""


class RobustBeamEvaluator(StochasticDSGEvaluator):
    """
    Small robust optimization problem: pick a material and a thickness for a beam under an uncertain load.

    - `material`: selection choice between 'steel' (stiff, heavy) and 'alu' (compliant, light)
    - `t`: continuous thickness
    - `load`: uncertain applied load; `E_steel` / `E_alu`: stiffness, only present for the selected material
    - `mass` is deterministic; `deflection` is stochastic and also constrained
    """

    stiffness = {'steel': 210., 'alu': 70.}
    density = {'steel': 7.8, 'alu': 2.7}

    def __init__(self, deflection_ref: float = None):
        self.seen_parameters: List[Dict] = []

        self.par_load = InputParameter('load')
        self.par_e = {'steel': InputParameter('E_steel'), 'alu': InputParameter('E_alu')}

        self.mass_node = MetricNode('mass', direction=-1, type_=MetricType.OBJECTIVE)
        self.deflection_node = MetricNode('deflection', direction=-1, type_=MetricType.OBJECTIVE)

        # Optional constraint node, present only in the steel branch, to exercise absent-constraint handling
        self.deflection_ref = deflection_ref
        self.stress_node = None
        if deflection_ref is not None:
            self.stress_node = MetricNode('stress', direction=-1, ref=deflection_ref, type_=MetricType.CONSTRAINT)

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
    return RobustBeamEvaluator()


def _one_instance(evaluator, x=None):
    return evaluator.get_graph(x if x is not None else evaluator.get_random_design_vector())[0]


def test_evaluate_passes_a_realization_not_an_index(beam):
    """Regression: the sample *index* used to be handed to _evaluate, so every sample was identical"""
    dsg = _one_instance(beam, [0, 3.])
    uq = MonteCarlo(n_evaluations=15, seed=42)
    space = dsg.stochastic_space

    beam.evaluate(dsg, uq.get_samples(space), uq, space)

    assert len(beam.seen_parameters) == 15
    loads = [parameters[beam.par_load] for parameters in beam.seen_parameters]
    assert all(isinstance(load, float) for load in loads)
    assert len(set(loads)) == 15
    assert np.std(loads) > 0.


def test_evaluate_only_passes_parameters_of_this_architecture(beam):
    """Branch-local parameters are only handed to the architectures that have them"""
    for material_idx, name in enumerate(['steel', 'alu']):
        beam.seen_parameters.clear()
        dsg = _one_instance(beam, [material_idx, 3.])
        uq = MonteCarlo(n_evaluations=3, seed=1)
        space = dsg.stochastic_space

        beam.evaluate(dsg, uq.get_samples(space), uq, space)

        parameters = beam.seen_parameters[0]
        assert set(parameters) == {beam.par_load, beam.par_e[name]}


def test_evaluate_returns_a_stochastic_result(beam):
    dsg = _one_instance(beam, [0, 3.])
    uq = MonteCarlo(n_evaluations=20, seed=42)
    space = dsg.stochastic_space

    result = beam.evaluate(dsg, uq.get_samples(space), uq, space)

    assert isinstance(result, StochasticResult)
    assert len(result.outputs) == len(beam.objectives) + len(beam.constraints)
    assert len(result.outputs[0].to_numpy()) == 20


def test_evaluate_stores_a_stochastic_output_per_metric(beam):
    dsg = _one_instance(beam, [0, 3.])
    uq = MonteCarlo(n_evaluations=20, seed=42)
    space = dsg.stochastic_space

    beam.evaluate(dsg, uq.get_samples(space), uq, space)

    for metric_node in dsg.metric_nodes:
        value = dsg.metric_value(metric_node)
        assert isinstance(value, StochasticOutput)
        assert len(value.to_numpy()) == 20

    # The deflection scatters (it depends on the parameters), the mass does not
    assert dsg.metric_value(beam.deflection_node).std() > 0.
    assert dsg.metric_value(beam.mass_node).std() == pytest.approx(0.)


def test_evaluated_instances_keep_their_own_outputs(beam):
    """Every evaluated architecture stores its own stochastic outputs"""
    uq = MonteCarlo(n_evaluations=20, seed=42)
    instances = []
    for material_idx in range(2):
        dsg = _one_instance(beam, [material_idx, 3.])
        space = dsg.stochastic_space
        beam.evaluate(dsg, uq.get_samples(space), uq, space)
        instances.append(dsg)

    means = [dsg.metric_value(beam.deflection_node).mean() for dsg in instances]
    assert means[0] != means[1]

    # Steel is stiffer, so it deflects less
    assert means[0] < means[1]


def test_evaluate_stores_physical_values(beam):
    """No objective/constraint sign conventions are applied by the evaluator"""
    dsg = _one_instance(beam, [0, 3.])
    uq = MonteCarlo(n_evaluations=20, seed=42)
    space = dsg.stochastic_space

    beam.evaluate(dsg, uq.get_samples(space), uq, space)

    assert dsg.metric_value(beam.mass_node).mean() == pytest.approx(7.8*3.)
    assert np.all(dsg.metric_value(beam.deflection_node).to_numpy() > 0.)


def test_evaluate_defaults_to_the_instance_space(beam):
    dsg = _one_instance(beam, [0, 3.])
    uq = MonteCarlo(n_evaluations=10, seed=42)

    result = beam.evaluate(dsg, uq.get_samples(dsg.stochastic_space), uq)
    assert len(result.outputs[0].to_numpy()) == 10


def test_evaluate_rejects_a_mismatched_sample_width(beam):
    dsg = _one_instance(beam, [0, 3.])
    uq = MonteCarlo(n_evaluations=5, seed=42)

    with pytest.raises(ValueError, match='parameter space'):
        beam.evaluate(dsg, np.zeros((5, 7)), uq, dsg.stochastic_space)


def test_evaluate_substitutes_absent_constraints_with_their_reference():
    """A constraint node that does not exist in an architecture takes its reference value, so it is satisfied"""
    evaluator = RobustBeamEvaluator(deflection_ref=100.)
    assert len(evaluator.constraints) == 1

    uq = MonteCarlo(n_evaluations=10, seed=42)
    n_obj = len(evaluator.objectives)

    alu = evaluator.get_graph([1, 3.])[0]  # No stress node in the alu branch
    assert evaluator.stress_node not in alu.metric_nodes
    result = evaluator.evaluate(alu, uq.get_samples(alu.stochastic_space), uq, alu.stochastic_space)
    assert np.all(result.outputs[n_obj].to_numpy() == 100.)
    assert alu.metric_value(evaluator.stress_node) is None

    steel = evaluator.get_graph([0, 3.])[0]
    assert evaluator.stress_node in steel.metric_nodes
    result = evaluator.evaluate(steel, uq.get_samples(steel.stochastic_space), uq, steel.stochastic_space)
    assert result.outputs[n_obj].std() > 0.
    assert isinstance(steel.metric_value(evaluator.stress_node), StochasticOutput)


def test_export_handles_stochastic_outputs(beam):
    """Regression: the export used to read a removed `assigned_statistics` attribute and to format a float"""
    dsg = _one_instance(beam, [0, 3.])
    uq = MonteCarlo(n_evaluations=10, seed=42)
    beam.evaluate(dsg, uq.get_samples(dsg.stochastic_space), uq)

    dsg._get_graph_for_export()
    title = beam.deflection_node.get_export_title()
    assert 'μ=' in title and 'σ=' in title


"""#################################
### 6. Robust optimization       ###
#################################"""


def test_problem_shape_matches_the_evaluator():
    """Regression: n_obj/measures used to be passed under the wrong keyword and silently ignored"""
    evaluator = RobustBeamEvaluator(deflection_ref=100.)
    problem = evaluator.get_problem(uq_method=MonteCarlo(n_evaluations=10, seed=42))

    assert problem.n_obj == len(evaluator.objectives) == 2
    assert problem.n_ieq_constr == len(evaluator.constraints) == 1
    assert problem.n_var == len(evaluator.des_vars)


def test_problem_measures_take_effect():
    """Regression: the constraint measure used to be dropped, so everything reduced with Mean()"""
    evaluator = RobustBeamEvaluator(deflection_ref=100.)
    problem = evaluator.get_problem(uq_method=MonteCarlo(n_evaluations=50, seed=42),
                                    obj_measure=[Margin(k=2.), Mean()],
                                    constr_measure=[Quantile(q=.9)])

    assert [type(m).__name__ for m in problem.obj_measure] == ['Margin', 'Mean']
    assert [type(m).__name__ for m in problem.ieq_constr_measure] == ['Quantile']

    x = np.array([[0, 3.]])
    out = problem.evaluate(x, return_as_dictionary=True)

    # Margin(k=2) on the deflection is strictly above its mean
    result = out['stochastic'][0]
    assert out['F'][0, 0] == pytest.approx(result.outputs[0].reduce(Margin(k=2.)))
    assert out['F'][0, 0] > result.outputs[0].mean()
    assert out['G'][0, 0] == pytest.approx(result.outputs[2].reduce(Quantile(q=.9)) - 100.)


def test_problem_union_parameter_space():
    """The problem samples every parameter of the template graph, so one matrix serves every architecture"""
    evaluator = RobustBeamEvaluator()
    problem = evaluator.get_problem(uq_method=MonteCarlo(n_evaluations=10, seed=42))

    assert problem.param_space.parameter_names == ['E_alu', 'E_steel', 'load']


def test_problem_evaluation():
    evaluator = RobustBeamEvaluator()
    problem = evaluator.get_problem(uq_method=MonteCarlo(n_evaluations=50, seed=42))

    x = np.array([[0, 2.], [1, 4.]])
    out = problem.evaluate(x, return_as_dictionary=True)

    assert out['F'].shape == (2, 2)
    assert np.all(np.isfinite(out['F']))

    # Objectives are ordered by name, so mass is the second one. It is minimized and therefore stored as-is
    assert [objective.name for objective in evaluator.objectives] == ['deflection', 'mass']
    assert out['F'][0, 1] == pytest.approx(7.8*2.)
    assert out['F'][1, 1] == pytest.approx(2.7*4.)


def test_problem_publishes_the_statistics():
    evaluator = RobustBeamEvaluator()
    problem = evaluator.get_problem(uq_method=MonteCarlo(n_evaluations=50, seed=42))

    out = problem.evaluate(np.array([[0, 2.], [1, 4.]]), return_as_dictionary=True)

    assert len(out['stochastic']) == 2
    for result in out['stochastic']:
        assert len(result.outputs) == 2

    # Regression for the blocker: realizations must actually reach the model
    deflection = out['stochastic'][0].outputs[0]
    assert len(deflection.to_numpy()) == 50
    assert len(set(deflection.to_numpy().tolist())) == 50
    assert deflection.std() > 0.


def test_problem_reported_statistics_reproduce_the_objective():
    evaluator = RobustBeamEvaluator()
    problem = evaluator.get_problem(uq_method=MonteCarlo(n_evaluations=50, seed=42),
                                    obj_measure=[Margin(k=1.5), Mean()])

    out = problem.evaluate(np.array([[0, 2.]]), return_as_dictionary=True)
    result = out['stochastic'][0]

    for j, measure in enumerate(problem.obj_measure):
        assert out['F'][0, j] == pytest.approx(result.outputs[j].reduce(measure))


def test_problem_uses_common_random_numbers():
    """All design points in a batch see the same realizations"""
    evaluator = RobustBeamEvaluator()
    problem = evaluator.get_problem(uq_method=MonteCarlo(n_evaluations=20, seed=42))

    problem.evaluate(np.array([[0, 2.], [0, 2.]]), return_as_dictionary=True)

    # Two identical design points, so both saw the same realizations in the same order
    first, second = evaluator.seen_parameters[:20], evaluator.seen_parameters[20:]
    assert [p[evaluator.par_load] for p in first] == [p[evaluator.par_load] for p in second]


def test_problem_derives_each_architecture_once_per_evaluation():
    """The architecture does not depend on the realization, so it must not be rebuilt per sample"""
    evaluator = RobustBeamEvaluator()
    problem = evaluator.get_problem(uq_method=MonteCarlo(n_evaluations=30, seed=42))

    n_calls, original = [0], evaluator.get_graph

    def _counting(*args, **kwargs):
        n_calls[0] += 1
        return original(*args, **kwargs)

    evaluator.get_graph = _counting
    problem.evaluate(np.array([[0, 2.], [1, 4.]]), return_as_dictionary=True)

    assert n_calls[0] == 2


def test_problem_with_polynomial_chaos():
    """The UQ method is a plain parameter, so a different one needs no change anywhere else"""
    evaluator = RobustBeamEvaluator()
    problem = evaluator.get_problem(uq_method=PolynomialChaos(n_evaluations=40, seed=42, degree=2))

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
    problem = evaluator.get_problem(uq_method=MonteCarlo(n_evaluations=25, seed=42),
                                    obj_measure=evaluator.obj_measure)

    assert problem.n_obj == 2
    assert problem.param_space.parameter_names == ['bsfc', 'drag_factor', 'eta_bat', 'headwind', 'payload']

    x = np.array([evaluator.get_random_design_vector() for _ in range(4)], dtype=float)
    out = problem.evaluate(x, return_as_dictionary=True)

    assert out['F'].shape == (4, 2)
    assert np.all(np.isfinite(out['F']))

    # Endurance is maximized, so it is stored negated; the graph keeps the physical value
    assert np.all(out['F'][:, 0] < 0.)
    endurance = out['stochastic'][0].outputs[0]
    assert endurance.mean() > 0.
    assert endurance.std() > 0.
