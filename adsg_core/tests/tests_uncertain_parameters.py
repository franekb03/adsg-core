"""
Tests for UncertainParameterNode and the uncertainty-propagation machinery built on top of it.

Ordered from simple to general:
1.  UncertainParameterNode itself (construction, sampling, export)
2.  Storing sampled values on the DSG
3.  Parameter nodes under the graph derivation (conditional existence)
4.  GraphProcessor integration (parameters are not design variables)
5.  Stochastic metric handling (StochasticMetricType)
6.  Monte Carlo uncertainty propagation
7.  A full robust optimization problem: MC is run for every selected design point
"""
import math
import pytest
import numpy as np
from typing import *
import chaospy as cp

from adsg_core.graph.adsg import DSGType
from adsg_core.graph.adsg_basic import *
from adsg_core.graph.adsg_nodes import *
from adsg_core.optimization.evaluator import *
from adsg_core.optimization.graph_processor import *
from adsg_core.optimization.uq_method import UQMethod


@pytest.fixture(autouse=True)
def _seed():
    """Chaospy's 'random' rule draws from the numpy global RNG, so seeding it makes the tests reproducible"""
    np.random.seed(42)


"""#################################
### 1. UncertainParameterNode    ###
#################################"""


def test_uncertain_parameter_node_basics():
    par_node = UncertainParameterNode('E', distribution=cp.Normal(10., 2.))

    assert par_node.name == 'E'
    assert par_node.idx is None
    assert par_node.is_uncertain
    assert par_node.sampled_value is None
    assert str(par_node) == 'PARAM[E]'
    assert repr(par_node)
    assert par_node.get_export_color()


def test_uncertain_parameter_node_nominal():
    """A parameter node without a distribution represents a fixed (deterministic) parameter"""
    par_node = UncertainParameterNode('rho', nominal=1.225)

    assert not par_node.is_uncertain
    assert par_node.nominal == 1.225
    assert par_node.get_export_title() == 'rho = 1.225'


def test_uncertain_parameter_node_sample_shape():
    par_node = UncertainParameterNode('E', distribution=cp.Normal(10., 2.))

    for n in [1, 5, 100]:
        values = par_node.sample(n)
        assert isinstance(values, np.ndarray)
        assert values.shape == (n,)
        assert np.all(np.isfinite(values))


def test_uncertain_parameter_node_sample_statistics():
    """Sampling many times should recover the distribution moments"""
    mu, sigma = 10., 2.
    par_node = UncertainParameterNode('E', distribution=cp.Normal(mu, sigma))

    values = par_node.sample(20000)
    assert values.mean() == pytest.approx(mu, abs=.1)
    assert values.std() == pytest.approx(sigma, abs=.1)


def test_uncertain_parameter_node_uniform():
    par_node = UncertainParameterNode('t', distribution=cp.Uniform(2., 4.))

    values = par_node.sample(5000)
    assert np.all(values >= 2.)
    assert np.all(values <= 4.)
    assert values.mean() == pytest.approx(3., abs=.1)


def test_uncertain_parameter_node_export_title():
    par_node = UncertainParameterNode('E', distribution=cp.Normal(10., 2.))
    assert par_node.get_export_title()

    # Once a value has been assigned, it is shown instead of the distribution
    par_node.sampled_value = 11.5
    assert par_node.get_export_title() == 'E = 11.5'


def test_uncertain_parameter_nodes_are_distinct():
    """Nodes are identity-based, so two parameters with the same name are still different nodes"""
    par_a = UncertainParameterNode('E', distribution=cp.Normal(0., 1.))
    par_b = UncertainParameterNode('E', distribution=cp.Normal(0., 1.))

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


def test_set_get_uncertain_parameter_value(n):
    par_a = UncertainParameterNode('A', distribution=cp.Normal(0., 1.))
    par_b = UncertainParameterNode('B', distribution=cp.Normal(5., 1.))
    dsg = _dsg_with_parameters(n, [par_a, par_b])

    assert dsg.feasible
    assert set(dsg.uncertain_parameter_nodes) == {par_a, par_b}

    assert dsg.uncertain_parameter_value(par_a) is None
    dsg.set_uncertain_parameter_value(par_a, .5)
    assert dsg.uncertain_parameter_value(par_a) == .5
    assert dsg.uncertain_parameter_values == {par_a: .5}

    dsg.set_uncertain_parameter_value(par_b, math.nan)
    assert math.isnan(dsg.uncertain_parameter_value(par_b))

    dsg.reset_uncertain_parameter_values()
    assert dsg.uncertain_parameter_values == {}
    assert dsg.uncertain_parameter_value(par_a) is None


def test_uncertain_parameter_values_is_a_copy(n):
    par_a = UncertainParameterNode('A', distribution=cp.Normal(0., 1.))
    dsg = _dsg_with_parameters(n, [par_a])

    dsg.set_uncertain_parameter_value(par_a, 1.)
    values = dsg.uncertain_parameter_values
    values[par_a] = 99.

    assert dsg.uncertain_parameter_value(par_a) == 1.


def test_uncertain_parameter_values_survive_copy(n):
    par_a = UncertainParameterNode('A', distribution=cp.Normal(0., 1.))
    dsg = _dsg_with_parameters(n, [par_a])

    dsg.set_uncertain_parameter_value(par_a, .5)
    dsg_copy = dsg.copy()

    assert dsg_copy.uncertain_parameter_value(par_a) == .5


def test_sample_parameters(n):
    par_a = UncertainParameterNode('A', distribution=cp.Normal(0., 1.))
    par_b = UncertainParameterNode('B', distribution=cp.Uniform(10., 20.))
    dsg = _dsg_with_parameters(n, [par_a, par_b])

    values = dsg.sample_parameters()

    assert set(values) == {par_a, par_b}
    assert all(isinstance(value, float) for value in values.values())
    assert 10. <= values[par_b] <= 20.

    # Sampled values are stored on the graph instance
    assert dsg.uncertain_parameter_values == values


def test_sample_parameters_overwrites_previous_sample(n):
    """Repeated sampling on one instance keeps only the latest draw"""
    par_a = UncertainParameterNode('A', distribution=cp.Normal(0., 1.))
    dsg = _dsg_with_parameters(n, [par_a])

    first = dsg.sample_parameters()[par_a]
    second = dsg.sample_parameters()[par_a]

    assert first != second
    assert dsg.uncertain_parameter_value(par_a) == second


def test_sample_parameters_no_parameters(n):
    dsg = BasicDSG()
    dsg.add_edges([(n[0], n[1])])
    dsg = dsg.set_start_nodes({n[0]})

    assert dsg.uncertain_parameter_nodes == []
    assert dsg.sample_parameters() == {}


"""#################################
### 3. Derivation / existence    ###
#################################"""


def test_parameter_node_conditional_existence(n):
    """A parameter node hung off a selection-choice option only exists if that option is selected"""
    par_common = UncertainParameterNode('common', distribution=cp.Normal(0., 1.))
    par_opt_a = UncertainParameterNode('only_a', distribution=cp.Normal(1., 1.))
    par_opt_b = UncertainParameterNode('only_b', distribution=cp.Normal(2., 1.))

    dsg = BasicDSG()
    dsg.add_edges([
        (n[0], par_common),
        (n[1], par_opt_a),
        (n[2], par_opt_b),
    ])
    dsg.add_selection_choice('C1', n[0], [n[1], n[2]])
    dsg = dsg.set_start_nodes({n[0]})

    processor = GraphProcessor(dsg)
    assert len(processor.des_vars) == 1

    seen = set()
    for opt_idx in range(2):
        graph, _, _ = processor.get_graph([opt_idx])

        par_nodes = set(graph.uncertain_parameter_nodes)
        assert par_common in par_nodes
        seen |= par_nodes

        # Only the parameter belonging to the selected option is present
        if par_opt_a in par_nodes:
            assert par_opt_b not in par_nodes
        else:
            assert par_opt_b in par_nodes

        # Sampling only touches the parameters that exist in this architecture
        assert set(graph.sample_parameters()) == par_nodes

    assert seen == {par_common, par_opt_a, par_opt_b}


def test_parameter_values_isolated_between_instances(n):
    """Each derived instance carries its own sampled values"""
    par_a = UncertainParameterNode('A', distribution=cp.Normal(0., 1.))

    dsg = BasicDSG()
    dsg.add_edges([(n[0], par_a)])
    dsg.add_selection_choice('C1', n[0], [n[1], n[2]])
    dsg = dsg.set_start_nodes({n[0]})

    processor = GraphProcessor(dsg)
    graph_a, _, _ = processor.get_graph([0])
    graph_b, _, _ = processor.get_graph([1])

    graph_a.set_uncertain_parameter_value(par_a, 1.)
    graph_b.set_uncertain_parameter_value(par_a, 2.)

    assert graph_a.uncertain_parameter_value(par_a) == 1.
    assert graph_b.uncertain_parameter_value(par_a) == 2.

    # The template graph is never touched by evaluating instances
    assert dsg.uncertain_parameter_value(par_a) is None


"""#################################
### 4. GraphProcessor            ###
#################################"""


def test_parameters_are_not_design_variables(n):
    """Uncertain parameters must never show up as design variables: the optimizer does not choose them"""
    par_a = UncertainParameterNode('A', distribution=cp.Normal(0., 1.))
    dv_node = DesignVariableNode('DV', bounds=(0., 1.))

    dsg = BasicDSG()
    dsg.add_edges([(n[0], par_a), (n[0], dv_node)])
    dsg.add_selection_choice('C1', n[0], [n[1], n[2]])
    dsg = dsg.set_start_nodes({n[0]})

    processor = GraphProcessor(dsg)

    des_var_names = [dv.name for dv in processor.des_vars]
    assert des_var_names == ['C1', 'DV']
    assert all(dv.node is not par_a for dv in processor.des_vars)
    assert par_a not in processor.design_variable_nodes


def test_processor_uncertain_parameter_nodes_sorted(n):
    """The processor exposes parameter nodes sorted by name (needs a sort key: DSGNode has no ordering)"""
    par_c = UncertainParameterNode('C', distribution=cp.Normal(0., 1.))
    par_a = UncertainParameterNode('A', distribution=cp.Normal(0., 1.))
    par_b = UncertainParameterNode('B', distribution=cp.Normal(0., 1.))

    dsg = BasicDSG()
    dsg.add_edges([(n[0], par_c), (n[0], par_a), (n[0], par_b)])
    dsg.add_selection_choice('C1', n[0], [n[1], n[2]])
    dsg = dsg.set_start_nodes({n[0]})

    processor = GraphProcessor(dsg)
    assert [par.name for par in processor.uncertain_parameter_nodes] == ['A', 'B', 'C']


"""#################################
### 5. Stochastic metrics        ###
#################################"""


class _NoopEvaluator(DSGEvaluator):

    def _evaluate(self, dsg: DSGType, metric_nodes: List[MetricNode]) -> Dict[MetricNode, float]:
        return {}


@pytest.fixture
def noop_evaluator(n):
    dsg = BasicDSG()
    dsg.add_edges([(n[0], MetricNode('M', direction=-1, type_=MetricType.OBJECTIVE))])
    dsg.add_selection_choice('C1', n[0], [n[1], n[2]])
    dsg = dsg.set_start_nodes({n[0]})
    return _NoopEvaluator(dsg)


def test_metric_node_stochastic_attributes():
    metric_node = MetricNode('M', direction=-1, stochastic_=StochasticMetricType.MARGIN, k=3.)

    assert metric_node.stochastic == StochasticMetricType.MARGIN
    assert metric_node.k == 3.

    plain_node = MetricNode('M2', direction=-1)
    assert plain_node.stochastic is None
    assert plain_node.k is None


def test_process_stochastic_qoi_mean(noop_evaluator):
    metric_node = MetricNode('M', direction=-1, stochastic_=StochasticMetricType.MEAN)
    assert noop_evaluator.process_stochastic_qoi(metric_node, mean=5., std=2.) == 5.


@pytest.mark.parametrize('k', [0., 1., 3.])
def test_process_stochastic_qoi_margin(noop_evaluator, k):
    """For a minimization/maximization metric, the margin formulation penalizes spread: mean +/- k*std"""
    metric_node = MetricNode('M', direction=-1, stochastic_=StochasticMetricType.MARGIN, k=k)
    assert noop_evaluator.process_stochastic_qoi(metric_node, mean=5., std=2.) == 5. + k*2.

    metric_node2 = MetricNode('M2', direction=1, stochastic_=StochasticMetricType.MARGIN, k=k)
    assert noop_evaluator.process_stochastic_qoi(metric_node2, mean=5., std=2.) == 5. - k*2.

def test_process_stochastic_qoi_margin_penalizes_spread(noop_evaluator):
    metric_node = MetricNode('M', direction=-1, stochastic_=StochasticMetricType.MARGIN, k=2.)

    robust = noop_evaluator.process_stochastic_qoi(metric_node, mean=5., std=2.)
    less_spread = noop_evaluator.process_stochastic_qoi(metric_node, mean=5., std=.5)

    # Same mean, less scatter --> better (lower) value for a minimization metric
    assert less_spread < robust

def test_process_stochastic_qoi_margin_compared_with_mean(noop_evaluator):
    metric_node_mean = MetricNode("M1", direction=-1, stochastic_=StochasticMetricType.MEAN, k=2.)
    metric_node_margin_max = MetricNode("M2", direction=1, stochastic_=StochasticMetricType.MARGIN, k=2.)
    metric_node_margin_min = MetricNode("M3", direction=-1, stochastic_=StochasticMetricType.MARGIN, k=2.)

    mean = noop_evaluator.process_stochastic_qoi(metric_node_mean, mean=5., std=2.)
    max = noop_evaluator.process_stochastic_qoi(metric_node_margin_max, mean=5., std=2.)
    min = noop_evaluator.process_stochastic_qoi(metric_node_margin_min, mean=5., std=2.)

    assert max < mean < min


def test_process_stochastic_qoi_quantile_not_implemented(noop_evaluator):
    metric_node = MetricNode('M', direction=-1, stochastic_=StochasticMetricType.QUANTILE)
    with pytest.raises(NotImplementedError):
        noop_evaluator.process_stochastic_qoi(metric_node, mean=5., std=2.)


"""#################################
### 6. Monte Carlo propagation   ###
#################################"""


def test_uq_method_mc_statistics(n):
    """MC over a pass-through function recovers the parameter distribution moments"""
    par_a = UncertainParameterNode('A', distribution=cp.Normal(10., 2.))
    dsg = _dsg_with_parameters(n, [par_a])

    mean, std = UQMethod.mc(dsg, lambda dsg_, sample: sample[par_a], n=5000)

    assert mean == pytest.approx(10., abs=.15)
    assert std == pytest.approx(2., abs=.15)


def test_uq_method_mc_deterministic_function(n):
    """A function that ignores the parameters has zero variance"""
    par_a = UncertainParameterNode('A', distribution=cp.Normal(10., 2.))
    dsg = _dsg_with_parameters(n, [par_a])

    mean, std = UQMethod.mc(dsg, lambda dsg_, sample: 7., n=50)

    assert mean == 7.
    assert std == 0.


def test_uq_method_mc_uses_sampled_values(n):
    """Every MC iteration draws a fresh sample and passes it to the evaluation function"""
    par_a = UncertainParameterNode('A', distribution=cp.Normal(0., 1.))
    dsg = _dsg_with_parameters(n, [par_a])

    seen = []

    def _func(dsg_, sample):
        seen.append(sample[par_a])
        # The sample is also readable off the graph instance
        assert dsg_.uncertain_parameter_value(par_a) == sample[par_a]
        return sample[par_a]

    UQMethod.mc(dsg, _func, n=25)

    assert len(seen) == 25
    assert len(set(seen)) == 25  # All draws distinct


def test_uq_method_mc_scales_with_parameter_spread(n):
    """More parameter scatter propagates to more output scatter"""
    def _std_for(sigma):
        par = UncertainParameterNode('A', distribution=cp.Normal(0., sigma))
        dsg = _dsg_with_parameters(n, [par])
        _, std = UQMethod.mc(dsg, lambda dsg_, sample: 2.*sample[par], n=4000)
        return std

    assert _std_for(1.) < _std_for(4.)


def test_uq_method_mc_multiple_parameters(n):
    """Independent parameters combine: var(a+b) = var(a) + var(b)"""
    par_a = UncertainParameterNode('A', distribution=cp.Normal(0., 3.))
    par_b = UncertainParameterNode('B', distribution=cp.Normal(0., 4.))
    dsg = _dsg_with_parameters(n, [par_a, par_b])

    mean, std = UQMethod.mc(dsg, lambda dsg_, sample: sample[par_a] + sample[par_b], n=20000)

    assert mean == pytest.approx(0., abs=.2)
    assert std == pytest.approx(5., abs=.25)  # sqrt(3^2 + 4^2)


"""#################################
### 7. Robust optimization       ###
#################################"""


class RobustBeamEvaluator(DSGEvaluator):
    """
    Small robust optimization problem: pick a material and a thickness for a beam under an uncertain load.

    Design variables:
    - `material`: selection choice between 'steel' (stiff, heavy) and 'alu' (compliant, light)
    - `t`: continuous thickness

    Uncertain parameters:
    - `load`: applied load (normal)
    - `E_steel` / `E_alu`: material stiffness, only present for the selected material

    Metrics:
    - `mass` is deterministic (no uncertainty)
    - `deflection` is stochastic: evaluated by Monte Carlo and reduced with the metric's
      StochasticMetricType (MEAN or MARGIN) --> this is what makes the problem *robust*
    """

    n_mc = 200

    stiffness = {'steel': 210., 'alu': 70.}
    density = {'steel': 7.8, 'alu': 2.7}

    def __init__(self, k=2., stochastic=StochasticMetricType.MARGIN, n_mc=None):
        self.k = k
        self.stochastic = stochastic
        if n_mc is not None:
            self.n_mc = n_mc

        self.n_evaluations = 0
        self.mc_samples_seen = []

        self.par_load = UncertainParameterNode('load', distribution=cp.Normal(100., 20.))
        self.par_e = {
            'steel': UncertainParameterNode('E_steel', distribution=cp.Normal(210., 10.)),
            'alu': UncertainParameterNode('E_alu', distribution=cp.Normal(70., 10.)),
        }
        self.mass_node = MetricNode('mass', direction=-1, type_=MetricType.OBJECTIVE)
        self.deflection_node = MetricNode(
            'deflection', direction=-1, type_=MetricType.OBJECTIVE, stochastic_=stochastic, k=k)
        self.material_nodes = {}

        super().__init__(self._build_dsg())

    def _build_dsg(self):
        dsg = BasicDSG()

        beam = NamedNode('beam')
        thickness = DesignVariableNode('t', bounds=(1., 5.))

        dsg.add_edges([
            (beam, self.mass_node),
            (beam, self.deflection_node),
            (beam, self.par_load),
            (beam, thickness),
        ])

        # Each material carries its own (uncertain) stiffness parameter
        for name in ['steel', 'alu']:
            material_node = NamedNode(name)
            self.material_nodes[name] = material_node
            dsg.add_edge(material_node, self.par_e[name])

        dsg.add_selection_choice('material', beam, [self.material_nodes['steel'], self.material_nodes['alu']])
        return dsg.set_start_nodes({beam})

    def _selected_material(self, dsg: DSGType) -> str:
        for name, material_node in self.material_nodes.items():
            if material_node in dsg.graph.nodes:
                return name
        raise RuntimeError('No material selected!')

    def _thickness(self, dsg: DSGType) -> float:
        for des_var_node, value in dsg.des_var_values.items():
            if des_var_node.name == 't':
                return value
        raise RuntimeError('Thickness not set!')

    def _deflection(self, dsg: DSGType, sample: Dict[UncertainParameterNode, float]) -> float:
        """Deflection under the sampled load and stiffness: delta = load / (E * t^3)"""
        material = self._selected_material(dsg)
        load = sample[self.par_load]
        e_modulus = sample[self.par_e[material]]
        return 1e3*load / (e_modulus * self._thickness(dsg)**3)

    def _evaluate(self, dsg: DSGType, metric_nodes: List[MetricNode]) -> Dict[MetricNode, float]:
        self.n_evaluations += 1

        # Deterministic metric: no Monte Carlo needed
        material = self._selected_material(dsg)
        mass = self.density[material] * self._thickness(dsg)

        # Stochastic metric: run Monte Carlo for THIS design point
        stochastic_nodes = [mn for mn in metric_nodes if mn.stochastic is not None]
        value_map = {}
        for stochastic_node in stochastic_nodes:
            value_map.update(self.propagate_uncertainty("MC", self._deflection, dsg, stochastic_node, n=self.n_mc))

        self.mc_samples_seen.append(len(dsg.uncertain_parameter_values))
        value_map[self.mass_node] = mass
        return value_map


@pytest.fixture
def robust_evaluator():
    return RobustBeamEvaluator(n_mc=100)


def test_robust_problem_structure(robust_evaluator):
    """The uncertain parameters do not enlarge the design space"""
    des_vars = robust_evaluator.des_vars

    assert [dv.name for dv in des_vars] == ['material', 't']
    assert des_vars[0].is_discrete
    assert des_vars[0].n_opts == 2
    assert not des_vars[1].is_discrete
    assert tuple(des_vars[1].bounds) == (1., 5.)

    assert len(robust_evaluator.objectives) == 2
    assert {obj.name for obj in robust_evaluator.objectives} == {'mass', 'deflection'}
    assert len(robust_evaluator.constraints) == 0

    # 3 parameters in the design space graph, but only 2 in any single architecture
    assert len(robust_evaluator.uncertain_parameter_nodes) == 3


def test_robust_problem_monte_carlo_per_design_point(robust_evaluator):
    """Each evaluated design point gets its own Monte Carlo run"""
    for opt_idx in range(2):
        graph, _, _ = robust_evaluator.get_graph([opt_idx, 3.])
        obj, con = robust_evaluator.evaluate(graph)

        assert len(obj) == 2
        assert len(con) == 0
        assert all(np.isfinite(obj))

        # Only the parameters existing in this architecture were sampled
        assert len(graph.uncertain_parameter_values) == 2
        assert robust_evaluator.par_load in graph.uncertain_parameter_values

    assert robust_evaluator.n_evaluations == 2


def test_robust_problem_metric_values_stored_on_instance(robust_evaluator):
    graph, _, _ = robust_evaluator.get_graph([0, 3.])
    robust_evaluator.evaluate(graph)

    metric_values = graph.metric_values
    assert len(metric_values) == 2
    assert all(np.isfinite(value) for value in metric_values.values())

    # The template graph is not mutated by evaluating an instance
    assert robust_evaluator.graph.metric_values == {}


def test_robust_problem_thicker_beam_deflects_less(robust_evaluator):
    """Sanity check of the underlying model through the full MC pipeline"""
    def _deflection_for(thickness):
        graph, _, _ = robust_evaluator.get_graph([0, thickness])
        obj, _ = robust_evaluator.evaluate(graph)
        return dict(zip([o.name for o in robust_evaluator.objectives], obj))['deflection']

    assert _deflection_for(4.) < _deflection_for(2.)


def test_robust_problem_margin_is_conservative():
    """The MARGIN formulation is never better than the MEAN one for a minimization objective"""
    x = [0, 3.]

    mean_eval = RobustBeamEvaluator(stochastic=StochasticMetricType.MEAN, n_mc=2000)
    graph, _, _ = mean_eval.get_graph(x)
    mean_obj, _ = mean_eval.evaluate(graph)
    i_deflection = [o.name for o in mean_eval.objectives].index('deflection')

    margin_eval = RobustBeamEvaluator(stochastic=StochasticMetricType.MARGIN, k=2., n_mc=2000)
    graph, _, _ = margin_eval.get_graph(x)
    margin_obj, _ = margin_eval.evaluate(graph)

    assert margin_obj[i_deflection] > mean_obj[i_deflection]


def test_robust_problem_k_controls_conservatism():
    """A larger k penalizes scatter more heavily"""
    x = [1, 2.]
    i_deflection = None
    values = []
    for k in [0., 1., 4.]:
        evaluator = RobustBeamEvaluator(stochastic=StochasticMetricType.MARGIN, k=k, n_mc=2000)
        graph, _, _ = evaluator.get_graph(x)
        obj, _ = evaluator.evaluate(graph)
        if i_deflection is None:
            i_deflection = [o.name for o in evaluator.objectives].index('deflection')
        values.append(obj[i_deflection])

    assert values[0] < values[1] < values[2]


def test_robust_problem_material_choice_changes_uncertainty():
    """
    Steel and aluminium have the same absolute stiffness scatter, but aluminium's is much larger
    relative to its mean --> aluminium is the less robust choice.
    """
    evaluator = RobustBeamEvaluator(stochastic=StochasticMetricType.MEAN, n_mc=4000)
    i_deflection = [o.name for o in evaluator.objectives].index('deflection')

    def _stats_for(opt_idx):
        graph, _, _ = evaluator.get_graph([opt_idx, 3.])
        obj, _ = evaluator.evaluate(graph)
        return obj[i_deflection]

    steel = _stats_for(0)
    alu = _stats_for(1)

    # Aluminium is more compliant, so it deflects more
    assert alu > steel


def test_robust_problem_over_all_discrete_design_points(robust_evaluator):
    """Run the full pipeline over every discrete design point, as an optimizer would"""
    assert robust_evaluator.get_n_valid_designs() == 2

    x_all, is_active_all = robust_evaluator.get_all_discrete_x()
    assert x_all.shape[0] == 2

    results = []
    for x_discrete in x_all:
        x = [int(x_discrete[0]), 3.]
        graph, x_imputed, is_active = robust_evaluator.get_graph(x)

        obj, con = robust_evaluator.evaluate(graph)
        assert all(np.isfinite(obj))
        assert len(graph.uncertain_parameter_values) == 2
        results.append(obj)

    assert len(results) == 2
    assert robust_evaluator.n_evaluations == 2

    # Every design point got its own Monte Carlo run of the configured size
    assert robust_evaluator.mc_samples_seen == [2, 2]


def test_robust_problem_random_design_points(robust_evaluator):
    """Random design vectors all evaluate to finite robust objective values"""
    for _ in range(10):
        x = [dv.rand() for dv in robust_evaluator.des_vars]
        graph, x_imputed, is_active = robust_evaluator.get_graph(x)

        obj, con = robust_evaluator.evaluate(graph)
        assert len(obj) == 2
        assert all(np.isfinite(obj))
        assert all(value > 0 for value in obj)
