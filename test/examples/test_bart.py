import math

import torch

import funsor
import funsor.distributions as dist
import funsor.ops as ops
from funsor.cnf import Contraction
from funsor.domains import bint, reals
from funsor.gaussian import Gaussian
from funsor.integrate import Integrate
from funsor.interpreter import interpretation
from funsor.montecarlo import monte_carlo
from funsor.pyro.convert import AffineNormal
from funsor.terms import Binary, Independent, Number, Reduce, Slice, Stack, Subs, Variable, reflect
from funsor.torch import Function, Tensor

num_origins = 2
num_destins = 2


def bounded_exp(x, bound):
    return (x - math.log(bound)).sigmoid() * bound


def unpack_gate_rate(gate_rate):
    batch_shape = gate_rate.shape[:-1]
    event_shape = (2, num_origins, num_destins)
    gate, rate = gate_rate.reshape(batch_shape + event_shape).unbind(-3)
    rate = bounded_exp(rate, bound=1e4)
    gate = torch.stack((torch.zeros_like(gate), gate), dim=-1)
    return gate, rate


def unpack_gate_rate_0(gate_rate):
    return unpack_gate_rate(gate_rate)[0]


def unpack_gate_rate_1(gate_rate):
    return unpack_gate_rate(gate_rate)[1]


def test_bart():
    with interpretation(reflect):
        q = Independent(
         Independent(
          Contraction(
           ops.nullop,
           ops.add,
           frozenset(),
           (Tensor(
             torch.tensor([[-0.6077086925506592, -1.1546266078948975, -0.7021151781082153, -0.5303535461425781, -0.6365622282028198, -1.2423288822174072, -0.9941254258155823, -0.6287292242050171], [-0.6987162828445435, -1.0875964164733887, -0.7337473630905151, -0.4713417589664459, -0.6674002408981323, -1.2478348016738892, -0.8939017057418823, -0.5238542556762695]], dtype=torch.float32),  # noqa
             (('time_b3',
               bint(2),),
              ('_event_1_b2',
               bint(8),),),
             'real'),
            Gaussian(
             torch.tensor([[[-0.3536059558391571], [-0.21779225766658783], [0.2840439975261688], [0.4531521499156952], [-0.1220812276005745], [-0.05519985035061836], [0.10932210087776184], [0.6656699776649475]], [[-0.39107921719551086], [-0.20241987705230713], [0.2170514464378357], [0.4500560462474823], [0.27945515513420105], [-0.0490039587020874], [-0.06399798393249512], [0.846565842628479]]], dtype=torch.float32),  # noqa
             torch.tensor([[[[1.984686255455017]], [[0.6699360013008118]], [[1.6215802431106567]], [[2.372016668319702]], [[1.77385413646698]], [[0.526767373085022]], [[0.8722561597824097]], [[2.1879124641418457]]], [[[1.6996612548828125]], [[0.7535632252693176]], [[1.4946647882461548]], [[2.642792224884033]], [[1.7301604747772217]], [[0.5203893780708313]], [[1.055436372756958]], [[2.8370864391326904]]]], dtype=torch.float32),  # noqa
             (('time_b3',
               bint(2),),
              ('_event_1_b2',
               bint(8),),
              ('value_b1',
               reals(),),)),)),
          'gate_rate_b4',
          '_event_1_b2',
          'value_b1'),
         'gate_rate_t',
         'time_b3',
         'gate_rate_b4')
        p_prior = Reduce(ops.logaddexp,
         Binary(ops.add,
          Subs(
           Reduce(ops.logaddexp,
            Binary(ops.add,
             Subs(
              Binary(ops.add,
               Binary(ops.add,
                Tensor(
                 torch.tensor(2.7672932147979736, dtype=torch.float32),
                 (),
                 'real'),
                Subs(
                 Subs(
                  Gaussian(
                   torch.tensor([-0.0, -0.0, 0.0, 0.0], dtype=torch.float32),
                   torch.tensor([[98.01002502441406, 0.0, -99.0000228881836, -0.0], [0.0, 98.01002502441406, -0.0, -99.0000228881836], [-99.0000228881836, -0.0, 100.0000228881836, 0.0], [-0.0, -99.0000228881836, 0.0, 100.0000228881836]], dtype=torch.float32),  # noqa
                   (('state_b13',
                     reals(2,),),
                    ('state(time=1)_b7',
                     reals(2,),),)),
                  ()),
                 ())),
               Subs(
                AffineNormal(
                 Tensor(
                  torch.tensor([[0.03488487750291824, 0.07356668263673782, 0.19946961104869843, 0.5386509299278259, -0.708323061466217, 0.24411526322364807, -0.20855577290058136, -0.2421337217092514], [0.41762110590934753, 0.5272183418273926, -0.49835553765296936, -0.0363837406039238, -0.0005282597267068923, 0.2704298794269562, -0.155222088098526, -0.44802337884902954]], dtype=torch.float32),  # noqa
                  (),
                  'real'),
                 Tensor(
                  torch.tensor([[-0.003566693514585495, -0.2848514914512634, 0.037103548645973206, 0.12648648023605347, -0.18501518666744232, -0.20899859070777893, 0.04121830314397812, 0.0054807960987091064], [0.0021788496524095535, -0.18700894713401794, 0.08187370002269745, 0.13554862141609192, -0.10477752983570099, -0.20848378539085388, -0.01393645629286766, 0.011670656502246857]], dtype=torch.float32),  # noqa
                  (('time_b8',
                    bint(2),),),
                  'real'),
                 Tensor(
                  torch.tensor([[0.5974780917167664, 0.864071786403656, 1.0236268043518066, 0.7147538065910339, 0.7423890233039856, 0.9462157487869263, 1.2132389545440674, 1.0596832036972046], [0.5787821412086487, 0.9178534150123596, 0.9074794054031372, 0.6600189208984375, 0.8473222255706787, 0.8426999449729919, 1.194266438484192, 1.0471148490905762]], dtype=torch.float32),  # noqa
                  (('time_b8',
                    bint(2),),),
                  'real'),
                 Variable('state(time=1)_b7', reals(2,)),
                 Variable('gate_rate_b6', reals(8,))),
                (('gate_rate_b6',
                  Binary(ops.GetitemOp(0),
                   Variable('gate_rate_t', reals(2, 8)),
                   Variable('time_b8', bint(2))),),))),
              (('state(time=1)_b7',
                Variable('_drop_0_b11', reals(2,)),),
               ('time_b8',
                Slice('time_b12', 0, 2, 2, 2),),)),
             Subs(
              Binary(ops.add,
               Binary(ops.add,
                Tensor(
                 torch.tensor(2.7672932147979736, dtype=torch.float32),
                 (),
                 'real'),
                Subs(
                 Subs(
                  Gaussian(
                   torch.tensor([-0.0, -0.0, 0.0, 0.0], dtype=torch.float32),
                   torch.tensor([[98.01002502441406, 0.0, -99.0000228881836, -0.0], [0.0, 98.01002502441406, -0.0, -99.0000228881836], [-99.0000228881836, -0.0, 100.0000228881836, 0.0], [-0.0, -99.0000228881836, 0.0, 100.0000228881836]], dtype=torch.float32),  # noqa
                   (('state_b9',
                     reals(2,),),
                    ('state(time=1)_b14',
                     reals(2,),),)),
                  ()),
                 ())),
               Subs(
                AffineNormal(
                 Tensor(
                  torch.tensor([[0.03488487750291824, 0.07356668263673782, 0.19946961104869843, 0.5386509299278259, -0.708323061466217, 0.24411526322364807, -0.20855577290058136, -0.2421337217092514], [0.41762110590934753, 0.5272183418273926, -0.49835553765296936, -0.0363837406039238, -0.0005282597267068923, 0.2704298794269562, -0.155222088098526, -0.44802337884902954]], dtype=torch.float32),  # noqa
                  (),
                  'real'),
                 Tensor(
                  torch.tensor([[-0.003566693514585495, -0.2848514914512634, 0.037103548645973206, 0.12648648023605347, -0.18501518666744232, -0.20899859070777893, 0.04121830314397812, 0.0054807960987091064], [0.0021788496524095535, -0.18700894713401794, 0.08187370002269745, 0.13554862141609192, -0.10477752983570099, -0.20848378539085388, -0.01393645629286766, 0.011670656502246857]], dtype=torch.float32),  # noqa
                  (('time_b10',
                    bint(2),),),
                  'real'),
                 Tensor(
                  torch.tensor([[0.5974780917167664, 0.864071786403656, 1.0236268043518066, 0.7147538065910339, 0.7423890233039856, 0.9462157487869263, 1.2132389545440674, 1.0596832036972046], [0.5787821412086487, 0.9178534150123596, 0.9074794054031372, 0.6600189208984375, 0.8473222255706787, 0.8426999449729919, 1.194266438484192, 1.0471148490905762]], dtype=torch.float32),  # noqa
                  (('time_b10',
                    bint(2),),),
                  'real'),
                 Variable('state(time=1)_b14', reals(2,)),
                 Variable('gate_rate_b6', reals(8,))),
                (('gate_rate_b6',
                  Binary(ops.GetitemOp(0),
                   Variable('gate_rate_t', reals(2, 8)),
                   Variable('time_b10', bint(2))),),))),
              (('state_b9',
                Variable('_drop_0_b11', reals(2,)),),
               ('time_b10',
                Slice('time_b12', 1, 2, 2, 2),),))),
            frozenset({'_drop_0_b11'})),
           (('time_b12',
             Number(0, 1),),)),
          Subs(
           dist.MultivariateNormal(
            Tensor(
             torch.tensor([0.0, 0.0], dtype=torch.float32),
             (),
             'real'),
            Tensor(
             torch.tensor([[10.0, 0.0], [0.0, 10.0]], dtype=torch.float32),
             (),
             'real'),
            Variable('value_b5', reals(2,))),
           (('value_b5',
             Variable('state_b13', reals(2,)),),))),
         frozenset({'state(time=1)_b14', 'state_b13'}))
        p_likelihood = Reduce(ops.add,
         Reduce(ops.logaddexp,
          Binary(ops.add,
           dist.Categorical(
            Binary(ops.GetitemOp(0),
             Binary(ops.GetitemOp(0),
              Subs(
               Function(unpack_gate_rate_0,
                reals(2, 2, 2),
                (Variable('gate_rate_b15', reals(8,)),)),
               (('gate_rate_b15',
                 Binary(ops.GetitemOp(0),
                  Variable('gate_rate_t', reals(2, 8)),
                  Variable('time_b18', bint(2))),),)),
              Variable('origin_b20', bint(2))),
             Variable('destin_b19', bint(2))),
            Variable('gated_b17', bint(2))),
           Stack(
            'gated_b17',
            (dist.Poisson(
              Binary(ops.GetitemOp(0),
               Binary(ops.GetitemOp(0),
                Subs(
                 Function(unpack_gate_rate_1,
                  reals(2, 2),
                  (Variable('gate_rate_b16', reals(8,)),)),
                 (('gate_rate_b16',
                   Binary(ops.GetitemOp(0),
                    Variable('gate_rate_t', reals(2, 8)),
                    Variable('time_b18', bint(2))),),)),
                Variable('origin_b20', bint(2))),
               Variable('destin_b19', bint(2))),
              Tensor(
               torch.tensor([[[1.0, 1.0], [5.0, 0.0]], [[0.0, 6.0], [19.0, 3.0]]], dtype=torch.float32),  # noqa
               (('time_b18',
                 bint(2),),
                ('origin_b20',
                 bint(2),),
                ('destin_b19',
                 bint(2),),),
               'real')),
             dist.Delta(
              Tensor(
               torch.tensor(0.0, dtype=torch.float32),
               (),
               'real'),
              Tensor(
               torch.tensor(0.0, dtype=torch.float32),
               (),
               'real'),
              Tensor(
               torch.tensor([[[1.0, 1.0], [5.0, 0.0]], [[0.0, 6.0], [19.0, 3.0]]], dtype=torch.float32),  # noqa
               (('time_b18',
                 bint(2),),
                ('origin_b20',
                 bint(2),),
                ('destin_b19',
                 bint(2),),),
               'real')),))),
          frozenset({'gated_b17'})),
         frozenset({'time_b18', 'origin_b20', 'destin_b19'}))

        p = p_prior + p_likelihood
        pq = p - q

    with interpretation(monte_carlo):
        Integrate(q, pq, frozenset(['gate_rate_t']))
