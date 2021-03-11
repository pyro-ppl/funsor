from collections import OrderedDict
from funsor.factory import make_funsor, Fresh, Has
from funsor.terms import Funsor, Number
# from funsor.linearize import linearize
from funsor.testing import assert_close, random_tensor
from funsor.domains import Real, Reals, Bint
from funsor.jvp import Tangent
from makefun import create_function, with_signature, partial

def test_mul():
    @make_funsor
    def Mul(
        x: Funsor,
        y: Funsor
    ) -> Fresh[lambda x: x]:
        return x * y
    x = random_tensor(OrderedDict(j=Bint[4]))
    y = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))
    dx = random_tensor(OrderedDict(j=Bint[4]))
    dy = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))
    #  x_ = Tangent((x, Tx))
    #  y_ = Tangent((y, Ty))
    def linear_fn(x, y, Tx, Ty):
        print("hello fellas!")
        x_ = Tangent((x, Tx))
        y_ = Tangent((y, Ty))
        return Mul(x_, y_)[1]

    breakpoint()
    LinearMul = partial(linear_fn, x=x, y=y)

    #  def linearize(fn, (x, y)):
    #      def linear_fn(Tx, Ty):
    #          x_ = Tangent((x, Tx))
    #          y_ = Tangent((y, Ty))
    #          return Mul(x_, y_)
    #      return

    # Mul(x, y), LinearMul = linearize(Mul, (x, y))
    @make_funsor
    def LinearMul(
        dx: Funsor,
        dy: Funsor
    ) -> Fresh[lambda dx: dx]:
        return x * dy + y * dx
    assert Mul(x, y) == x * y
