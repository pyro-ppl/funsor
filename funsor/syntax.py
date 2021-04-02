# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import ast
import functools
import inspect

from . import ops

PREFIX_OPERATORS = (
    ("~", ops.invert, ast.Invert),
    ("+", ops.pos, ast.UAdd),
    ("-", ops.neg, ast.USub),
)
INFIX_OPERATORS = (
    ("+", ops.add, ast.Add),
    ("-", ops.sub, ast.Sub),
    ("*", ops.mul, ast.Mult),
    ("/", ops.truediv, ast.Div),
    ("//", ops.floordiv, ast.FloorDiv),
    ("%", ops.mod, ast.Mod),
    ("**", ops.pow, ast.Pow),
    ("<<", ops.lshift, ast.LShift),
    (">>", ops.rshift, ast.RShift),
    ("|", ops.or_, ast.BitOr),
    ("^", ops.xor, ast.BitXor),
    ("&", ops.and_, ast.BitAnd),
    ("@", ops.matmul, ast.MatMult),
    ("==", ops.eq, ast.Eq),
    ("!=", ops.ne, ast.NotEq),
    ("<", ops.lt, ast.Lt),
    ("<=", ops.le, ast.LtE),
    (">", ops.gt, ast.Gt),
    (">=", ops.ge, ast.GtE),
)

PREFIX_TO_NODE = {k: v for k, _, v in PREFIX_OPERATORS}
INFIX_TO_NODE = {k: v for k, _, v in INFIX_OPERATORS}


class OpTransformer(ast.NodeTransformer):
    def __init__(self, infix, prefix, const):
        assert isinstance(infix, dict)
        assert isinstance(prefix, dict)
        assert isinstance(const, dict)
        self.infix = {INFIX_TO_NODE[k]: v for k, v in infix.items()}
        self.prefix = {PREFIX_TO_NODE[k]: v for k, v in prefix.items()}
        self.const = const

    def visit_Constant(self, node):
        node = self.generic_visit(node)
        var = self.const.get(node.value)
        if var is not None:
            node = ast.Name(id=var, ctx=ast.Load())
        return node

    def visit_UnaryOp(self, node):
        node = self.generic_visit(node)
        var = self.prefix.get(type(node.op))
        if var is not None:
            node = ast.Call(
                func=ast.Name(id=var, ctx=ast.Load()), args=[node.operand], keywords=[]
            )
        return node

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        var = self.infix.get(type(node.op))
        if var is not None:
            node = ast.Call(
                func=ast.Name(id=var, ctx=ast.Load()),
                args=[node.left, node.right],
                keywords=[],
            )
        return node

    def visit_Compare(self, node):
        node = self.generic_visit(node)

        # Restrict to the binary case.
        assert len(node.ops) == len(node.comparitors)
        if len(node.ops) > 1:
            raise NotImplementedError(
                "Please decompose stacked comparisons into conjunctions of "
                "binary comparisons, e.g. 'x < y < z' as '(x < y) & (y < z)'"
            )
        node_op = node.ops[0]
        node_right = node.comparitors[0]

        var = self.infix.get(type(node_op))
        if var is not None:
            node = ast.Call(
                func=ast.Name(id=var, ctx=ast.Load()),
                args=[node.left, node_right],
                keywords=[],
            )
        return node


def rewrite_ops(infix={}, prefix={}, const={}):
    """
    Decorator to replace infix binary operators, prefix unary operators, and
    constants (nullary operators) in the decorated function's code with named
    variables.

    For example the following code::

        @rewrite_ops({"+": "sum_op", "*": "prod_op"})
        def product_rule(sum_op, prod_op, lhs, rhs, d):
            return d(lhs) * rhs + lhs * d(rhs)

    will be rewritten as::

        def product_rule(sum_op, prod_op, lhs, rhs, d):
            return sum_op(prod_op(d(lhs), rhs), prod_op(lhs, d(rhs)))

    .. warning:: This must be used as the innermost decorator.

    .. warning:: This requires Python 3.9+ and should not yet be used in
        Funsor library code.

    :param dict infix: An optional mapping from infix operator symbol to
        variable name.
    :param dict prefix: An optional mapping from prefix operator symbol to
        variable name.
    :param dict const: An optional mapping from constant literal to variable
        name.
    :returns: A decorator
    :rtype: callable
    """
    transformer = OpTransformer(infix, prefix, const)

    def decorator(fn):
        source = inspect.getsource(fn)

        # Strip indentation and all decorators.
        indent = len(source) - len(source.lstrip())
        lines = []
        discard = True
        for line in source.split("\n"):
            line = line[indent:]
            if discard:
                if line.startswith("def "):
                    discard = False
                else:
                    continue
            lines.append(line)
        source = "\n".join(lines)
        assert source

        # Transform the function.
        a = ast.parse(source)
        a_t = transformer.visit(a)
        source_t = ast.unparse(a_t)
        result = {}
        exec(source_t, globals(), result)
        fn_t = result[fn.__name__]
        functools.update_wrapper(fn_t, fn)
        return fn_t

    return decorator


__all__ = [
    "INFIX_OPERATORS",
    "PREFIX_OPERATORS",
    "rewrite_ops",
]
