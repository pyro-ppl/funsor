# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import ast
import functools
import inspect

NODE_TO_STR = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.Pow: "**",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.BitAnd: "&",
    ast.MatMult: "@",
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}
STR_TO_NODE = {v: k for k, v in NODE_TO_STR.items()}


class BinOpTransformer(ast.NodeTransformer):
    def __init__(self, ops_to_fns):
        assert isinstance(ops_to_fns, dict)
        for k, v in ops_to_fns.items():
            assert isinstance(k, str), k
            assert isinstance(v, str), v
        self.types_to_vars = {STR_TO_NODE[k]: v for k, v in ops_to_fns.items()}

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        var = self.types_to_vars.get(type(node.op))
        if var is None:
            return node
        func = ast.Name(
            id=var,
            ctx=ast.Load(),
        )
        node = ast.Call(
            func=func,
            args=[node.left, node.right],
            keywords=[],
        )
        return node


def rewrite_ops(ops_to_fns):
    """
    Decorator to replace infix binary operators in the decorated function's
    code with named binary variables, either globals or function arguments.

    For example the following code::

        @rewrite_ops({"+": "sum_op", "*": "prod_op"})
        def product_rule(sum_op, prod_op, lhs, rhs, d):
            return d(lhs) * rhs + lhs * d(rhs)

    will be rewritten as::

        def product_rule(sum_op, prod_op, lhs, rhs, d):
            return sum_op(prod_op(d(lhs), rhs), prod_op(lhs, d(rhs)))

    .. warning:: This must be used as the innermost decorator, and must fit on
        a single line.

    :param dict ops_to_fns: A mapping from operator symbol to variable name.
    :returns: A decorator
    :rtype: callable
    """
    transformer = BinOpTransformer(ops_to_fns)

    def decorator(fn):
        source = inspect.getsource(fn)

        # Strip indentation, this decorator, and all above decorators.
        indent = len(source) - len(source.lstrip())
        lines = []
        discard = True
        for line in source.split("\n"):
            line = line[indent:]
            if discard:
                if line.startswith("@rewrite_ops"):
                    discard = False
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
    "rewrite_ops",
]
