# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import ast

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

__all__ = [
    "INFIX_OPERATORS",
    "PREFIX_OPERATORS",
]
