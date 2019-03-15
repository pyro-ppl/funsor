from __future__ import absolute_import, division, print_function

# Design notes:
# Component 1: Recognition interpreter generator
# Component 2: Rewrites of recognized patterns
# Component 3: Constant folding/CSE? Just use eager?

import funsor.ops as ops
from funsor.interpreter import interpretation, reinterpret
from funsor.registry import KeyedRegistry
from funsor.terms import Binary, Funsor, reflect


class Pattern(Funsor):
    """
    Recognized pattern term
    """
    pass


class Free(Pattern):
    pass


class Choice(Pattern):
    pass


def make_recognizer(pattern):
    # make a recognizer for a Funsor pattern, rewriting the first match

    # semiring matcher: sum = any_match, product = all_match
    # proof: a*(b+c) = a*b + a*c -> p0(Choice(p1, p2)) = Choice(p0(p1), p0(p2))
    # match query as contraction: match(pattern, expr)
    # dfs strategy: keep trying to substitute subexpressions
    # match(p0(free), pattern)(Choice(match(p1(free)), match(p2(free))))

    assert isinstance(pattern, Funsor)

    # generating a recognizer from a funsor:
    # 1) generate recognizers for the sub-expressions
    # 2) generate recognizer for the node by unwrapping and matching

    def recognizer(term, *args):
        new_args = tuple(arg.v if isinstance(arg, Pattern) else arg for arg in args)
        if all(isinstance(x, Pattern) or not isinstance(x, Funsor) for x in args):
            return _recognizer(term, *new_args)  # propagate Pattern type
        else:
            return reflect(term, *new_args)  # stop Pattern propagation here

    _recognizer = KeyedRegistry(default=lambda *args: None)
    recognizer.register = _recognizer.register

    # register special matchers for parser combinator primitives (Free, Choice, etc)
    @recognizer.register(Choice, (list, tuple))
    def recognize_choice(choices):
        if any(isinstance(choice, Pattern) for choice in choices):
            pass
        else:
            pass

    @recognizer.register(Free)
    def recognize_free():
        pass

    # now recursively register matchers for subexpressions
    def make_and_reflect(term, *args):
        if not issubclass(term, Pattern):
            recognizer.register(term, *map(type, args))(
                lambda _term, *_args: Pattern(reflect(_term, *_args))
            )
        return reflect(term, *args)

    with interpretation(make_and_reflect):
        reinterpret(pattern)

    # dynamically generate recognized class
    # this is the thing we write nonstandard interpretations for to do rewriting
    # we need one reflect interpretation per expansion of Choices in pattern
    class recognized(Funsor):
        pass

    return recognizer, recognized


# examples...
double_add = Binary(ops.add, Binary(ops.add, Free(), Free()), Free())
# a + b + c + d -> double_add(a, b, c + d)


affine = Choice((
    Binary(ops.add, Free(), Free()),
    Binary(ops.mul, Free(), Free()),
))

affine_recognizer, Affine = make_recognizer(affine)
