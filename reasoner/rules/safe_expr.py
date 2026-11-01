"""Compile rule conditions from YAML into fast Python callables — safely.

Only a whitelisted expression subset is accepted: numbers, strings, known feature names,
comparisons (incl. `in`), and/or/not, + - * /, unary minus, lists/tuples and calls to abs/min/max.
Anything else (attribute access, imports, other calls, subscripts…) is rejected before compiling.
"""
from __future__ import annotations

import ast

_ALLOWED = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.UAdd,
            ast.Compare, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.In, ast.NotIn,
            ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Constant, ast.Name, ast.Load,
            ast.List, ast.Tuple, ast.Call)
_FUNCS = {"abs": abs, "min": min, "max": max}


class RuleSyntaxError(ValueError):
    pass


def compile_condition(expr: str, variables: set[str]):
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise RuleSyntaxError(f"cannot parse {expr!r}: {e.msg}") from None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise RuleSyntaxError(f"{type(node).__name__} not allowed in {expr!r}")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in _FUNCS
                                               and not node.keywords):
            raise RuleSyntaxError(f"only abs/min/max calls are allowed in {expr!r}")
        if isinstance(node, ast.Name) and node.id not in variables and node.id not in _FUNCS:
            raise RuleSyntaxError(f"unknown name {node.id!r} in {expr!r}; known: {sorted(variables)}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, str, bool)):
            raise RuleSyntaxError(f"constant {node.value!r} not allowed in {expr!r}")
    code = compile(tree, "<rule>", "eval")
    env = {"__builtins__": {}, **_FUNCS}
    return lambda ctx: bool(eval(code, env, ctx))  # noqa: S307 — AST validated above
