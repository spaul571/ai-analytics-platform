"""Sandboxed execution of LLM-generated pandas code (Task B2, Phase 2).

SECURITY RATIONALE
------------------
The model writes Python, and we execute it. That is an arbitrary-code-execution
sink by construction, so the defence cannot be a substring blocklist. Checking
`if "import" in code` is defeated by `__import__("os")`, and checking for `os`
is defeated by `getattr(__builtins__, "ev" + "al")`. Both are one-liners.

The defence here has three independent layers. Each one alone would be
insufficient; an attack must defeat all three.

Layer 1 - AST validation (static, before anything runs).
    The code is parsed into a syntax tree and every node is checked against an
    allowlist of statement and expression types. Import, ImportFrom, With,
    Try, FunctionDef, ClassDef, Global, Nonlocal, Lambda, and comprehension-free
    Await/Yield are all rejected outright. Any attribute access or name whose
    identifier begins with an underscore is rejected, which kills the entire
    dunder-traversal family (`__class__`, `__globals__`, `__subclasses__`,
    `__builtins__`, `__import__`) in one rule. Introspection builtins that can
    manufacture references at runtime (getattr, setattr, eval, exec, compile,
    globals, locals, vars, dir, open, input, __import__) are rejected by name.

Layer 2 - restricted execution namespace.
    The code runs with `__builtins__` replaced by a small explicit dict. Even if
    a reference survived Layer 1, there is no `open`, no `eval`, no `__import__`
    in scope to reach. Only `df`, `pd`, and `np` are exposed.

Layer 3 - timeout.
    Execution runs on a worker thread and is abandoned if it exceeds the budget,
    so a runaway `df.merge(df, how="cross")` cannot hang the UI.

KNOWN LIMITATIONS (stated honestly for the report's Evaluation section):
    - Layer 3 abandons the thread rather than killing it; CPython offers no safe
      way to terminate a running thread. A timed-out computation keeps consuming
      CPU until it finishes. It cannot return a value or block the UI, but it is
      not free.
    - This is a language-level sandbox, not an OS-level one. It is appropriate
      because the untrusted input comes from a local model we ourselves prompt,
      not from the public internet. Exposing this endpoint to untrusted users
      would demand process isolation (a container or a seccomp jail) instead.
"""

from __future__ import annotations

import ast
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Execution budget for a single generated snippet. The whole dataset is ~10k
# rows, so any legitimate query finishes in milliseconds; anything approaching
# this limit is pathological.
EXECUTION_TIMEOUT_SECONDS = 10.0

# The variable the generated code must assign its answer to.
RESULT_VARIABLE = "result"

# Builtins that can manufacture arbitrary references at runtime, escape the
# namespace, or touch the host. Rejected by name during AST validation, and
# absent from the execution namespace regardless.
FORBIDDEN_CALLS = {
    "__import__", "eval", "exec", "compile", "open", "input", "breakpoint",
    "getattr", "setattr", "delattr", "hasattr",
    "globals", "locals", "vars", "dir",
    "exit", "quit", "help", "memoryview", "id",
}

# Builtins the generated code legitimately needs. These are injected into the
# execution namespace in place of the real __builtins__.
SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "divmod": divmod, "enumerate": enumerate, "filter": filter, "float": float,
    "int": int, "isinstance": isinstance, "len": len, "list": list, "map": map,
    "max": max, "min": min, "range": range, "reversed": reversed, "round": round,
    "set": set, "slice": slice, "sorted": sorted, "str": str, "sum": sum,
    "tuple": tuple, "zip": zip,
    "True": True, "False": False, "None": None,
}

# Names the generated code is allowed to reference. This MUST include the safe
# builtins: the validator (Layer 1) and the execution namespace (Layer 2) are
# separate mechanisms, and if the validator's allowlist is narrower than the
# namespace, it rejects code the namespace would happily have run. That is
# exactly what happened on the first benchmark run - `len(df)` was refused even
# though `len` was sitting in SAFE_BUILTINS - so the two lists are now derived
# from one source rather than maintained by hand.
ALLOWED_NAMES = {"df", "pd", "np", RESULT_VARIABLE} | set(SAFE_BUILTINS)

# Statement and expression node types the generated code may contain. Anything
# absent from this set is rejected. Allowlisting rather than blocklisting means
# a Python version that adds a new node type fails closed, not open.
ALLOWED_NODES: set[type[ast.AST]] = {
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign,
    ast.Name, ast.Load, ast.Store, ast.Attribute, ast.Call, ast.Constant,
    ast.keyword, ast.Starred,
    ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Slice, ast.Subscript,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not, ast.Invert,
    ast.And, ast.Or, ast.BitAnd, ast.BitOr, ast.BitXor,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
    ast.Is, ast.IsNot,
    ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp, ast.comprehension,
    ast.JoinedStr, ast.FormattedValue,
}


class SandboxViolation(Exception):
    """Raised when generated code fails static validation. Never executed."""


class SandboxTimeout(Exception):
    """Raised when generated code exceeds the execution budget."""


@dataclass
class ExecutionResult:
    """Outcome of running one generated snippet."""

    value: pd.DataFrame | pd.Series | float | int | str
    execution_ms: float
    code: str

    def as_frame(self) -> pd.DataFrame:
        """Normalise any result shape into a DataFrame for charting/display."""
        if isinstance(self.value, pd.DataFrame):
            return self.value.reset_index(drop=False) if self.value.index.name else self.value
        if isinstance(self.value, pd.Series):
            frame = self.value.reset_index()
            if len(frame.columns) == 2:
                frame.columns = [self.value.index.name or "Category", self.value.name or "Value"]
            return frame
        # A bare list or ndarray means the model ended on .tolist(), .values, or
        # .index and threw the labels away. The values are still worth showing,
        # so render them as a column rather than stuffing the whole list into a
        # single cell. The prompt discourages this shape; this is the safety net.
        if isinstance(self.value, (list, tuple, np.ndarray)):
            return pd.DataFrame({"Value": list(self.value)})
        return pd.DataFrame([{"Value": self.value}])


def validate(code: str) -> ast.Module:
    """Statically validate generated code. Layer 1 of the sandbox.

    Args:
        code: The Python snippet emitted by the LLM.

    Returns:
        The parsed AST, ready to compile.

    Raises:
        SandboxViolation: if the code contains anything outside the allowlist.
    """
    if not code.strip():
        raise SandboxViolation("Generated code was empty.")

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise SandboxViolation(f"Generated code is not valid Python: {exc.msg}") from exc

    for node in ast.walk(tree):
        node_type = type(node)

        if node_type not in ALLOWED_NODES:
            raise SandboxViolation(
                f"Disallowed syntax: {node_type.__name__}. "
                "Only expressions and simple assignments over df/pd/np are permitted "
                "(no imports, loops, functions, or file access)."
            )

        # Kill the dunder-traversal escape family in a single rule. Without this,
        # `().__class__.__base__.__subclasses__()` reaches arbitrary classes and
        # from there the os module, using nothing but attribute access.
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise SandboxViolation(
                f"Disallowed attribute access: {node.attr!r}. "
                "Underscore-prefixed attributes are blocked."
            )

        if isinstance(node, ast.Name):
            if node.id.startswith("_"):
                raise SandboxViolation(f"Disallowed name: {node.id!r}.")
            if node.id in FORBIDDEN_CALLS:
                raise SandboxViolation(
                    f"Disallowed builtin: {node.id!r}. "
                    "Introspection and I/O builtins are blocked."
                )
            # Reading an unknown name is an error; writing one is fine, since
            # intermediate variables are legitimate.
            if isinstance(node.ctx, ast.Load) and node.id not in ALLOWED_NAMES:
                # Allow names the snippet assigned to itself earlier.
                assigned = {
                    t.id
                    for n in ast.walk(tree)
                    if isinstance(n, ast.Assign)
                    for t in n.targets
                    if isinstance(t, ast.Name)
                }
                if node.id not in assigned:
                    raise SandboxViolation(
                        f"Unknown name: {node.id!r}. "
                        f"Only {', '.join(sorted(ALLOWED_NAMES))} are available."
                    )

    # The pipeline reads the answer out of RESULT_VARIABLE, so the code must
    # actually assign it.
    assigns_result = any(
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == RESULT_VARIABLE for t in n.targets)
        for n in ast.walk(tree)
    )
    if not assigns_result:
        raise SandboxViolation(
            f"Generated code must assign its answer to a variable named "
            f"{RESULT_VARIABLE!r}."
        )

    return tree


def execute(code: str, df: pd.DataFrame, timeout: float = EXECUTION_TIMEOUT_SECONDS) -> ExecutionResult:
    """Validate and run generated code against the dataset.

    Args:
        code: Python snippet that must assign to `result`.
        df: The dataset. A shallow copy is passed in so the snippet cannot
            mutate the application's canonical frame.
        timeout: Execution budget in seconds.

    Returns:
        ExecutionResult holding whatever the snippet bound to `result`.

    Raises:
        SandboxViolation: if static validation fails.
        SandboxTimeout: if execution exceeds the budget.
        Exception: whatever the snippet itself raised (a KeyError on a
            hallucinated column, most often). The pipeline feeds this back to
            the model for its single auto-retry.
    """
    tree = validate(code)
    compiled = compile(tree, filename="<llm_generated>", mode="exec")

    # Layer 2: the only things in scope are the dataset and the two libraries.
    namespace: dict[str, object] = {
        "__builtins__": SAFE_BUILTINS,
        "df": df.copy(deep=False),
        "pd": pd,
        "np": np,
    }

    outcome: queue.Queue = queue.Queue(maxsize=1)

    def _run() -> None:
        try:
            exec(compiled, namespace)  # noqa: S102 - the point of this module
            outcome.put(("ok", namespace.get(RESULT_VARIABLE)))
        except Exception as exc:  # noqa: BLE001 - forwarded to the retry loop
            outcome.put(("error", exc))

    # Layer 3: bound the runtime.
    start = time.perf_counter()
    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout)

    if worker.is_alive():
        raise SandboxTimeout(
            f"Generated code exceeded the {timeout:.0f}s execution budget."
        )

    status, payload = outcome.get_nowait()
    elapsed_ms = (time.perf_counter() - start) * 1000

    if status == "error":
        raise payload  # type: ignore[misc]

    if payload is None:
        raise SandboxViolation(f"{RESULT_VARIABLE!r} was never assigned a value.")

    return ExecutionResult(value=payload, execution_ms=elapsed_ms, code=code)
