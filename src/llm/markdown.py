"""Make LLM prose safe to hand to Streamlit's Markdown renderer.

Streamlit renders `$...$` as LaTeX maths. A narrative about money is therefore a
minefield, because the model writes two dollar amounts in one sentence and the
renderer reads everything between them as an equation:

    input   Tables show the largest loss at $17,725.48, indicating this area
            requires review. The cumulative loss totals $20,387.14.
    output  Tables show the largest loss at -\\17,725.48
            *indicatingthisarearequiresreview.Thecumulativelosstotals* -$20,387.14$

The sentence is silently turned into italic serif with its spaces stripped. No
error is raised and no number is altered - the answer is simply unreadable, which
in a live demo is worse than a visible failure.

The prompts ask the model not to do this and it does it anyway: money written as
`$1,234` is overwhelmingly common in its training data, and one instruction does
not outweigh that. Prompting reduces the frequency; only escaping removes it.

So every dollar sign that reaches the renderer is escaped. The maths syntax is not
something this application ever needs - there are no equations in a sales
narrative - so disabling it wholesale costs nothing and closes the failure class
rather than patching instances of it.

Code spans are left alone: inside backticks a backslash is literal, so escaping
there would put a visible `\\$` on screen.
"""

from __future__ import annotations

import re

# The model also emits $200$ - a dollar amount wrapped as if it were an equation.
# Stripping the closing delimiter first means the escape below leaves "$200"
# rather than a stray trailing "$".
_MATH_WRAPPED_NUMBER = re.compile(r"\$(-?[\d,]+(?:\.\d+)?)\$")

# The model also writes a negative amount as "$-$17,725.48", putting the minus in
# its own maths span. Rendered as maths that looked like a stray italic dash;
# escaped naively it reads "$-$17,725.48". Either way the sign belongs in front.
_MATH_NEGATIVE_SIGN = re.compile(r"\$-\$(?=[\d])")

# Backtick spans, including multi-backtick fences, are skipped.
_CODE_SPAN = re.compile(r"(`+)(.+?)\1", re.S)

# A dollar sign the author has already escaped must not be escaped twice.
_UNESCAPED_DOLLAR = re.compile(r"(?<!\\)\$")


def _unwrap_math_numbers(text: str) -> str:
    """Turn `$200$` into `$200`, and `$-74,142$` into `-$74,142`."""

    def replace(match: re.Match) -> str:
        number = match.group(1)
        if number.startswith("-"):
            return f"-${number[1:]}"
        return f"${number}"

    return _MATH_WRAPPED_NUMBER.sub(replace, _MATH_NEGATIVE_SIGN.sub("-$", text))


def _escape_outside_code(text: str) -> str:
    parts: list[str] = []
    cursor = 0
    for span in _CODE_SPAN.finditer(text):
        parts.append(_UNESCAPED_DOLLAR.sub(r"\\$", text[cursor : span.start()]))
        parts.append(span.group(0))  # verbatim: escaping inside backticks would show
        cursor = span.end()
    parts.append(_UNESCAPED_DOLLAR.sub(r"\\$", text[cursor:]))
    return "".join(parts)


def render_safe(text: str) -> str:
    """Prepare model prose for `st.markdown`, so money reads as money.

    Applied to everything the model writes that reaches the renderer: the Task B
    narrative, the preset insights, the chart caption and the agent's answers and
    reasoning steps.
    """
    if not text:
        return text
    return _escape_outside_code(_unwrap_math_numbers(text))


def render_plain(text: str) -> str:
    """Undo `render_safe`'s escaping for renderers that have no maths syntax.

    reportlab and python-docx never interpret `$`, so the escape that rescues the
    dashboard would put a visible backslash in the exported PDF and Word reports.
    The narrative arrives here already escaped - the export path is downstream of
    the dashboard, not parallel to it - so it is unescaped on the way out.
    """
    if not text:
        return text
    return text.replace("\\$", "$")
