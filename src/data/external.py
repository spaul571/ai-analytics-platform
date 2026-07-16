"""External data the dataset does not contain (Task D4).

The Superstore file is four years of US order lines and nothing else. It cannot
say whether 24 November 2016 was a Thursday like any other or Thanksgiving, and
"do our sales move around public holidays?" is a question about retail that the
data alone cannot answer. This module fetches that missing calendar from
Nager.Date, a free, keyless public-holiday API.

WHY THIS IS THE ONE PLACE THE PROJECT TOUCHES THE NETWORK
---------------------------------------------------------
Everything else here is local by construction. Giving an LLM agent network reach
is where agents usually acquire their worst failure mode: a tool returns
attacker-controlled prose, that prose enters the model's context, and the model
follows it. The guards against that are structural, not hopeful:

    Fixed host      The URL is built from constants. The model supplies a year
                    and nothing else; it cannot name a host, a path or a scheme,
                    so it cannot point this at an arbitrary server.
    Validated year  Rejected unless it is an integer inside the dataset's own
                    span. A year outside the data could not be joined to anything
                    anyway.
    Structured only Two fields are kept per holiday - an ISO date and a name -
                    and the name is truncated. Nothing else from the response
                    reaches the model.
    Bounded         An 8-second timeout, and a hard cap on holidays returned.
    Cached          Results are memoised per (year, country): the same year is
                    fetched once per process no matter how often it is asked for.

The response is a list of dates. There is no free-text field for an attacker to
write instructions into, which is the actual reason this API was chosen over a
web search: not that search is unimplementable, but that a search result is prose
and prose is an injection surface.

Network failure is expected, not exceptional - Streamlit Cloud can be offline,
the API can be down. It raises `ExternalDataError`, the agent turns that into an
observation, and the model carries on with the local tools instead of the run
dying.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache

import requests

# The dataset's own span. A holiday outside it has nothing to join to.
MIN_YEAR = 2014
MAX_YEAR = 2017

# Fixed. The model never supplies any part of this.
_API_HOST = "https://date.nager.at"
_API_PATH = "/api/v3/PublicHolidays/{year}/{country}"

_TIMEOUT_SECONDS = 8
_MAX_HOLIDAYS = 40
_MAX_NAME_CHARS = 60


class ExternalDataError(RuntimeError):
    """The external lookup could not be completed."""


@dataclass(frozen=True)
class Holiday:
    """One public holiday, reduced to the two fields we will actually use."""

    day: date
    name: str


def _clean_name(value: object) -> str:
    """Reduce a holiday name to one short line of printable text.

    The name is the only free text that crosses into the model's context, so it
    is stripped of anything structural before it gets there. Newlines are the
    part that matters: the observation is a formatted block of lines, and a name
    carrying its own newlines could forge extra lines inside it - a fake tool
    result, in a position the model reads as the executor's own words. Collapsing
    whitespace makes the name incapable of that.

    What survives is up to 60 printable characters from a fixed, known host. That
    is a deliberate residual: it is enough to render "Thanksgiving Day" and too
    little to carry a paragraph, but it is not zero, and a compromised upstream
    could still spend those characters on a short instruction. The reason that is
    acceptable here and would not be for a search tool is quantity and source -
    60 characters from date.nager.at rather than a page of text from wherever a
    query happened to land.
    """
    text = str(value or "Holiday")
    text = "".join(char if char.isprintable() else " " for char in text)
    text = " ".join(text.split())  # collapse every run of whitespace to one space
    return text[:_MAX_NAME_CHARS] or "Holiday"


@lru_cache(maxsize=16)
def fetch_holidays(year: int, country: str = "US") -> tuple[Holiday, ...]:
    """Public holidays for one year. Cached; raises ExternalDataError on failure.

    Returns a tuple rather than a list because lru_cache hands the same object to
    every caller and a mutable one could be edited under them.
    """
    try:
        year = int(year)
    except (TypeError, ValueError) as exc:
        raise ExternalDataError(f"{year!r} is not a year.") from exc

    if not MIN_YEAR <= year <= MAX_YEAR:
        raise ExternalDataError(
            f"The dataset covers {MIN_YEAR}-{MAX_YEAR}; {year} has no orders to compare."
        )
    if not (isinstance(country, str) and len(country) == 2 and country.isalpha()):
        raise ExternalDataError(f"{country!r} is not a two-letter country code.")

    url = _API_HOST + _API_PATH.format(year=year, country=country.upper())
    try:
        response = requests.get(url, timeout=_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout as exc:
        raise ExternalDataError(
            f"The holiday API did not respond within {_TIMEOUT_SECONDS}s."
        ) from exc
    except requests.RequestException as exc:
        raise ExternalDataError(f"The holiday API could not be reached: {exc}") from exc
    except ValueError as exc:
        raise ExternalDataError("The holiday API returned something that is not JSON.") from exc

    if not isinstance(payload, list):
        raise ExternalDataError("The holiday API returned an unexpected shape.")

    holidays: list[Holiday] = []
    seen: set[date] = set()
    for entry in payload[:_MAX_HOLIDAYS]:
        if not isinstance(entry, dict):
            continue
        try:
            day = date.fromisoformat(str(entry.get("date", "")))
        except ValueError:
            continue  # a date we cannot parse is one we cannot join on
        # The API lists a holiday once per observing county, so a single day can
        # arrive several times over. Duplicates would double in the model's
        # context and add nothing: the join only cares about the date.
        if day in seen:
            continue
        seen.add(day)
        name = _clean_name(entry.get("localName") or entry.get("name"))
        holidays.append(Holiday(day=day, name=name))

    if not holidays:
        raise ExternalDataError(f"The holiday API returned no usable holidays for {year}.")
    return tuple(holidays)
