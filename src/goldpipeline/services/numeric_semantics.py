"""What a number in an article *means*, and why it was accepted.

The deterministic numeric scanner asks one question of every literal in a draft:
is this a market price the context supports? Answering it needs a second
question the old implementation never asked - *is this literal a price at all?*

A production Run made the gap visible. The article said the window moved
``66.140`` points; the candles were in the 4300s; the scanner compared the two
and reported a fabricated price far outside the market range. The number was
correct, reproducible, and not a price. Round 1 added ``context.levels.atr`` and
zone widths, which are the same kind of number, so the confusion was about to
get louder rather than quieter.

**The fix is classification, not tolerance.** Nothing here says "if a number is
far from the market, let it through" - that would delete the check. Instead a
number may be accepted only when something explicitly says what it is:

1. a **verified source claim** - the writer named a context path, the path
   resolves, and the resolved value matches what the claim says it is. The
   path's declared type (:class:`~goldpipeline.schemas.common.NumericRole`) then
   says whether the number is a price or a distance;
2. an **explicit derived fact** - one of a short, closed list of formulas
   computed from the context in Python.

Everything else stays exactly as strict as before. In particular there is no
arithmetic search: this module never tries combinations of context values
looking for one that matches, because a scanner that will accept any number
expressible as *a - b* over eighty candle values accepts nearly everything.

**Type beats value, always.** 14.25 is a plausible ATR and a plausible price.
Only the declaration distinguishes them, so only the declaration is consulted -
and a magnitude never satisfies an absolute-price statement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from enum import StrEnum
from typing import Any, get_args

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from goldpipeline.schemas.common import NumericRole
from goldpipeline.schemas.context import AnalysisContext
from goldpipeline.services.claim_resolver import ClaimPathError, resolve_path

logger = logging.getLogger(__name__)

MAX_ROLE_DEPTH = 6
"""How far to unwrap an annotation looking for a role marker."""


class SemanticType(StrEnum):
    """What kind of number a literal is, once something has vouched for it."""

    ABSOLUTE_PRICE = "ABSOLUTE_PRICE"
    """A level on the instrument's own scale. The only kind the range check judges."""

    MAGNITUDE = "MAGNITUDE"
    """A distance in price units: ATR, a zone width, a net change."""

    PERCENTAGE = "PERCENTAGE"
    """A proportion. Never comparable to a candle range."""

    NON_MARKET_NUMBER = "NON_MARKET_NUMBER"
    """A count, an index, a score - nothing to do with the price scale."""

    UNKNOWN_PRICE_LIKE = "UNKNOWN_PRICE_LIKE"
    """Nothing vouched for it. Treated as a possible fabricated price."""

    # ---- refinements ------------------------------------------------------
    # Finer meanings a later check needs, each a member of one of the coarse
    # families above. The scanner in production only ever constructs the five
    # coarse members, and `family` maps every refinement back to one of them,
    # so nothing that compares by family changes behaviour. Refinements exist
    # because a digest must keep "net change +105" apart from "range 139":
    # both are magnitudes, and one word for both is how they get swapped.
    #
    # None of these says where a number came from. ABSOLUTE_PRICE is
    # ABSOLUTE_PRICE whether MT5, TradingView or any later provider produced
    # the authoritative value; provenance is a separate field on the fact.

    PRICE_NET_CHANGE = "PRICE_NET_CHANGE"
    """End minus start over a window. Signed. A magnitude, not a price."""

    PRICE_RANGE = "PRICE_RANGE"
    """High minus low over a window. Never negative. Not the net change."""

    PERCENT_CHANGE = "PERCENT_CHANGE"
    """A proportion of a price: ``0.21%``. A percentage."""

    QUANTITY = "QUANTITY"
    """A measured amount with a unit that is not price: lots, ounces, holdings."""

    MASS_TONNES = "MASS_TONNES"
    """Tonnes, as ETF flows are reported. ``9.98 tấn`` is never a quote."""

    COUNT = "COUNT"
    """A whole number of things: sessions, candles, days."""

    MONETARY_NON_PRICE = "MONETARY_NON_PRICE"
    """Money that is not the instrument's price: ``141 triệu USD`` of inflows."""

    DATE_TIME = "DATE_TIME"
    """A clock time or calendar date. Digits, but not a number about the market."""

    UNKNOWN = "UNKNOWN"
    """Nothing vouched for it and nothing says it is a price either.

    Distinct from :attr:`UNKNOWN_PRICE_LIKE` because a bare ``9.98`` and a bare
    ``4323`` are both unexplained, and only one of them looks like gold.
    """

    @property
    def family(self) -> SemanticType:
        """The coarse meaning this member belongs to.

        The five original members are their own family. Every refinement maps
        to exactly one of them, so a check written against the coarse
        vocabulary keeps working when handed a refined value.
        """
        return _FAMILY.get(self, self)


_FAMILY: dict[SemanticType, SemanticType] = {
    SemanticType.PRICE_NET_CHANGE: SemanticType.MAGNITUDE,
    SemanticType.PRICE_RANGE: SemanticType.MAGNITUDE,
    SemanticType.PERCENT_CHANGE: SemanticType.PERCENTAGE,
    SemanticType.QUANTITY: SemanticType.NON_MARKET_NUMBER,
    SemanticType.MASS_TONNES: SemanticType.NON_MARKET_NUMBER,
    SemanticType.COUNT: SemanticType.NON_MARKET_NUMBER,
    SemanticType.MONETARY_NON_PRICE: SemanticType.NON_MARKET_NUMBER,
    SemanticType.DATE_TIME: SemanticType.NON_MARKET_NUMBER,
    SemanticType.UNKNOWN: SemanticType.UNKNOWN,
}
"""Refinement to family. Everything not listed is its own family."""


class DerivedFactKind(StrEnum):
    """The closed list of formulas the scanner will reproduce.

    Short on purpose. Every entry here is a number the article may state without
    citing anything, so the list is the exemption surface - and an exemption
    surface that grows with the square of the candle count is not one.
    """

    NET_CHANGE = "NET_CHANGE"
    NET_CHANGE_PERCENT = "NET_CHANGE_PERCENT"
    WINDOW_RANGE = "WINDOW_RANGE"


@dataclass(frozen=True)
class KnownNumber:
    """One value the article is entitled to state, and the reason why.

    ``origin`` exists so the scanner can answer "why was this accepted?" without
    anyone reading the implementation. It is safe to log: a path or a formula
    name, never article text.
    """

    value: Decimal
    semantic: SemanticType
    origin: str

    def satisfies(self, required: SemanticType) -> bool:
        """Whether this known value may stand in for *required*.

        No widening. A magnitude does not become a price because the numbers
        match, which is the entire point of carrying the type around.

        A requirement stated coarsely (``MAGNITUDE``) is met by any member of
        that family, so a value refined as ``PRICE_RANGE`` satisfies it. A
        requirement stated finely (``PRICE_NET_CHANGE``) is met only by that
        exact meaning - a range is not a net change, whatever the family. For
        the coarse members production constructs today this is identity.
        """
        if required.family is required:
            return self.semantic.family is required
        return self.semantic is required


REFINED_DERIVED_KIND: dict[DerivedFactKind, SemanticType] = {
    DerivedFactKind.NET_CHANGE: SemanticType.PRICE_NET_CHANGE,
    DerivedFactKind.NET_CHANGE_PERCENT: SemanticType.PERCENT_CHANGE,
    DerivedFactKind.WINDOW_RANGE: SemanticType.PRICE_RANGE,
}
"""What each derived formula means, in the refined vocabulary.

The formula catalog itself still labels its values with the coarse family -
that is production, and it is not touched. This table is how a later check
asks "which magnitude is this?" without the catalog having to change first.
Complete by construction; a test holds it equal to the enum.
"""


# --------------------------------------------------------------------------
# declared type of a context path
# --------------------------------------------------------------------------


def _role_of(field: FieldInfo) -> NumericRole | None:
    """Find the role marker on a declared field.

    Pydantic keeps unrecognised annotation members in ``metadata`` for a plain
    field, but an optional field (``Magnitude | None``) hides them inside the
    union, so both places are searched. ``ContextLevels.atr`` is exactly that
    second shape, which is why this is not a one-liner.
    """
    for item in field.metadata:
        if isinstance(item, NumericRole):
            return item

    def walk(annotation: Any, depth: int = 0) -> NumericRole | None:
        if depth > MAX_ROLE_DEPTH:
            return None
        for arg in get_args(annotation):
            if isinstance(arg, NumericRole):
                return arg
            found = walk(arg, depth + 1)
            if found is not None:
                return found
        return None

    return walk(field.annotation)


def classify_path(context: AnalysisContext, path: str) -> SemanticType:
    """The declared meaning of the value at *path*.

    Walks the same object graph :func:`resolve_path` walks, reading each step's
    declared field rather than guessing from what it finds. A path that does not
    resolve, or that ends somewhere with no numeric role, is
    ``UNKNOWN_PRICE_LIKE`` - unclassified is never a licence.
    """
    try:
        resolve_path(context, path)
    except ClaimPathError:
        return SemanticType.UNKNOWN_PRICE_LIKE

    current: Any = context
    role: NumericRole | None = None

    for segment in path.strip().split(".")[1:]:
        name = segment.split("[")[0]
        model = type(current)
        if not isinstance(current, BaseModel) or name not in model.model_fields:
            return SemanticType.UNKNOWN_PRICE_LIKE

        role = _role_of(model.model_fields[name])
        current = getattr(current, name)
        for index in _indices(segment):
            if not isinstance(current, list) or not -len(current) <= index < len(current):
                return SemanticType.UNKNOWN_PRICE_LIKE
            current = current[index]
            role = None  # the list element itself carries no role; its fields do

    if role is NumericRole.PRICE:
        return SemanticType.ABSOLUTE_PRICE
    if role is NumericRole.MAGNITUDE:
        return SemanticType.MAGNITUDE
    if isinstance(current, Decimal):
        # A declared number with no role: real, but not vouched for as a price.
        return SemanticType.NON_MARKET_NUMBER
    return SemanticType.NON_MARKET_NUMBER


def _indices(segment: str) -> list[int]:
    out: list[int] = []
    rest = segment[len(segment.split("[")[0]) :]
    for chunk in rest.split("[")[1:]:
        try:
            out.append(int(chunk.rstrip("]")))
        except ValueError:
            return out
    return out


# --------------------------------------------------------------------------
# rendering tolerance
# --------------------------------------------------------------------------


def rendered_as(value: Decimal, literal: str) -> bool:
    """Whether *literal* is a faithful rendering of *value*.

    An article may write ``66.140`` as ``66.14`` or ``66.1``; all three are the
    same fact. So the test is not "are these close" but "does the known value
    round to exactly what was printed, at the precision it was printed to".

    That distinction is what keeps the tolerance from becoming a loophole. A
    literal of ``4373.13`` accepts a known ``4373.127`` and rejects a known
    ``4373.14`` - the window is one ulp of the *printed* precision, never a
    fixed margin in points, so a nearby fabricated price is not blessed by a
    neighbouring real one.
    """
    text = literal.strip().replace(",", "").replace("+", "").rstrip("%")
    if not text:
        return False
    try:
        printed = Decimal(text)
    except (InvalidOperation, ValueError):
        return False

    exponent = printed.as_tuple().exponent
    if not isinstance(exponent, int):
        return False

    if exponent >= 0:
        # A bare integer gets no rounding latitude. This pipeline never renders
        # a price as a whole number - `format_price` pads to two decimals - so
        # "4326" is not a rounding of 4325.70, it is a different number. Allowing
        # it would hand every integer literal a half-point window in which some
        # real candle value can usually be found, which is how a fabricated
        # level gets blessed by a neighbour. Observed on a real draft: the
        # genuine findings for '4326' and '4331' vanished until this line existed.
        return value == printed

    quantum = Decimal(1).scaleb(exponent)
    try:
        return value.quantize(quantum, rounding=ROUND_HALF_UP) == printed
    except (InvalidOperation, ArithmeticError):
        return False


__all__ = [
    "REFINED_DERIVED_KIND",
    "DerivedFactKind",
    "KnownNumber",
    "SemanticType",
    "classify_path",
    "rendered_as",
]
