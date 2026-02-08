from collections.abc import Sequence
from typing import TypedDict, NotRequired


class FeatureIdentifiers(TypedDict):
    targets: Sequence[str]
    future: NotRequired[Sequence[str]]
    past: NotRequired[Sequence[str]]
