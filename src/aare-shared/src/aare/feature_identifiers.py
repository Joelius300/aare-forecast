from typing import TypedDict, NotRequired


class FeatureIdentifiers(TypedDict):
    targets: list[str]
    future: NotRequired[list[str]]
    past: NotRequired[list[str]]
