from typing import Any


def trav(json: Any, *path: str) -> Any:
    """Traverse dict-like but with better error msg if a key is missing (raises Value- or KeyError)."""
    cur = json
    traversed: list[str] = []
    for seg in path:
        if cur is None:
            raise ValueError(f"Object at '{'.'.join(traversed)}' is None, cannot traverse further to '{seg}'")
        if not isinstance(cur, dict):
            raise ValueError(
                f"Passed json has reached a non-dict at '{'.'.join(traversed)}', but tried to keep going deeper with '{seg}'"
            )
        if seg not in cur:
            raise KeyError(f"Key '{seg}' missing after '{'.'.join(traversed)}'")

        cur = cur[seg]
        traversed.append(seg)

    return cur
