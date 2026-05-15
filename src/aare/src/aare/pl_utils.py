from collections.abc import Sequence

import polars as pl


def make_quantiles(col: pl.Expr, quantiles: Sequence[float], median_as_col: bool = True):
    """
    Get quantile expressions with '_q25.0' suffix for example.
    Aggregate the col as median without name change if median_as_col is true.
    """
    return ([col.median()] if median_as_col else []) + [
        col.quantile(q).name.suffix(f"_q{q * 100:.1f}") for q in quantiles
    ]
