import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Literal, Optional, cast
from collections.abc import Sequence

from darts import TimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.models.forecasting.global_baseline_models import _GlobalNaiveModel

from aare_train.compat.types import DataTransformers

logger = logging.getLogger(__name__)


def historical_forecasts(
    model: ForecastingModel,
    val: list[TimeSeries],
    horizon: int,
    stride: int,
    parallel: bool | int | Literal["auto"] = False,
    verbose=False,
    future_cov: TimeSeries | Sequence[TimeSeries] | None = None,
    num_samples=128,
    data_transformers: Optional[DataTransformers] = None,
    random_state: Optional[int] = 42,
) -> list[list[TimeSeries]]:
    """Simulate historical forecasts with a pre-trained model on one or more validation series, optionally in parallel."""
    # does some validation and prep, then uses _historical_forecasts_parallel
    if len(val) == 0:
        raise ValueError("Must pass at least one validation series")

    if not model.supports_transferable_series_prediction:
        raise ValueError("Cannot evaluate a model which doesn't support transferable prediction.")

    # could break in any release since it's not public api
    if not model._supports_non_retrainable_historical_forecasts:  # pyright: ignore[reportPrivateUsage]
        # currently (24.03) the only models that support transferable series prediction but not
        # non-retrainable historical forecasts are local ensemble models IIRC.
        raise ValueError("Cannot evaluate a model which doesn't support non-retrainable historical forecasts.")

    parallel = _validate_parallel(parallel, model, val)

    if isinstance(model, _GlobalNaiveModel):
        # only takes the components etc. global naive don't care about the values
        model.fit(val[0])

    # simulate historical forecasts (without retraining!)
    return _historical_forecasts_parallel(
        model, val, stride, horizon, parallel, verbose, future_cov, num_samples, data_transformers, random_state
    )


def _historical_forecasts_parallel(
    model: ForecastingModel,
    val: list[TimeSeries],
    stride: int,
    horizon: int,
    parallel: bool | int,
    verbose: bool,
    future_cov: TimeSeries | Sequence[TimeSeries] | None,
    num_samples: int,
    data_transformers: Optional[DataTransformers],
    random_state: Optional[int],
) -> list[list[TimeSeries]]:
    if (
        future_cov is not None
        and not isinstance(future_cov, TimeSeries)
        and isinstance(future_cov, Sequence)
        and len(future_cov) != len(val)
    ):
        raise ValueError("When passing future_cov as a list, it must have the same number of entries as val")

    actual_num_samples = num_samples if model.supports_probabilistic_prediction else 1

    if not parallel:
        return cast(
            list[list[TimeSeries]],
            # (most likely) uses extreme_lags to find where to start forecasting
            # produces multiple forecasts (TimeSeries) with the specified stride FOR EACH SUBSERIES
            model.historical_forecasts(
                val,  # passing multiple ts so we get multiple sets of forecasts back
                # future_covariates can handle slices like val but also just a big chunk with the relevant data
                future_covariates=future_cov,
                stride=stride,
                forecast_horizon=horizon,
                last_points_only=False,
                retrain=False,
                num_samples=actual_num_samples,
                verbose=verbose,  # seemingly only for retraining, so probably useless
                data_transformers=data_transformers,
                random_state=random_state,
            ),
        )

    # keep track of the index so order can be reconstructed
    val_idx = list(enumerate(val))
    # longest series first
    prioritized = sorted(val_idx, key=lambda i_ts: len(i_ts[1]), reverse=True)

    # prepare covariates for both split and unified series
    if future_cov is None or isinstance(future_cov, TimeSeries):
        future_covs = [future_cov] * len(prioritized)
    else:
        assert isinstance(future_cov, Sequence), "future_cov is not a sequence?!"
        future_covs = []
        for i, _ in prioritized:
            future_covs.append(future_cov[i])

    logger.debug(f"Creating historical forecasts for [{', '.join((str(len(i_ts[1])) for i_ts in prioritized))}]")

    with ProcessPoolExecutor(max_workers=None if parallel is True else parallel) as pool:
        hf = pool.map(
            _parallel_forecast_step,
            prioritized,
            [model] * len(prioritized),
            [stride] * len(prioritized),
            [horizon] * len(prioritized),
            [actual_num_samples] * len(prioritized),
            future_covs,
            [data_transformers] * len(prioritized),
            [random_state] * len(prioritized),
        )

        hf = list(hf)  # makes it easier and the overhead is nothing
        # restore original order
        return [i_ts[1] for i_ts in sorted(hf, key=lambda iv: iv[0])]


def _parallel_forecast_step(
    i_ts: tuple[int, TimeSeries],
    model: ForecastingModel,
    stride: int,
    horizon: int,
    num_samples: int,
    future_cov: TimeSeries | None,
    data_transformers: Optional[DataTransformers],
    random_state: Optional[int],
):
    assert future_cov is None or isinstance(future_cov, TimeSeries), "Invalid type of future_cov"
    i, ts = i_ts
    forecasts = model.historical_forecasts(
        ts,  # passing now a single ts -> get a single set of forecasts
        future_covariates=future_cov,  # must now be a single series
        stride=stride,
        forecast_horizon=horizon,
        last_points_only=False,
        retrain=False,
        num_samples=num_samples,
        verbose=False,  # no need in another process
        data_transformers=data_transformers,
        random_state=random_state,
    )

    return i, cast(list[TimeSeries], forecasts)


def _validate_parallel(
    parallel: bool | int | Literal["auto"], model: ForecastingModel, val: list[TimeSeries]
) -> bool | int:
    # Thank you, Python, you did it again... bool is a subclass of int OMFG
    if parallel == "auto":
        # not sure if this is a good heuristic, but probably not too bad for us
        # TODO could add a guard that it still uses parallelization if horizon > output_chunk_length
        #  because AR is never optimized IIRC. NOPE, for LR not true anymore since 0.40.0!
        parallel = False if model.supports_optimized_historical_forecasts else True
    elif parallel is False:
        if not model.supports_optimized_historical_forecasts and len(val) > 1:
            logger.warning(
                "Model does not support optimized historical forecasts and you're evaluating multiple "
                + "validation series; you might benefit from parallelized evaluation."
            )
    elif parallel is True or type(parallel) is int:
        if model.supports_optimized_historical_forecasts:
            logger.warning(
                "Model already supports optimized forecasts, unclear if parallelization improves performance"
            )
        if len(val) == 1:
            logger.warning("Only evaluating on a single validation slice, parallelization does not make sense.")
            parallel = False
        if type(parallel) is int and parallel < 2:
            raise ValueError("Parallelization must be of degree 2 or higher")
    else:
        raise ValueError(f"Invalid option for parallel: {parallel}")

    # now parallel can only be True, False or an int > 2
    assert type(parallel) in (bool, int), "parallel is something other than bool or int?!"

    return parallel
