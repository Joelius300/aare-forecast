import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from pathlib import Path
    from datetime import timedelta

    import polars as pl
    import polars.selectors as cs
    import plotly.express as px

    from aare.pl_utils import make_quantiles
    from aare_train.paths import DATA_FOLDER
    from aare_plotly.charts import quantile_line_chart

    return (
        DATA_FOLDER,
        Path,
        cs,
        make_quantiles,
        pl,
        px,
        quantile_line_chart,
        timedelta,
    )


@app.cell
def _(DATA_FOLDER):
    data_dir = DATA_FOLDER / "isolated_tasks" / "2025-05_bug-fix-eval"
    live_forecast_path = data_dir / "live_forecast_nowcasting_temp-1.0.parquet"
    simulated_forecast_10_path = data_dir / "sim_forecast_nowcasting_temp-1.0.parquet"
    simulated_forecast_11_path = data_dir / "sim_forecast_nowcasting_temp-1.1.parquet"
    actual_raw_path = data_dir / "actual_raw.parquet"
    return (
        actual_raw_path,
        live_forecast_path,
        simulated_forecast_10_path,
        simulated_forecast_11_path,
    )


@app.cell
def _():
    tz = "Europe/Zurich"
    return (tz,)


@app.cell
def _(Path, cs, pl, tz):
    def read_pq(path: Path):
        return (
            pl.scan_parquet(path)
            .with_columns(cs.datetime().dt.convert_time_zone(tz).dt.cast_time_unit("ns"), cs.float().cast(pl.Float32))
            .collect()
        )

    return (read_pq,)


@app.cell
def _(
    actual_raw_path,
    live_forecast_path,
    read_pq,
    simulated_forecast_10_path,
    simulated_forecast_11_path,
):
    live_forecasts = read_pq(live_forecast_path)
    sim_forecasts_10 = read_pq(simulated_forecast_10_path)
    sim_forecasts_11 = read_pq(simulated_forecast_11_path)
    actual_raw = read_pq(actual_raw_path)
    return actual_raw, live_forecasts, sim_forecasts_10, sim_forecasts_11


@app.cell
def _(pl):
    c_actual = pl.col("actual")
    c_run_ts = pl.col("run_ts")
    c_time = pl.col("time")
    return c_actual, c_run_ts, c_time


@app.cell
def _():
    lower_quant = 0.10
    inner_lower_quant = 0.25
    inner_upper_quant = 0.75
    upper_quant = 0.90
    quantiles = [lower_quant, inner_lower_quant, inner_upper_quant, upper_quant]
    return (quantiles,)


@app.cell
def _(c_run_ts):
    # the forecasts made at minute 15 are accurate because then all covariate data is up to date.
    # likely I'll change the schedule to only forecast at minute 15 in prod.
    accurate_run_filter = c_run_ts.dt.minute() == 15
    return (accurate_run_filter,)


@app.cell
def _(actual_raw, live_forecasts, pl, sim_forecasts_10, sim_forecasts_11):
    combined_forecasts = (
        live_forecasts.select("run_ts", "time", live="temp_bern")
        .join(
            sim_forecasts_10.select("run_ts", "time", pl.col("temp_bern").alias("simulated_1.0")),
            on=["run_ts", "time"],
            how="full",
            coalesce=True,
        )
        .join(
            sim_forecasts_11.select("run_ts", "time", pl.col("temp_bern").alias("simulated_1.1")),
            on=["run_ts", "time"],
            how="full",
            coalesce=True,
        )
        .join(
            actual_raw.select(time="_time", actual="temperature_bern").drop_nulls(),
            on="time",
            how="full",
            coalesce=True,
        )
        .select("run_ts", "time", "actual", "live", "simulated_1.0", "simulated_1.1")  # reorder
        .sort("run_ts", "time")
    )
    combined_forecasts
    return (combined_forecasts,)


@app.cell
def _(accurate_run_filter, combined_forecasts, pl):
    # based on the results of notebook 23 and the realization that casting everything to float32 before comparison helps,
    # here I try to confirm that the simulation is accurate if the input data is the same (which we now know is only given
    # for minute 15+, because the air temperature is updated in influxdb 15min later than it's measured).
    # interestingly, there are very few hours that are always off by a fixed amount or twice that amount. no clue what this is
    # and the difference is so small, I don't care about it for now.
    (
        combined_forecasts.filter(accurate_run_filter)
        .select("run_ts", "time", (pl.col("live") - pl.col("simulated_1.0")).abs().alias("simulation_error"))
        .filter(pl.col("simulation_error") > 0)
    )
    return


@app.cell
def _(accurate_run_filter, c_actual, combined_forecasts, pl):
    forecast_errors_sim = combined_forecasts.filter(accurate_run_filter).select(
        "run_ts",
        "time",
        (c_actual - pl.col("simulated_1.0")).abs().alias("ae_1.0"),
        (c_actual - pl.col("simulated_1.1")).abs().alias("ae_1.1"),
    )

    forecast_errors_sim.describe()
    return (forecast_errors_sim,)


@app.cell
def _(
    c_run_ts,
    c_time,
    forecast_errors_sim,
    make_quantiles,
    pl,
    quantile_line_chart,
    quantiles,
):
    _cols = ["ae_1.0", "ae_1.1"]
    _df = (
        forecast_errors_sim.lazy()
        .with_columns(lag=(c_time - c_run_ts).dt.total_hours(fractional=False) + 1)
        .group_by("lag")
        .agg(make_quantiles(pl.col(_cols), quantiles))
        .sort("lag")
        .collect()
    )

    quantile_line_chart(
        _df,
        _cols,
        x_col="lag",
        title="Absolute errors of 1.0 model and 1.1 model (both simulated)",
        yaxis_label="Absolute Error (°C)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As we can see in the chart above, the new model using FIRST features is clearly better in the first few hours, which is exactly what we hoped for. The median forecast of the first hour wen down from 0.12°C off to 0.07°C off. Note, the data this is using is very challenging sprint weather with high fluctuations. We would expect lower errors in summer when temperatures stabilize. In higher lags, the effect vanishes and general inaccuracy dominates the inaccuracy caused by the misaligned target. This is also expected, but it's important to confirm that the alignment correction does not lead to consistently worse performance at higher lags.

    Note, I didn't evaluate the diffpatch model (nowcasting_temp 1.0 with diff_threshold fixed to 1) separately, so the effects here are a combination of the target misalignment fix and the diff_threshold fix.

    Whether this new model with correctly aligned target is good enough for our use case is hard to say. Below is an exploration tool to visualize how different the forecasts would've looked at certain times. My feeling says it's good enough for a first release, but it would definitely be nice to implement bias correction. Chances are though, that I won't find the time.
    """)
    return


@app.cell
def _(mo):
    lookback_slider = mo.ui.slider(1, 36, step=1, value=24, debounce=True, label="Lookback (h):")
    horizon_slider = mo.ui.slider(12, 96, step=1, value=36, debounce=True, label="Horizon (h):")

    lookback_slider, horizon_slider
    return horizon_slider, lookback_slider


@app.cell
def _(c_run_ts, combined_forecasts):
    known_times = combined_forecasts.select(c_run_ts.drop_nulls().unique()).to_series().to_list()
    return (known_times,)


@app.cell
def _(
    c_run_ts,
    c_time,
    combined_forecasts,
    horizon_slider,
    known_times,
    lookback_slider,
    pl,
    px,
    time_slider,
    timedelta,
):
    _lookback = lookback_slider.value
    _horizon = horizon_slider.value
    _now = known_times[time_slider.value]
    _actual_available_til = _now - timedelta(minutes=8)  # visually adjust to the fact that db is updated 8 min late
    _start = _now - timedelta(hours=_lookback)
    _end = _now + timedelta(hours=_horizon)

    _df = (
        combined_forecasts.lazy()
        .filter(c_time >= _start, c_time <= _end, c_run_ts.is_null() | ((c_run_ts >= _start) & (c_run_ts <= _now)))
        .filter(c_run_ts.is_null() | (c_run_ts == c_run_ts.max()))
        .sort(c_time)
        .collect()
    )
    _marker_size = 6
    _marker_size_small = 3
    _size = (
        _df.select(
            pl.when(c_time > _actual_available_til).then(pl.lit(_marker_size_small)).otherwise(pl.lit(_marker_size))
        )
        .to_series()
        .to_list()
    )

    fig = px.scatter(
        _df,
        x="time",
        y=["actual", "live", "simulated_1.0", "simulated_1.1"],
        title=f"Simulated forecasts at {_now.isoformat()}",
        subtitle="Live forecast is also shown; should only rarely differ slightly from simulated_1.0",
    )
    fig.update_yaxes(title_text="Water temperature [°C]")
    fig.update_traces(marker=dict(size=_marker_size))
    fig.update_traces(selector=dict(name="actual"), marker=dict(size=_size, line_width=0))
    # workaround for bug fixed in https://github.com/plotly/plotly.py/pull/5508
    fig.add_vline(x=_now, line_width=2, line_dash="dot", opacity=0.75)
    fig.add_annotation(x=_now, y=0, yref="paper", text=_now.strftime("%H:%M"), showarrow=False)
    fig
    return


@app.cell
def _(known_times, mo):
    time_slider = mo.ui.slider(0, len(known_times), step=1, debounce=True, full_width=True)
    time_slider
    return (time_slider,)


if __name__ == "__main__":
    app.run()
