import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    from datetime import timedelta
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    import polars.selectors as cs

    from aare.params import read_params
    from aare.paths import METRICS_FOLDER

    return METRICS_FOLDER, cs, go, mo, pl, px, read_params, timedelta


@app.cell(hide_code=True)
def _(read_params):
    params = read_params()
    tz = params["general"]["timezone"]
    return params, tz


@app.cell(hide_code=True)
def _(METRICS_FOLDER, pl, timedelta, tz):
    model = "LR-dev"
    raw_metrics = pl.read_csv(METRICS_FOLDER / "raw" / f"{model}.csv")
    raw_metrics = raw_metrics.with_columns(pl.col("run_ts", "time").str.to_datetime(time_zone=tz))
    raw_metrics = raw_metrics.with_columns(lag=((pl.col("time") - pl.col("run_ts")) / timedelta(hours=1) + 1).cast(int))
    raw_metrics = raw_metrics.with_columns(ae=pl.col("err").abs(), adpd=pl.col("dpd").abs())
    return (raw_metrics,)


@app.cell(hide_code=True)
def _(mo, params):
    horizon_range = mo.ui.range_slider(
        1, params["general"]["forecast_horizon"], step=1, debounce=True, label="Horizons to analyze", value=[1, 36]
    )
    horizon_range
    return (horizon_range,)


@app.cell(hide_code=True)
def _(mo):
    hour_range = mo.ui.range_slider(0, 23, step=1, debounce=True, label="Hours to include", value=[0, 23])
    hour_range
    return (hour_range,)


@app.cell(hide_code=True)
def _(horizon_range, hour_range, pl, raw_metrics):
    horizon_filtered = raw_metrics.filter(
        pl.col("lag") >= horizon_range.value[0], pl.col("lag") <= horizon_range.value[1]
    )
    hour_filtered = horizon_filtered.filter(
        pl.col("time").dt.hour() >= hour_range.value[0], pl.col("time").dt.hour() <= hour_range.value[1]
    )
    return (hour_filtered,)


@app.cell(hide_code=True)
def _(cs, go, hour_filtered, pl):
    _col = pl.col("err")
    _df = (
        hour_filtered.lazy()
        .sort("run_ts")
        .group_by("run_ts")
        # .group_by_dynamic("time", every="1d")
        .agg(
            _col.abs().median(),
            _col.abs().quantile(0.125).name.suffix("_q125"),
            _col.abs().quantile(0.875).name.suffix("_q875"),
        )
        .sort("run_ts")
        .with_columns(cs.float().rolling_mean(30 * 24 // 5, center=True))
        .collect()
    )

    _times = _df["run_ts"]
    _fig = go.Figure(
        [
            go.Scatter(
                name="err",
                x=_times,
                y=_df["err"],
                mode="lines",
            ),
            go.Scatter(
                name="Q 87.5%",
                x=_times,
                y=_df["err_q875"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            ),
            go.Scatter(
                name="Q 12.5%",
                x=_times,
                y=_df["err_q125"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                fill="tonexty",
            ),
        ]
    )

    _fig.update_layout(
        yaxis=dict(
            title=dict(
                text="Absolute Forecast Error [°C]",
            )
        ),
        title=dict(
            text="Forecast error over validation period",
            subtitle=dict(
                text="Prediction interval shows quantiles for errors within each forecast. 75% of all errors lie within shaded area. Heavy smoothing applied."
            ),
        ),
        hovermode="x",
    )

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    month_range = mo.ui.range_slider(1, 12, step=1, debounce=True, label="Months to include", value=[4, 9])
    month_range
    return (month_range,)


@app.cell(hide_code=True)
def _(hour_filtered, month_range, pl):
    month_filtered = hour_filtered.filter(
        pl.col("time").dt.month() >= month_range.value[0], pl.col("time").dt.month() <= month_range.value[1]
    )
    return (month_filtered,)


@app.cell(hide_code=True)
def _(cs, month_filtered, pl):
    run_metrics = month_filtered.group_by("run_ts").agg(pl.col("ae", "adpd").mean().name.prefix("m"))

    model_metrics = run_metrics.select(
        cs.float().median(),
        cs.float().quantile(0.95).name.suffix("_q95"),
        cs.float().quantile(0.75).name.suffix("_q75"),
    ).to_dicts()[0]
    return model_metrics, run_metrics


@app.cell(hide_code=True)
def _(horizon_range, hour_range, mo, model_metrics, month_range):
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    mo.md(
        f"Only taking into account forecasts for the months **{months[month_range.value[0] - 1]} - {months[month_range.value[1] - 1]}** and hours **{hour_range.value[0]:02d}:00 - {hour_range.value[1]:02d}:00**, and only looking at predictions made **{horizon_range.value[0]} - {horizon_range.value[1]} hours** into the future, the absolute errors across all steps (hours) are averaged.\n\nThe median forecast has a mean absolute error (MAE) of **{model_metrics['mae']:.3f} °C**. Across the entire validation series, 75% of forecasts have a MAE of {model_metrics['mae_q75']:.3f} °C or less, and 95% have a MAE of {model_metrics['mae_q95']:.3f} °C or less.  \nIf only the maximum temperature each day is relevant (regardless of timing), then the median forecast is off by **{model_metrics['madpd']:.3f} °C.** 75% of all forecasts have a daily peak difference of {model_metrics['madpd_q75']:.3f} °C or less and 95% have a DPD of {model_metrics['madpd_q95']:.3f} °C or less."
    )
    return


@app.cell(hide_code=True)
def _(go, month_filtered, pl):
    _col = pl.col("err")
    _df = (
        month_filtered.lazy()
        .group_by("lag")
        .agg(
            _col.abs().median(),
            _col.abs().quantile(0.125).name.suffix("_q125"),
            _col.abs().quantile(0.875).name.suffix("_q875"),
        )
        .sort("lag")
        .collect()
    )

    _times = _df["lag"]
    _fig = go.Figure(
        [
            go.Scatter(
                name="err",
                x=_times,
                y=_df["err"],
                mode="lines",
            ),
            go.Scatter(
                name="Q 87.5%",
                x=_times,
                y=_df["err_q875"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            ),
            go.Scatter(
                name="Q 12.5%",
                x=_times,
                y=_df["err_q125"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                fill="tonexty",
            ),
        ]
    )

    _fig.update_layout(
        yaxis=dict(
            title=dict(
                text="Absolute Forecast Error [°C]",
            )
        ),
        xaxis=dict(
            title=dict(
                text="Hours into to future",
            )
        ),
        title=dict(
            text="Forecast error per lag",
            subtitle=dict(
                text="Prediction interval shows quantiles for errors within each lag. 75% of all errors lie within shaded area.."
            ),
        ),
        hovermode="x",
    )

    _fig
    return


@app.cell(hide_code=True)
def _(px, run_metrics):
    px.violin(run_metrics, x=["madpd", "mae"], box=True, title="Distribution of MAE and MADPD of all forecasts")
    return


if __name__ == "__main__":
    app.run()
