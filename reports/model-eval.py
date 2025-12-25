import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium", app_title="Aare Oraku Model Eval")


@app.cell
def _():
    # TODO: add interactive chart with slider to slide through all available forecasts, to kind of get a feel for a model
    return


@app.cell
def _(default_model, mo, model_names):
    model_select = mo.ui.dropdown(options=model_names, value=default_model, label="Model to evaluate")
    model_select
    return (model_select,)


@app.cell(hide_code=True)
def _(mo):
    month_range = mo.ui.range_slider(1, 12, step=1, debounce=True, label="Months to include", value=[4, 9])
    month_range
    return (month_range,)


@app.cell(hide_code=True)
def _(max_horizon, mo):
    horizon_range = mo.ui.range_slider(
        1, max_horizon, step=1, debounce=True, label="Horizons to analyze", value=[1, 36]
    )
    horizon_range
    return (horizon_range,)


@app.cell(hide_code=True)
def _(mo):
    hour_range = mo.ui.range_slider(0, 23, step=1, debounce=True, label="Hours to include", value=[0, 23])
    hour_range
    return (hour_range,)


@app.cell(hide_code=True)
def _(horizon_range, hour_range, mo, model_metrics, month_range):
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    mo.md(
        f"Only taking into account forecasts for the months **{months[month_range.value[0] - 1]} - {months[month_range.value[1] - 1]}** and hours **{hour_range.value[0]:02d}:00 - {hour_range.value[1]:02d}:00**, and only looking at predictions made **{horizon_range.value[0]} - {horizon_range.value[1]} hours** into the future, the absolute errors across all steps (hours) are averaged.\n\nThe median forecast has a mean absolute error (MAE) of **{model_metrics['mae']:.3f} °C**. Across the entire validation series, 75% of forecasts have a MAE of {model_metrics['mae_q75']:.3f} °C or less, and 95% have a MAE of {model_metrics['mae_q95']:.3f} °C or less.  \nIf only the maximum temperature each day is relevant (regardless of timing), then the median forecast is off by **{model_metrics['madpd']:.3f} °C.** 75% of all forecasts have a daily peak difference of {model_metrics['madpd_q75']:.3f} °C or less and 95% have a DPD of {model_metrics['madpd_q95']:.3f} °C or less."
    ).callout("success")
    return


@app.cell(hide_code=True)
def _(
    cs,
    horizon_filter,
    hour_filter,
    month_filter,
    pl,
    quantile_line_chart,
    raw_metrics,
):
    _col = pl.col("err")
    _df = (
        raw_metrics.lazy()
        .filter(hour_filter, horizon_filter)
        .sort("run_ts")
        .group_by("run_ts")
        # .group_by_dynamic("time", every="1d")
        .agg(
            _col.abs().median(),
            _col.abs().quantile(0.125).name.suffix("_q12.5"),
            _col.abs().quantile(0.875).name.suffix("_q87.5"),
            # pl.col("start_time", "end_time").unique().item(),  # this would be better, but needs polars >=1.35.0
            pl.col("start_time", "end_time").unique().first(),
        )
        .sort("run_ts")
        .with_columns(cs.float().rolling_mean(30 * 24 // 5, center=True), valid=month_filter)
        .collect()
    )

    quantile_line_chart(
        _df,
        title="Forecast error over validation period",
        subtitle="Prediction interval shows quantiles for errors within each forecast. 75% of all errors lie within shaded area. Heavy smoothing applied.",
        yaxis_label="Absolute Forecast Error [°C]",
    )
    return


@app.cell(hide_code=True)
def _(
    horizon_filter,
    hour_filter,
    month_filter,
    pl,
    quantile_line_chart,
    raw_metrics,
):
    _col = pl.col("err")
    _df = (
        raw_metrics.lazy()
        .filter(month_filter, hour_filter)
        .group_by("lag")
        .agg(
            _col.abs().median(),
            _col.abs().quantile(0.125).name.suffix("_q12.5"),
            _col.abs().quantile(0.875).name.suffix("_q87.5"),
        )
        .sort("lag")
        .with_columns(valid=horizon_filter)
        .collect()
    )

    quantile_line_chart(
        _df,
        "lag",
        yaxis_label="Absolute Forecast Error [°C]",
        xaxis_label="Hours into to future",
        title="Forecast error per hour into the future",
        subtitle="Prediction interval shows quantiles for errors within each lag. 75% of all errors lie within shaded area.",
    )
    return


@app.cell(hide_code=True)
def _(px, run_metrics):
    px.violin(run_metrics, x=["madpd", "mae"], box=True, title="Distribution of MAE and MADPD of all forecasts")
    return


@app.cell
async def _():
    import sys

    running_wasm = sys.platform == "emscripten"

    if running_wasm:
        import micropip

        await micropip.install("tzdata")
        await micropip.install("plotly[express]")
        await micropip.install("polars")
    return (running_wasm,)


@app.cell
def _():
    import itertools
    from datetime import timedelta
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    import polars.selectors as cs

    return cs, go, itertools, pl, px, timedelta


@app.cell
def _(itertools):
    tz = "Europe/Zurich"
    models = {
        "LR": [
            "dev",
        ],
    }
    baseline_models = ["LOCF", "SNAIVE", "MEAN", "TIMESFM"]

    model_names = [
        f"{model}-{version}"
        for model_key, versions in models.items()
        for model, version in zip(itertools.repeat(model_key), versions)
    ] + baseline_models
    default_model = "LR-dev"
    return default_model, model_names, tz


@app.cell
def _(model_select):
    model = model_select.value
    return (model,)


@app.cell
def _(mo, model, running_wasm):
    def get_data_path() -> str | None:
        notebook_loc = mo.notebook_location()
        if running_wasm:
            return str(notebook_loc / "public" / f"{model}.csv") if notebook_loc is not None else None

        # avoid importing aare.paths, because it will then try to fetch it from pypi, which is a security risk
        metrics_folder = notebook_loc.parent / "data" / "metrics"
        path = metrics_folder / "raw" / f"{model}.csv"
        return str(path) if path.exists() else None

    data_path = get_data_path()
    if not data_path:
        raise ValueError("Could not locate data file!")
    return (data_path,)


@app.cell
def _(data_path, mo, pl, timedelta, tz):
    @mo.cache
    def load_raw(path: str, tz: str) -> pl.DataFrame:
        raw_metrics = pl.read_csv(data_path)

        # parse timestamps
        raw_metrics = raw_metrics.with_columns(pl.col("run_ts", "time").str.to_datetime(time_zone=tz))
        # add lag (horizon) column
        raw_metrics = raw_metrics.with_columns(
            lag=((pl.col("time") - pl.col("run_ts")) / timedelta(hours=1) + 1).cast(int)
        )
        # add error columns
        raw_metrics = raw_metrics.with_columns(ae=pl.col("err").abs(), adpd=pl.col("dpd").abs())

        # add start and end time for each forecast run
        raw_metrics = raw_metrics.join(
            raw_metrics.group_by("run_ts").agg(pl.min("time").alias("start_time"), pl.max("time").alias("end_time")),
            on="run_ts",
        )

        return raw_metrics

    raw_metrics = load_raw(data_path, tz)
    return (raw_metrics,)


@app.cell
def _(pl, raw_metrics):
    max_horizon = raw_metrics.select(pl.max("lag")).item()
    return (max_horizon,)


@app.cell
def _(horizon_range, hour_range, month_range, pl, raw_metrics):
    horizon_filter = (pl.col("lag") >= horizon_range.value[0]) & (pl.col("lag") <= horizon_range.value[1])
    hour_filter = (pl.col("time").dt.hour() >= hour_range.value[0]) & (pl.col("time").dt.hour() <= hour_range.value[1])
    month_filter = (
        (pl.col("start_time").dt.month() >= month_range.value[0])
        & (pl.col("end_time").dt.month() <= month_range.value[1])
        & (pl.col("start_time").dt.month() <= pl.col("end_time").dt.month())
    )

    filtered_all = raw_metrics.filter(month_filter, horizon_filter, hour_filter)
    return filtered_all, horizon_filter, hour_filter, month_filter


@app.cell
def _(cs, filtered_all, pl):
    run_metrics = filtered_all.group_by("run_ts").agg(pl.col("ae", "adpd").mean().name.prefix("m"))

    model_metrics = run_metrics.select(
        cs.float().median(),
        cs.float().quantile(0.95).name.suffix("_q95"),
        cs.float().quantile(0.75).name.suffix("_q75"),
    ).to_dicts()[0]
    return model_metrics, run_metrics


@app.cell
def _(go, pl):
    def quantile_line_chart(
        df: pl.DataFrame,
        x_col="run_ts",
        val_col="err",
        lower_quant=0.125,
        upper_quant=0.875,
        *,
        title: str,
        subtitle: str | None = None,
        yaxis_label: str,
        xaxis_label: str | None = None,
    ) -> go.Figure:
        x = df[x_col]
        fig = go.Figure(
            [
                go.Scatter(
                    name=val_col,
                    x=x,
                    y=df[val_col],
                    mode="lines",
                ),
                go.Scatter(
                    name=f"Q {upper_quant:.1%}",
                    x=x,
                    y=df[f"{val_col}_q{upper_quant * 100}"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                ),
                go.Scatter(
                    name=f"Q {lower_quant:.1%}",
                    x=x,
                    y=df[f"{val_col}_q{lower_quant * 100}"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    fill="tonexty",
                ),
            ]
        )

        fig.update_layout(
            yaxis=dict(
                title=dict(
                    text=yaxis_label,
                )
            ),
            xaxis=dict(
                title=dict(
                    text=xaxis_label,
                )
            ),
            title=dict(
                text=title,
                subtitle=dict(
                    text=subtitle,
                ),
            ),
            hovermode="x",
        )

        if "valid" in df.columns:
            invalid_periods = (
                df.lazy()
                .with_columns(blackout=~pl.col("valid"))
                .with_columns(group_id=pl.col("blackout").rle_id())
                .filter("blackout")
                .group_by("group_id")
                .agg(
                    start=pl.min(x_col),
                    end=pl.max(x_col),
                )
                .drop("group_id")
                .collect()
            )

            for start, end in invalid_periods.iter_rows():
                fig.add_vrect(
                    start,
                    end,
                    line_width=0,
                    fillcolor="darkgrey",
                    opacity=0.2,
                    annotation_text="ignored",
                    annotation_position="top left",
                )

        return fig

    return (quantile_line_chart,)


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
