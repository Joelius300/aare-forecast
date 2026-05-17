# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.21.0",
#     "plotly[express]",
#     "polars",
#     "tzdata",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium", app_title="Aare Oraku Model Evaluation")


@app.cell
def _(mo):
    mo.md(r"""
    # Aare Oraku Model Evaluation

    This notebook shows the accuracy of different Aare Oraku models and allows comparisons with baselines.

    Use the dropdown to select which model to evaluate. Use the sliders to tune how the models are evaluated, respectively which forecasts or parts of forecasts are considered when calculating the metrics.
    Most users of aare.guru will only look at the forecasts during the daytime in summer.
    Which forecast horizon (how many hours into the future) people are interested in probably depends on many factors; with the slider you can evaluate different views.
    Beware that you can introduce biases, especially when selecting very strict evaluation criteria.

    It's important to note that this evaluation is very optimistic because all models that use external data as input like air temperature are evaluated on true measurement data.
    During inference (on aare.guru), this external data comes from forecasting services like MeteoTest, so it will contain inaccuracies that are propagated to our models.
    How well the model performs with external forecast inputs we can't know yet, but it's very likely that it will be worse than this evaluation shows.
    How much worse it will be depends on the accuracy of the external forecast services and sensitivity of our model.
    A [preliminary analysis of real forecasts](https://github.com/Joelius300/aare-forecast/blob/main/notebooks/21_measurement-vs-forecast-eval.ipynb) showed that they might be 15-25% worse than test forecasts of the same model
    during the same period. This is for the full 4-day forecasts, restricting evaluation to just the data shown in the
    aare.guru main page, it drops to ~10%. Relative to the actual temperature, it's only ~1% worse on average.

    If you uncheck the test dataset and instead look at the validation data, the evaluation will be even more optimistic because the validation set is used to tune the model and select the best one, which introduces a bias.
    The test set is designed to be evaluated only once a model is tuned and selected to avoid such biases and get the most realistic estimate for the real-world model accuracy.
    Luckily, the simple models used in the MVP are not overtuned and show very low differences between validation and test set.

    To make sure the Aare Oraku forecasts are accurate enough, they are continually monitored and evaluated.
    At the same time, all historical forecasts including all external inputs are stored for future evaluation.
    """)
    return


@app.cell(hide_code=True)
def _(default_model, mo, model_names):
    model_select = mo.ui.dropdown(options=model_names, value=default_model, label="Model to evaluate")
    model_select
    return (model_select,)


@app.cell(hide_code=True)
def _(mo, model):
    _out = None
    if model == "LR-dev-live-proto":
        _out = mo.md(
            "This model is special and you probably don't care for it. It was the prototype model that ran with real forecast data. When the test data checkbox is checked, you can analyze the evaluation with **measured** air temperature. If you uncheck it, it will _not_ use validation data, but instead it will show real historical forecast made with **forecasted** air temperature. This allows comparison between real historical forecasts and simulated forecasts on test data for the same time period."
        ).callout("info")
    elif model == "nowcasting_temp-1.0":
        _out = mo.md(
            "The 1.0 model was trained with hourly mean water temperature as target and is therefore systematically misaligned. The evaluation was retroactively corrected to use the new target (point in time measurements) for error calculation."
        ).callout("info")

    _out
    return


@app.cell(hide_code=True)
def _(mo):
    test_checkbox = mo.ui.checkbox(
        label="Whether to use test data for the evaluation (as opposed to validation data with even more bias)",
        value=True,
    )
    test_checkbox
    return (test_checkbox,)


@app.cell(hide_code=True)
def _(mo):
    month_range = mo.ui.range_slider(1, 12, step=1, debounce=True, label="Months to include", value=[4, 9])
    month_range
    return (month_range,)


@app.cell(hide_code=True)
def _(get_horizon_range, max_horizon, mo, set_horizon_range):
    # max_horizons depends on the data, so this need extra code to stay on the same value if the data changes
    horizon_range = mo.ui.range_slider(
        1,
        max_horizon,
        step=1,
        debounce=True,
        label="Horizons to analyze",
        value=get_horizon_range(),
        on_change=set_horizon_range,
    )
    horizon_range
    return (horizon_range,)


@app.cell(hide_code=True)
def _(mo):
    hour_range = mo.ui.range_slider(0, 23, step=1, debounce=True, label="Hours to include", value=[7, 21])
    hour_range
    return (hour_range,)


@app.cell(hide_code=True)
def _(horizon_range, hour_range, mo, model, model_metrics, month_range):
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    mo.md(
        f"## Evaluation of {model} model\n\nOnly taking into account forecasts for the months **{months[month_range.value[0] - 1]} - {months[month_range.value[1] - 1]}** and hours **{hour_range.value[0]:02d}:00 - {hour_range.value[1]:02d}:00**, and only looking at predictions made **{horizon_range.value[0]} - {horizon_range.value[1]} hours** into the future, the absolute errors across all steps (hours) are averaged.\n\nThe median forecast has a mean absolute error (MAE) of **{model_metrics['mae']:.3f} °C**. Across the entire evaluation data, 80% of forecasts have a MAE of {model_metrics['mae_q80']:.3f} °C or less, and 95% have a MAE of {model_metrics['mae_q95']:.3f} °C or less.  \nIf only the maximum/peak temperature each day is relevant (regardless of timing), then the median forecast is off by **{model_metrics['madpd']:.3f} °C.** 80% of all forecasts have a daily peak difference (DPD) of {model_metrics['madpd_q80']:.3f} °C or less and 95% have a DPD of {model_metrics['madpd_q95']:.3f} °C or less."
    ).callout("success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The charts below visualize the forecast errors in different dimensions. They show for example that the volatile summer temperatures are harder to predict and that predictions further into the future are less accurate (not surprising). The boxplots also show that the lowest errors have the highest density confirming that errors below the median are mostly concentrated to really low errors while larger errors are more spread out up to extreme outliers.

    The grayed out areas labeled 'ignored' are times excluded by the filter above. They are shown but not included in the metric calculations.
    """)
    return


@app.cell(hide_code=True)
def _(
    cs,
    eval_stride,
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
            _col.abs().quantile(0.1).name.suffix("_q10.0"),
            _col.abs().quantile(0.25).name.suffix("_q25.0"),
            _col.abs().quantile(0.75).name.suffix("_q75.0"),
            _col.abs().quantile(0.9).name.suffix("_q90.0"),
            # item instead of first would be better, but needs polars >=1.35.0 and pyodide is behind
            pl.col("start_time", "end_time").unique().first(),
        )
        .sort("run_ts")
        # smooth all runs in a week, at least one day. For this to work, must divide by stride (best case every hour is evaluated).
        .with_columns(
            cs.float().rolling_mean(7 * 24 // eval_stride, center=True, min_samples=24 // eval_stride),
            valid=month_filter,
        )
        .collect()
    )

    quantile_line_chart(
        _df,
        title="Forecast error over evaluation period",
        subtitle="Prediction interval shows quantiles for errors within each forecast. 80% of all errors lie within shaded area, 50% within the inner shading. Weekly smoothing applied.",
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
            _col.abs().quantile(0.1).name.suffix("_q10.0"),
            _col.abs().quantile(0.25).name.suffix("_q25.0"),
            _col.abs().quantile(0.75).name.suffix("_q75.0"),
            _col.abs().quantile(0.9).name.suffix("_q90.0"),
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
        subtitle="Prediction interval shows quantiles for errors within each lag. 80% of all errors lie within shaded area, 50% within the inner shading.",
    )
    return


@app.cell(hide_code=True)
def _(px, run_metrics):
    px.violin(run_metrics, x=["madpd", "mae"], box=True, title="Distribution of MAE and MADPD of all forecasts")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Below you can inspect individual forecasts to 'get a feeling' instead of just relying on metrics.
    """)
    return


@app.cell(hide_code=True)
def _(
    add_ignored_rects,
    all_run_ts,
    fc_i,
    get_invalid_periods,
    horizon_filter,
    hour_filter,
    month_filter,
    pl,
    px,
    raw_metrics,
):
    _df = raw_metrics.filter(fc_i=fc_i.value).with_columns(valid=month_filter & hour_filter & horizon_filter)
    _metrics = _df.filter("valid").select(pl.mean("ae", "adpd").mean().name.prefix("m")).to_dicts()[0]
    _metrics_fmt = {k: (f"{v:.3f}" if v is not None else "-") for k, v in _metrics.items()}

    _fig = px.line(
        _df,
        x="time",
        y=["pred", "actual"],
        color_discrete_sequence=["#73bf69", "#5694f2"],
        title=f"Forecast from {all_run_ts[fc_i.value]} | MAE: {_metrics_fmt['mae']} | MADPD: {_metrics_fmt['madpd']}",
    )
    _fig.update_yaxes(title_text="Water temperature [°C]")
    _fig.update_xaxes(title=None)

    _invalid_periods = get_invalid_periods(_df, "time")
    add_ignored_rects(_fig, _invalid_periods)

    _fig
    return


@app.cell
def _(all_run_ts, fc_i, mo):
    mo.md(rf"""
    Evaluation range: {all_run_ts[0]} - {all_run_ts[-1]}

    Currently displaying: {all_run_ts[fc_i.value]}
    """)
    return


@app.cell
def _(mo, pl, raw_metrics):
    all_run_ts = raw_metrics.select(pl.col("run_ts").unique(maintain_order=True)).to_series().to_list()
    fc_i = mo.ui.slider(0, raw_metrics.select(pl.max("fc_i")).item(), step=1, full_width=True)
    fc_i
    return all_run_ts, fc_i


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Below is the model file which shows when it was trained, what input variables and lags it uses, and some more hparams.
    """)
    return


@app.cell
async def _(baseline_models, get_meta_path, model):
    if model in baseline_models:
        _meta = "No meta file for baseline models"
    else:
        try:
            _meta_path = get_meta_path()

            if "http" in _meta_path:
                from pyodide.http import pyfetch

                _response = await pyfetch(_meta_path)
                _meta = await _response.json()
            else:
                import json

                with open(_meta_path) as _meta_file:
                    _meta = json.load(_meta_file)
        except Exception:
            _meta = f"Could not load meta file of {model}"

    _meta
    return


@app.cell
def _():
    import sys

    running_wasm = sys.platform == "emscripten"
    return (running_wasm,)


@app.cell
def _():
    # separately so markdown shows up faster
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import itertools
    import pathlib
    from datetime import timedelta
    import plotly
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    import polars.selectors as cs

    return cs, go, itertools, pathlib, pl, plotly, px, timedelta


@app.cell
def _(itertools):
    tz = "Europe/Zurich"
    # todo if not running wasm, populate this list with files from disk?
    models = {
        "nowcasting_temp": ["1.0", "1.2"],
        "LR-dev": [
            "live-proto",
        ],
    }
    baseline_models = ["LOCF", "SNAIVE", "MEAN"]

    model_names = [
        f"{model}-{version}"
        for model_key, versions in models.items()
        for model, version in zip(itertools.repeat(model_key), versions)
    ] + baseline_models
    default_model = "nowcasting_temp-1.2"
    return baseline_models, default_model, model_names, tz


@app.cell
def _(model_select):
    model = model_select.value
    return (model,)


@app.cell
def _(get_data_path):
    data_path = get_data_path()
    if not data_path:
        raise ValueError("Could not locate data file!")
    return (data_path,)


@app.cell
async def _(data_path, load_raw, tz):
    raw_metrics = await load_raw(data_path, tz)
    return (raw_metrics,)


@app.cell
def _(pl, raw_metrics):
    max_horizon = raw_metrics.select(pl.max("lag")).item()
    eval_stride = int(
        raw_metrics.select(pl.col("run_ts").unique(maintain_order=True).diff().mean()).item().total_seconds() / 3600
    )
    return eval_stride, max_horizon


@app.cell
def _(get_horizon_range, max_horizon, set_horizon_range):
    _cur_horizon_range = get_horizon_range()
    if max_horizon < _cur_horizon_range[1]:
        _start = _cur_horizon_range[0] if _cur_horizon_range[0] < max_horizon else 1
        set_horizon_range([_start, max_horizon])
    return


@app.cell
def _(horizon_range, hour_range, month_range, pl, raw_metrics):
    horizon_filter = (pl.col("lag") >= horizon_range.value[0]) & (pl.col("lag") <= horizon_range.value[1])
    hour_filter = (pl.col("time").dt.hour() >= hour_range.value[0]) & (pl.col("time").dt.hour() <= hour_range.value[1])
    # no need to filter month if 1-12 is allowed (gets rid of annoying edge case with forecasts across the year border)
    month_filter = pl.lit(month_range.value[0] == 1 and month_range.value[1] == 12) | (
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
        cs.float().quantile(0.80).name.suffix("_q80"),
        cs.float().quantile(0.95).name.suffix("_q95"),
    ).to_dicts()[0]
    return model_metrics, run_metrics


@app.cell
def _(add_ignored_rects, get_invalid_periods, go, pl, plotly):
    def quantile_line_chart(
        df: pl.DataFrame,
        x_col="run_ts",
        val_col="err",
        lower_quant=0.1,
        inner_lower_quant=0.25,
        inner_upper_quant=0.75,
        upper_quant=0.9,
        *,
        title: str,
        subtitle: str | None = None,
        yaxis_label: str,
        xaxis_label: str | None = None,
    ) -> go.Figure:
        x = df[x_col]
        weak_quant_color = "rgb(109, 210, 189)"
        strong_quant_color = "rgb(88, 170, 153)"
        fig = go.Figure(
            [
                go.Scatter(
                    name=f"Q {upper_quant:.1%}",
                    x=x,
                    y=df[f"{val_col}_q{upper_quant * 100}"],
                    mode="lines",
                    line=dict(width=0, color=weak_quant_color),
                    showlegend=False,
                ),
                go.Scatter(
                    name=f"Q {lower_quant:.1%}",
                    x=x,
                    y=df[f"{val_col}_q{lower_quant * 100}"],
                    mode="lines",
                    line=dict(width=0, color=weak_quant_color),
                    showlegend=False,
                    fill="tonexty",
                ),
                go.Scatter(
                    name=f"Q {inner_upper_quant:.1%}",
                    x=x,
                    y=df[f"{val_col}_q{inner_upper_quant * 100}"],
                    mode="lines",
                    line=dict(width=0, color=strong_quant_color),
                    showlegend=False,
                ),
                go.Scatter(
                    name=f"Q {inner_lower_quant:.1%}",
                    x=x,
                    y=df[f"{val_col}_q{inner_lower_quant * 100}"],
                    mode="lines",
                    line=dict(width=0, color=strong_quant_color),
                    showlegend=False,
                    fill="tonexty",
                ),
                # last so it's drawn on top of the quantile regions. must re-set color to first trace's color.
                go.Scatter(
                    name=val_col,
                    x=x,
                    y=df[val_col],
                    mode="lines",
                    line=dict(color=plotly.colors.DEFAULT_PLOTLY_COLORS[0]),
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
            invalid_periods = get_invalid_periods(df, x_col)
            add_ignored_rects(fig, invalid_periods)

        return fig

    return (quantile_line_chart,)


@app.cell
def _(pl):
    def get_invalid_periods(df: pl.DataFrame, index_col: str, valid_col="valid") -> pl.DataFrame:
        """Get start and end tuples for 'invalid' periods (~valid)."""
        return (
            df.lazy()
            .with_columns(blackout=~pl.col(valid_col))
            .with_columns(group_id=pl.col("blackout").rle_id())
            .filter("blackout")
            .group_by("group_id")
            .agg(
                start=pl.min(index_col),
                end=pl.max(index_col),
            )
            .drop("group_id")
            .collect()
        )

    return (get_invalid_periods,)


@app.cell
def _(go, pl):
    def add_ignored_rects(fig: go.Figure, invalid_periods: pl.DataFrame):
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

        # marimo x plotly update logic is fucked and sometimes annotations are left over and fuck up the chart.
        # trying to remove those leftover annotations here doesn't work because they aren't in the fig object at all...

    return (add_ignored_rects,)


@app.cell
def _(data_path, mo, pl, running_wasm, timedelta):
    @mo.cache
    async def load_raw(path: str, tz: str) -> pl.DataFrame:
        if not running_wasm:
            raw_metrics = pl.read_parquet(data_path)
        else:
            import pyarrow
            import pyarrow.parquet as pq
            from pyodide.http import pyfetch

            # polars is not compiled with parquet support in wasm builds, so use this workaround.
            # https://github.com/pola-rs/polars/issues/20876
            response = await pyfetch(data_path)
            table = pq.read_table(pyarrow.BufferReader(await response.bytes()))
            raw_metrics = pl.from_arrow(table)

        raw_metrics = (
            raw_metrics.lazy()
            # add lag column
            .with_columns(lag=((pl.col("time") - pl.col("run_ts")) / timedelta(hours=1) + 1).cast(int))
            # add absolute error columns
            .with_columns(ae=pl.col("err").abs(), adpd=pl.col("dpd").abs())
            # add unique integer id for each forecast
            .with_columns(pl.col("run_ts").rle_id().alias("fc_i"))
            .collect()
        )

        # add start and end time for each forecast run
        raw_metrics = raw_metrics.join(
            raw_metrics.group_by("run_ts").agg(pl.min("time").alias("start_time"), pl.max("time").alias("end_time")),
            on="run_ts",
        )

        return raw_metrics

    return (load_raw,)


@app.cell
def _(mo, model, pathlib, running_wasm, test_checkbox):
    def get_assets_dir() -> pathlib.PurePath:
        notebook_loc = mo.notebook_location()
        if running_wasm:
            return notebook_loc / "public" if notebook_loc is not None else None

        return notebook_loc.parent / "data"

    def get_data_path() -> str | None:
        file_name = model + ("-test" if test_checkbox.value else "") + ".parquet"
        asset_dir = get_assets_dir()
        if asset_dir is None:
            raise ValueError("Could not get asset dir")

        if running_wasm:
            return str(asset_dir / file_name)

        path = asset_dir / "metrics" / "raw" / file_name
        return str(path) if path.exists() else None

    def get_meta_path() -> str | None:
        file_name = model + ".json"
        asset_dir = get_assets_dir()
        if asset_dir is None:
            raise ValueError("Could not get asset dir")

        if running_wasm:
            return str(asset_dir / file_name)

        path = asset_dir / "models" / model / file_name
        return str(path) if path.exists() else None

    return get_data_path, get_meta_path


@app.cell
def _(mo):
    get_horizon_range, set_horizon_range = mo.state([1, 14])
    return get_horizon_range, set_horizon_range


if __name__ == "__main__":
    app.run()
