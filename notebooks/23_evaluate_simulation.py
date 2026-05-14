import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from datetime import datetime
    import subprocess

    import polars as pl
    import polars.selectors as cs
    import plotly.express as px
    import psycopg_pool
    import pytz

    from aare_timescale.postgres import copy_to_df_pl

    return copy_to_df_pl, cs, datetime, pl, psycopg_pool, px, pytz, subprocess


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Evaluating real vs simulated forecasts in bulk

    Using marimo for nicer interactivity, but this is still just an experimental (unreproducible) playground notebook.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt

    plt.rcParams["figure.figsize"] = (16, 9)
    return


@app.cell
def _():
    import logging

    logging.basicConfig(level="DEBUG")
    logging.getLogger("fsspec").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("dulwich").setLevel(logging.WARNING)
    logging.getLogger("MARKDOWN").setLevel(logging.WARNING)
    return


@app.cell
async def _(psycopg_pool):
    pool = psycopg_pool.AsyncConnectionPool(
        "host=127.0.0.1 dbname=aare_oraku user=postgres password=password", open=False
    )
    await pool.open()
    return (pool,)


@app.cell
def _(datetime):
    tz = "Europe/Zurich"
    since = datetime(2026, 1, 1).astimezone()
    return since, tz


@app.cell
async def _(copy_to_df_pl, cs, pl, pool, since, tz):
    async with pool.connection() as _conn:
        forecasts = await copy_to_df_pl(
            _conn, "SELECT * FROM forecast WHERE run_ts >= %(since)s ORDER BY run_ts, time", dict(since=since)
        )
        forecasts = (
            forecasts.with_columns(cs.string().str.to_datetime(time_zone=tz))
            .with_columns(pl.col("run_ts").dt.truncate("1h").alias("run_ts_hour"))
            .set_sorted(["run_ts", "time"])
        )
    forecasts.shape
    return (forecasts,)


@app.cell
async def _(copy_to_df_pl, pl, pool, since, tz):
    async with pool.connection() as _conn:
        forecast_meta = await copy_to_df_pl(
            _conn, "SELECT * FROM forecast_meta WHERE run_ts >= %(since)s ORDER BY run_ts", dict(since=since)
        )
        forecast_meta = (
            forecast_meta.with_columns(pl.col("run_ts", "finished_at").str.to_datetime(time_zone=tz))
            .with_columns(pl.col("run_ts").dt.truncate("1h").alias("run_ts_hour"))
            .set_sorted("run_ts")
        )
    forecast_meta.shape
    return (forecast_meta,)


@app.cell
def _(forecast_meta):
    forecast_meta.filter(model_name="nowcasting_temp", model_version="1.0")
    return


@app.cell
def _(forecast_meta, pl):
    # same query with forecasts = only successful. with forecast_meta also failures.
    # can of course only compare runs of the same model, oops.
    hist_run_ts = forecast_meta.filter(model_name="nowcasting_temp", model_version="1.0").select(
        pl.col("run_ts").unique()
    )
    hist_run_ts
    return (hist_run_ts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First checking if the forecast frequency of 15min results in different forecasts (in theory that should only be the case when meteotest updates their forecasts).
    """)
    return


@app.cell
def _(cs, forecasts, pl):
    # mean standard deviation within one hour, e.g. std of the 00:00, 00:15, 00:30 and 00:45 forecasts for every hour (and then global mean). it seems like there are tiny differences, as you can confirm below.
    forecasts.group_by(["run_ts_hour", "time"]).agg(cs.float().std()).select(pl.median("temp_bern"))
    return


@app.cell
def _(forecasts):
    forecasts.sort(["run_ts_hour", "time", "run_ts"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Must confirm if this comes from updated meteotest forecasts or is just due to rounding errors.
    """)
    return


@app.cell
async def _(copy_to_df_pl, cs, pl, pool, since, tz):
    async with pool.connection() as _conn:
        meteo = await copy_to_df_pl(
            _conn,
            "SELECT * FROM meteotest WHERE run_ts >= %(since)s AND location = 'BERN' ORDER BY run_ts, time",
            dict(since=since),
        )
        meteo = (
            meteo.set_sorted(["run_ts", "time"])
            .with_columns(pl.col("run_ts", "time").str.to_datetime(time_zone=tz), cs.ends_with("_error").cast(float))
            .with_columns(pl.col("run_ts").dt.truncate("1h").alias("run_ts_hour"))
        )
    meteo.shape
    return (meteo,)


@app.cell
def _(cs, meteo, pl):
    # there doesn't seem to be any std within hour forecasts, confirm with sorted values below
    meteo.group_by(["run_ts_hour", "time"]).agg(cs.float().std()).select(pl.median("tt"))
    return


@app.cell
def _(meteo):
    meteo.sort(["run_ts_hour", "time", "run_ts"])
    return


@app.cell
def _(cs, meteo, pl, px):
    changed_times = (
        meteo.lazy()
        .with_columns(in_day=pl.col("run_ts") - pl.col("run_ts").dt.truncate("1d"))
        .sort(["time", "run_ts"])
        .with_columns(cs.numeric().diff().abs() > 0.0001)
        .with_columns(changed=pl.any_horizontal(cs.boolean()))
        .group_by("run_ts")
        .agg(pl.col("changed").sum())
        .with_columns(pl.col("run_ts").dt.convert_time_zone("UTC"))
        .sort("run_ts")
        .collect()
    )

    px.line(changed_times, x="run_ts", y="changed")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    this initial analysis chart shows:
    - every hour (between 00:30:10 and 00:45:10 [the 10s is just because of service startup latency]) ~12 values are updated in the forecast. I would assume these are the next 12 hours.
    - there seems to be a large update every 6 hours (05:00, 11:00, 17:00, 23:00 UTC) for the entire horizon (5 days)
    - every 12 hours (07:00 and 19:00 UTC) 91 values are updated. not sure what that is about but good to know.

    Most importantly, MeteoTest does **not** update their forecasts more than once per hour (as of 02.05.2026).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulation of existing forecasts

    Since the forecasts appear to vary despite constant input (likely/hopefully due to rounding errors), we won't be able to get the exact values with simulation. But if the std of the two is in line with the std of multiple real forecasts within an hour, that should be proof that we can accurately recreate/simulate historical forecasts.
    """)
    return


@app.cell
def _(cs, hist_run_ts, pl):
    # first tested with 100 runs first, results in 0.65s per run, so for 12000 runs it would take 130min
    # to reduce the simulation time, only simulate one per hour (the first one) since we know the covariate
    # inputs don't change during the hour.
    simulation_run_ts = (
        hist_run_ts.with_columns(pl.col("run_ts").alias("run_ts_exact"))
        .group_by_dynamic("run_ts", every="1h")
        .agg(cs.all().first())
        .select(pl.col("run_ts_exact").dt.to_string())
        .to_series()
        .to_list()
    )
    simulation_run_ts
    return


@app.cell
def _(hist_run_ts, pl, subprocess):
    if False:
        proc = subprocess.Popen(
            "uv run src/oraku-forecast/main.py --logging-level INFO --model-path data/models/nowcasting_temp-1.0/nowcasting_temp-1.0.json --simulate-runts".split()
            # + simulation_run_ts  # <-- subsampled, see above
            + hist_run_ts.select(pl.col("run_ts").dt.to_string()).to_series().to_list()  # <-- all extracted run_ts
        )
    return


@app.cell
def _(cs, pl, tz):
    simulated_forecasts_a = pl.read_parquet("simulated_runs/2026-05-02T16:04:41.parquet").with_columns(
        cs.datetime().dt.convert_time_zone(tz).dt.cast_time_unit("us")
    )
    simulated_forecasts_a
    return (simulated_forecasts_a,)


@app.cell
def _(cs, pl, tz):
    simulated_forecasts_b = pl.read_parquet("simulated_runs/2026-05-02T20:54:49.parquet").with_columns(
        cs.datetime().dt.convert_time_zone(tz).dt.cast_time_unit("us")
    )
    simulated_forecasts_b
    return (simulated_forecasts_b,)


@app.cell
def _(pl, simulated_forecasts_a, simulated_forecasts_b):
    simulated_compare = simulated_forecasts_a.join(simulated_forecasts_b, on=("run_ts", "time"), suffix="_b")
    simulated_compare.with_columns(diff=(pl.col("temp_bern") - pl.col("temp_bern_b")).abs())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Oh no, both simulations today have the exact same outputs. Until now it seemed that same inputs could result in different outputs since run_ts at 00:15 and 00:30 had different values for the same hour despite confirming that meteotest values aren't updated in between those runs. Fuuuuuck. Maybe it's GPU vs CPU? Regardless we can't waste too much time on this...

    Ah turns out I was running with a default model of LR-dev, which is trained on FIRST features, not MEAN. Still, analysis below shows that there are differences between the prod forecasts and the simulated ones, but they are much smaller ~0.005°C on average and max 0.045°C. The difference increases with longer horizons due to compounding effects. The big question is why the hell are there no differences when you simulate the same thing twice but tbh I don't have time for that right now and the simulation is close enough to be useful for now.
    """)
    return


@app.cell
def _(cs, pl, tz):
    # simulation with nowcasting_temp-1.0 (like prod)
    simulated_forecasts_prod = pl.read_parquet(
        "simulated_runs/2026-05-03T12:34:25_nowcasting_temp-1.0.parquet"
        # "simulated_runs/2026-05-03T11:06:45_nowcasting_temp-1.0.parquet"
    ).with_columns(cs.datetime().dt.convert_time_zone(tz).dt.cast_time_unit("us"))
    simulated_forecasts_prod.shape
    return (simulated_forecasts_prod,)


@app.cell
def _(pl, simulated_forecasts_prod):
    # check if there is also inter-hour variance despite constant inputs. if not, this would confirm that there is some fundamental difference between the way forecasts are made, possible due to gpu vs cpu and it's likely that the simulation is working correctly. if there is variance, this could still be true, but it's more likely that the simulation isn't working correctly (than if there isn't variance).
    (
        simulated_forecasts_prod.with_columns(pl.col("run_ts").dt.truncate("1h").alias("run_ts_hour")).sort(
            ["run_ts_hour", "time", "run_ts"]
        )
    )
    return


@app.cell
def _(cs, pl, tz):
    # simulation with patched nowcasting_temp-1.0 to increase diff_threshold for water temperature
    simulated_forecasts_diffpatch = pl.read_parquet(
        "simulated_runs/2026-05-03T11:05:43_nowcasting_temp-1.0-diffpatch.parquet"
    ).with_columns(cs.datetime().dt.convert_time_zone(tz).dt.cast_time_unit("us"))
    return (simulated_forecasts_diffpatch,)


@app.cell
def _(forecasts, pl, simulated_forecasts_prod):
    actual_compare = (
        forecasts.select("run_ts", "time", temp_bern_prod="temp_bern")
        .join(simulated_forecasts_prod.select("run_ts", "time", temp_bern_simulated="temp_bern"), on=("run_ts", "time"))
        .with_columns(diff=(pl.col("temp_bern_prod") - pl.col("temp_bern_simulated")).abs())
    )
    actual_compare
    return (actual_compare,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Good and bad news.

    The simulated forecasts at 00:15, 00:30 and 00:45 are all very close and most likely only differ because of rounding error and small differences when transfering the data via postgres and also potentially cpu vs gpu. The diff seems to always be <0.000001°C.

    However, the first forecast so at 00:00, has a higher diff in the range of 0.005 in the higher horizons. This is a strong indicator that the simulation logic does something wrong, e.g. fetches different covariates than expected.

    **TODO investigate!!**
    """)
    return


@app.cell
def _(actual_compare):
    actual_compare.describe()
    return


@app.cell
def _(
    actual_compare,
    cs,
    datetime,
    pl,
    px,
    pytz,
    simulated_forecasts_diffpatch,
    tz,
):
    # shortest horizon forecasts (always <1h between run_ts and time). also filter to roughly when nowcasting_temp-1.0 was deployed.
    _best_forecasts = (
        actual_compare.filter(pl.col("run_ts") > datetime(2026, 3, 18, tzinfo=pytz.timezone(tz)))
        .join(
            simulated_forecasts_diffpatch.select("run_ts", "time", temp_bern_diffpatch="temp_bern"),
            on=["run_ts", "time"],
        )
        .sort(["time", "run_ts"])
        .group_by_dynamic("time", every="1h")
        .agg(cs.all().last())
    )

    px.line(
        _best_forecasts, x="time", y=["temp_bern_prod", "temp_bern_simulated", "temp_bern_diffpatch"]
    ).update_layout(hovermode="x")
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Okay sunntigsjoel hie ds update:

    - di zwöi runs wod gmacht hesch und hie referenziert si mitm LR-dev model gloffe wis ir dev_config.yaml steit entsprechend isch klar si d outputs nid di gliche wi ir produktion.
    - di zwöi runs si iz weird und fucky wöu d inputdate si no MEAN (hesch ja temporär wider reverted) aber trainiert ischs uf FIRST worde (siehe letscht change in dvc.lock)
    - iz am sunnti machsch:
      - nomau simulation la loufe mitm nowcasting-temp-1.0 und luege obs di gliche wärte git (sött!)
      - när mit nowcasting-temp-1.0-diffpatch simuliere und luege obs 1. besser wird und 2. haut plotte mit de angere zum luege obs di momente hät gfixt wos so blödi, fauschi egge het gä.
      - när mit LR-dev oder wasoimmer la loufe mit de gliche hparams wie nowcasting-temp-1.0 aber code und input uf FIRST statt MEAN

    -> continued work in 24_compare_simulations.ipynb
    """)
    return


if __name__ == "__main__":
    app.run()
