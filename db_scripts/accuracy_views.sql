-- create error table
CREATE TABLE forecast_temp_error (
    run_ts      timestamptz NOT NULL,
    time        timestamptz NOT NULL,
    err         float NOT NULL,
    location    text NOT NULL,

    PRIMARY KEY (run_ts, time)
);

-- as hypertable
SELECT create_hypertable(
    'forecast_temp_error',
    'time'
);

-- create function to join data and put it into forecast_temp_error table
CREATE OR REPLACE PROCEDURE refresh_forecast_temp_error(job_id INT DEFAULT NULL, config JSONB DEFAULT NULL)
LANGUAGE plpgsql
AS $$
DECLARE
    lookback interval;
BEGIN
    -- Read lookback interval from config JSON
    lookback := COALESCE(
        (config->>'lookback')::interval,
        interval '2 days'
    );

    INSERT INTO forecast_temp_error (
        run_ts,
        time,
        err,
        location
    )
    SELECT
        f.run_ts,
        f.time,
        abs(a.temperature - f.temp_bern) AS err,
        'BERN' as location
    FROM forecast f
    INNER JOIN mirror_hydro a
        ON f.time = a.time
    WHERE a.location = 'BERN'
      AND a.temperature IS NOT NULL
      AND f.time >= now() - lookback
    ON CONFLICT (run_ts, time)
    DO UPDATE SET
        err = EXCLUDED.err;
END;
$$;

-- schedule job to automatically refresh errors. Note you could also configure initial_start, job_name, and timezone.
SELECT add_job(
    'refresh_forecast_temp_error',
    schedule_interval => interval '5 minutes',
    initial_start => now(),
    config => jsonb_build_object(
        'lookback', '2 days'
    )
);

-- 3. run once with all data
call refresh_forecast_temp_error(config => jsonb_build_object(
    'lookback', (now() - '2026-01-01T00:00:00')::text
));

-- 4. try compression here before rolling it out on the other tables; low stakes
ALTER TABLE forecast_temp_error SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'location',
    timescaledb.compress_orderby = 'run_ts DESC, time DESC'
);

SELECT add_compression_policy('forecast_temp_error', compress_after => INTERVAL '1 day');

-- 5. Query it like so

SELECT
    time_bucket(
        '1 week',
        time,
        timezone => 'Europe/Zurich'
    ) AS bucket,
    percentile_cont(0.5) within group (order by err) as "err_p50",
    percentile_cont(0.8) within group (order by err) as "err_p80",
    percentile_cont(0.95) within group (order by err) as "err_p95"
FROM (
    SELECT time, err
    FROM forecast_temp_error
    WHERE run_ts - time <= interval '36 hours'
)
GROUP BY bucket
ORDER BY bucket DESC;

-- TODO continuous aggregates, something like this. Beware, this requires the timescale-db toolkit extension, which
-- is only available in the timescaledb-ha image and I'm too scared to try and upgrade the db right now.
-- I guess we'll just use forecast_temp_error table for now, which should speed things up quite a bit already.

-- Daily aggregate
CREATE MATERIALIZED VIEW forecast_temp_error_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket(
        '1 day',
        time,
        timezone => 'Europe/Zurich'
    ) AS bucket,

    avg(err) AS mean_err,

    percentile_agg(err) AS err_percentiles
FROM forecast_temp_error
GROUP BY bucket;

-- Weekly aggregate stacked on daily. Can also do another one for monthly.
CREATE MATERIALIZED VIEW forecast_temp_error_weekly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket(
        '1 week',
        bucket,
        timezone => 'Europe/Zurich'
    ) AS bucket,

    mean(rollup(err_percentiles)) AS mean_err,

    rollup(err_percentiles) AS err_percentiles
FROM forecast_temp_error_daily
GROUP BY bucket;

