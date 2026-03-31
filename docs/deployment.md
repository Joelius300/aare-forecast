# Deployment guide

For initial service setup, see [initial dokku setup](#initial-dokku-setup).

Always make sure `DOKKU_HOST` is set correctly.

I've not worked with it yet, but it seems dokku has their own remote client script:
<https://dokku.com/docs/deployment/remote-commands/>

## Building and deploying

```bash
just build-service v0.2.0
just deploy-service v0.2.0
```
resp.
```bash
just build-api v0.2.0
just deploy-api v0.2.0
```

### Uploading model

```bash
just upload-model nowcasting_temp-1.0
```

## Executing dokku commands

Either open an interactive shell/prompt with

```bash
ssh -t "dokku@$DOKKU_HOST" shell
```

or run directly

```bash
ssh "dokku@$DOKKU_HOST" config aare-oraku-forecast
ssh "dokku@$DOKKU_HOST" config:set aare-oraku-forecast ORAKU_LOKI_URL=...
```

## Running service container (one-off)

```bash
ssh -t "dokku@$DOKKU_HOST" run aare-oraku-forecast
```

You get a bash prompt from this where you can open a python repl or whatever.

To run SQL directly, you can open an SQL repl directly via `postgres:connect` (see below).
If you need python to construct the SQL or need library functions or whatever, use a snippet like this.

```python
import psycopg
import os

conn = psycopg.connect(os.getenv("ORAKU_CONNECTION_STRING"), autocommit=True)
cur = conn.execute("""...""")
cur.close()
conn.close()
```

## Open DB connection (SQL repl)

```bash
ssh -t "dokku@$DOKKU_HOST" postgres:connect aare-oraku-forecast
```

See also [backup.md](backup.md) for how to use local psql to download/export data.

# Initial dokku setup

Before a service can be deployed, it must be created in the dokku instance.
Here are some examples how this could be done. This is just for inspiration, the exact commands are different.

```bash
dokku apps:create aare-oraku-forecast
dokku ps:set aare-oraku-forecast restart-policy no
dokku postgres:create aare-oraku-forecast -i timescale/timescaledb -I latest-pg17
dokku postgres:link aare-oraku-forecast aare-oraku-forecast
dokku config:set aare-oraku-forecast ORAKU_CONNECTION_STRING=$(dokku config:get aare-oraku-forecast DATABASE_URL)
dokku storage:ensure-directory aare-oraku-models
dokku storage:mount aare-oraku-forecast /var/lib/dokku/data/storage/aare-oraku-models:/models
dokku config:set aare-oraku-forecast ORAKU_MODEL_PATH=/models/LR-dev/LR-dev.json
```

```bash
dokku apps:create aare-oraku-api
dokku domains:set aare-oraku-api aare-oraku.$DOKKU_HOST
dokku letsencrypt:enable aare-oraku-api
dokku postgres:link aare-oraku-forecast aare-oraku-api
dokku config:set aare-oraku-api ORAKU_CONNECTION_STRING=$(dokku config:get aare-oraku-api DATABASE_URL)
```

sometimes you need to escape spaces etc. in config/env variables.

```bash
dokku apps:create aare-oraku-influx2pg
dokku ps:set aare-oraku-influx2pg restart-policy no
dokku postgres:link aare-oraku-forecast aare-oraku-influx2pg
dokku config:set aare-oraku-influx2pg ORAKU_CONNECTION_STRING=$(dokku config:get aare-oraku-influx2pg DATABASE_URL)
dokku config:set --no-restart aare-oraku-influx2pg "ORAKU_FIELDS=[\"hydro/temperature:mean_1h@bern\",\"hydro/flow:mean_1h@bern\",\"smn/tt:mean_1h@bern\"]"
```