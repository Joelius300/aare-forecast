# Deployment guide

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