# Deployment guide

Always make sure `DOKKU_HOST` is set correctly.

I've not worked with it yet, but it seems dokku has their own remote client script: <https://dokku.com/docs/deployment/remote-commands/>

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

```bash
ssh "dokku@$DOKKU_HOST" config aare-oraku-forecast
ssh "dokku@$DOKKU_HOST" config:set aare-oraku-forecast ORAKU_LOKI_URL=...
```

## Running service container

```bash
ssh -t "dokku@$DOKKU_HOST" run aare-oraku-forecast
```

## Open DB connection

```bash
ssh -t "dokku@$DOKKU_HOST" postgres:connect aare-oraku-forecast
```