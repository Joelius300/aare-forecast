# Backup

Deployment/dokku backups are handled, this is only about the data.
Can also be used to get a local copy of the data for analysis.

## Manual backup

Enter dokku shell

```bash
ssh -t "dokku@$DOKKU_HOST" shell
```

Temporarily expose database and note port

```
postgres:expose aare-oraku-forecast
```

Grab db password for example from

```
config:get aare-oraku-forecast ORAKU_CONNECTION_STRING
```

In another terminal on the host, use psql and/or pg_dump to get the data.
Must adjust port and user here and provide password when prompted.

```bash
pg_dump -h "$DOKKU_HOST" -p ... -U user -d aare_oraku_forecast -Fc -Z 9 -f aare_oraku_forecast.dump
```

```bash
psql -h "$DOKKU_HOST" -p ... -U user -d aare_oraku_forecast
```

When working with \copy and timescaledb don't forget to use a subquery and not just a table name.
```
\copy (select * from forecast) to 'forecast.csv'
```

Don't forget to unexpose the postgres database again before closing the dokku shell!

```bash
postgres:unexpose aare-oraku-forecast
```