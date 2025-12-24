from datetime import datetime, timedelta, UTC

from fastapi import Response

HTTP_TIME_FMT = "%a, %d %b %Y %H:%M:%S GMT"


def get_last_modified(run_ts: datetime):
    return run_ts.astimezone(UTC).strftime(HTTP_TIME_FMT)


def set_client_caching(
    response: Response, now: datetime, run_ts: datetime, expected_interval: timedelta, cache_tolerance: timedelta
):
    """Set the Cache-Control and Last-Modified headers depending on the run_ts, current time and expected interval."""
    # no matter what data it is (city, horizon, ss cache hit or not, ...), we expect it to be updated in a
    # specific interval. Using that, we can calculate for how long the response should be valid/fresh.
    # Since we use a simple time-based heuristic, we subtract a tolerance to hopefully never wrongfully claim
    # data is still valid (at the cost of saying a response might be stale earlier resulting in more requests).
    # This version heavily emphasizes fresh data for everyone at the cost of more requests to our servers.
    likely_valid_for = run_ts + expected_interval - now - cache_tolerance
    max_age_sec = int(likely_valid_for.total_seconds())
    if max_age_sec > 0:
        response.headers["Cache-Control"] = f"public, max-age={max_age_sec}"
    else:
        # if we're close to an update, tell the client that it cannot reuse the response we're now sending without
        # checking with the server first to see if there was an update.
        response.headers["Cache-Control"] = "public, max-age=0, must-revalidate"

    # One thing I didn't implement is `stale-while-revalidate`. This would allow the browsers or CDNs to immediately
    # server stale data, but start a background process to fetch the latest data so the next request should get the
    # fresh version. For this to work, the SWR time needs to cover the update time, so in our case maybe 2*tolerance.
    # Secondly, this works best if the API is behind a CDN so only one user gets the stale data, then everyone else
    # gets fresh versions. If you don't have a CDN, the first request _per device_ will be stale and only when the next
    # is sent (user refreshed somehow), it will be fresh again. The benefit is a faster response at the cost of slightly
    # outdated data. Since we don't plan on using a CDN and the request is pretty fast anyway, we don't need this. If
    # we had tons of users, this plus a CDN would definitely make sense to avoid a stampede of requests when everyone's
    # max-age expires simultaneously. Also note that client-side caching without intermediary is much less effective anyway.

    # Additionally, setting Last-Modified allows the client to send a request if If-Modified-Since which we can
    # catch and compare to the run_ts. If it didn't change, we don't need to send the data over the network and
    # can just return 304 NOT MODIFIED. This saves bandwidth, not server compute. And it's kinda sexy.
    response.headers["Last-Modified"] = get_last_modified(run_ts)


def response_still_fresh(if_modified_since: str | None, run_ts: datetime) -> bool:
    """Check if the provided timestamp matches the run_ts. If true, the data has not changed since the last request."""
    if not if_modified_since:
        return False

    # the datetime from the header needs to be _interpreted_ as UTC, the run_ts needs to be _converted_ to UTC!
    last_changed_req = datetime.strptime(if_modified_since, HTTP_TIME_FMT).replace(tzinfo=UTC)
    run_ts = run_ts.astimezone(UTC).replace(microsecond=0)

    return last_changed_req == run_ts
