from datetime import timedelta

from lib.latest_cache import LatestCache
from lib.oraku_settings import settings

# Valid variables (used for caching keys and as df column names)
VALID_VARIABLES = ("temp", "flow")

# the server side cache is only for the default request, so default city, default horizon and no or very recent 'from'
# TODO will need to be in the global app state as well if also used in health.
#  extract init like this into own module and also make it X loc since we're gonna need that in the future anyway.
#  Health endpoint needs some streamlined method to fetch all of them
#  in parallel, actually maybe it can/should use a custom method to just fetch the max run_ts once per var (table).
#  Redundant but you could also check that the last forecast run was in recent enough and successful, but meh.
# Also, this would directly support setting different caching policies for different variable and cities if needed :)
latest_caches = {
    var: LatestCache(timedelta(seconds=settings.expected_interval_sec), timedelta(seconds=settings.cache_tolerance_sec))
    for var in VALID_VARIABLES
}
