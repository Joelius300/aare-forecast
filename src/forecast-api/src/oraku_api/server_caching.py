from datetime import timedelta

from oraku_api.latest_cache import LatestCache
from oraku_api.oraku_settings import OrakuSettings

# valid variables are used as caching keys and must match the df column names
VALID_VARIABLES = ("temp", "flow")

CachesType = dict[str, LatestCache]


# the server side cache is only for the default request, so default city, default horizon and no or very recent 'from'
def init_latest_caches(settings: OrakuSettings) -> CachesType:
    # in the future, each location should have their own cache. maybe integrating more tightly into fetching process
    # like a wrapper makes sense so that for example the health endpoint can easily update all caches and return a
    # status without manually calling fetch_forecast for each. also, this would directly support setting different
    # caching policies for different variable and cities if needed.
    return {
        var: LatestCache(
            timedelta(seconds=settings.expected_interval_sec), timedelta(seconds=settings.cache_tolerance_sec)
        )
        for var in VALID_VARIABLES
    }
