from collections.abc import Sequence
import logging

logger = logging.getLogger(__name__)


def setup_logging(
    level: str,
    loki_url: str | None,
    loki_pw: str | None,
    loki_app_name: str | None = None,
    additional_loggers: Sequence[str] | None = None,
):
    """
    Sets up basic logging config, ignores some spammy logs and sets up loki if provided.
    Loki can be registered for additional logging scopes, useful for non-propagated loggers like uvicorn.
    """
    logging.basicConfig(level=level)
    logging.getLogger("dulwich").setLevel(logging.WARNING)
    logging.getLogger("fsspec").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    if not loki_url:
        return

    if not loki_app_name:
        raise ValueError("If loki should be configured, must pass an app name.")

    if level == "DEBUG":
        logger.warning(f"Loki not configured because logging level is '{level}'")
        return

    # the initial None is to get the root logger
    scopes: list[str | None] = [None] + (list(additional_loggers) if additional_loggers else [])

    import logging_loki

    loki_handler = logging_loki.LokiHandler(
        url=f"{loki_url}/loki/api/v1/push",
        tags={"application": loki_app_name},
        auth=("loki", loki_pw) if loki_pw else None,
        version="2",
        verify_ssl=loki_url.startswith("https"),
    )

    for scope in scopes:
        root_logger = logging.getLogger(scope)
        root_logger.addHandler(loki_handler)
