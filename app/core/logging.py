"""Structured (key=value) logging configuration."""

import logging
import logging.config

LOG_FORMAT = "ts=%(asctime)s level=%(levelname)s logger=%(name)s %(message)s"


def configure_logging(level: str = "INFO") -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"kv": {"format": LOG_FORMAT, "datefmt": "%Y-%m-%dT%H:%M:%S%z"}},
            "handlers": {
                "console": {"class": "logging.StreamHandler", "formatter": "kv"},
            },
            "root": {"level": level.upper(), "handlers": ["console"]},
        }
    )
