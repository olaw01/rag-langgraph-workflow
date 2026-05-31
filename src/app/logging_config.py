import logging # import python standard logging module

# This gives consistent, readable logs across modules

# Define a function to set global logging configuration; default level INFO
def configure_logging(level: str = "INFO") -> None:
    # Use basicConfig to configure the root logger for the whole app
    # Set log level (e.g., INFO/DEBUG). upper() makes it case-insensitive
    # Log line format: timestamp, level, logger name (module), and message
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )