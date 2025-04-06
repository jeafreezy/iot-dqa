# Logging

This document provides examples for updating the default logging configuration in the `iot-dqa` package.

## Adding File Logging

You can log messages to a file using the `add_file_logging` function:

```python
from iot_dqa.utils import add_file_logging
# Example usage:
add_file_logging("custom-log-file.log")
```

## Configuring Logging Levels

You can configure the logging level for the package using the `configure_logging` function:

```python
from iot_dqa.utils import configure_logging
# Example usage:
configure_logging(logging.DEBUG)  # Set logging level to DEBUG
```