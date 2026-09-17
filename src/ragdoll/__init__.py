"""ragdoll: staged, method-agnostic RAG implementation and evaluation harness."""

import logging

LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'

# This is an application, not a library other projects import, so configuring
# handlers on import (normally a library anti-pattern) is fine here: it means
# every entry point gets console logging for free, with no setup call needed.
_logger = logging.getLogger('ragdoll')
_logger.setLevel(logging.DEBUG)
if not _logger.handlers:
    _console_handler = logging.StreamHandler()
    _console_handler.setLevel(logging.INFO)
    _console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    _logger.addHandler(_console_handler)
