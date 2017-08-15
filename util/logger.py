#!/usr/bin/env python3
import logging
import os

logging.basicConfig(format='%(message)s', level=logging.DEBUG)


def get_logger():
    logger = logging.getLogger("default_logger")
    return logger

# TODO different formatting for levels ...
_logger = get_logger()
log = _logger.info
warn = _logger.warn
error = _logger.error
debug = _logger.debug


def setup_file_logger(logfile=None, append=True):
    if logfile is not None:
        if not logfile.endswith(".log"):
            logfile += ".log"

        log("Setting up file logging to: {}".format(logfile))
        logdir = os.path.dirname(logfile)
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        if append:
            file_mode = "a"
        else:
            file_mode = "w"
        handler = logging.FileHandler(logfile, mode=file_mode)
        logger = get_logger()
        logger.addHandler(handler)

