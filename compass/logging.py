import logging
import sys
import subprocess


def start(test_name, log_filename=None):
    """
    Start logging either to a log file or stdout/stderrs

    Parameters
    ----------
    test_name : str
        A unique name for the testcase or step to be used when creating a new
        logger

    log_filename : str, optional
        The name of a file where output should be written.  If none is
        supplied, output goes to stdout/stderr

    Returns
    -------
    logger : logging.Logger
        The logger for output

    handler : logging.Handler
        The handler for logging (e.g. a file or stdout/stderr)

    old_stdout : file
        The old sys.stdout, saved so it can be restored later by calling
        :py:func:`compass.logging.stop()`

    old_stderr : file
        The old sys.stderr, saved so it can be restored later by calling
        :py:func:`compass.logging.stop()`

    """

    if log_filename is not None:
        # redirect output to a log file
        logger = logging.getLogger(test_name)
        handler = logging.FileHandler(log_filename)
    else:
        logger = logging.getLogger(test_name)
        handler = logging.StreamHandler(sys.stdout)

    formatter = CompassFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if log_filename is not None:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)
    else:
        old_stdout = None
        old_stderr = None

    return logger, handler, old_stdout, old_stderr


def stop(logger, handler, old_stdout, old_stderr):
    """
    Stop the given logger and perform any necessary clean-up.  The input
    argument are the return values from :py:func:`compass.logging.start()`

    Parameters
    ----------
    logger : logging.Logger
        The logger we want to stop

    handler : logging.Handler
        The handler for logging (e.g. a file or stdout/stderr)

    old_stdout : file
        The old sys.stdout to be restored

    old_stderr : file
        The old sys.stderr to be restored
    """
    if old_stdout is not None:
        handler.close()
        # restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    # remove the handlers from the logger (probably only necessary if
    # writeLogFile==False)
    logger.handlers = []


class CompassFormatter(logging.Formatter):
    """
    A custom formatter for logging
    Modified from:
    https://stackoverflow.com/a/8349076/7728169
    https://stackoverflow.com/a/14859558/7728169
    """
    # Authors
    # -------
    # Xylar Asay-Davis

    # printing error messages without a prefix because they are sometimes
    # errors and sometimes only warnings sent to stderr
    dbg_fmt = "DEBUG: %(module)s: %(lineno)d: %(msg)s"
    info_fmt = "%(msg)s"
    err_fmt = info_fmt

    def __init__(self, fmt=info_fmt):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._fmt

        # Replace the original format with one customized by logging level
        if record.levelno == logging.DEBUG:
            self._fmt = CompassFormatter.dbg_fmt

        elif record.levelno == logging.INFO:
            self._fmt = CompassFormatter.info_fmt

        elif record.levelno == logging.ERROR:
            self._fmt = CompassFormatter.err_fmt

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._fmt = format_orig

        return result


def check_call(args, logger):
    """
    Call the given subprocess with logging to the given logger.

    Parameters
    ----------
    args : list
        A list of argument to the subprocess

    logger : logging.Logger
        The logger to write output to

    Raises
    ------
    subprocess.CalledProcessError
        If the given subprocess exists with nonzero status

    """

    process = subprocess.Popen(args, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if stdout:
        stdout = stdout.decode('utf-8')
        for line in stdout.split('\n'):
            logger.info(line)
    if stderr:
        stderr = stderr.decode('utf-8')
        for line in stderr.split('\n'):
            logger.error(line)

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode,
                                            ' '.join(args))


class StreamToLogger(object):
    """
    Modified based on code by:
    https://www.electricmonk.nl/log/2011/08/14/redirect-stdout-and-stderr-to-a-logger-in-python/
    Copyright (C) 2011 Ferry Boender
    License: GPL, see https://www.electricmonk.nl/log/posting-license/
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, str(line.rstrip()))

    def flush(self):
        pass
