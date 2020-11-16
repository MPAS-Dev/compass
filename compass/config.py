import os
import configparser
from io import StringIO
from importlib import resources


def duplicate_config(config):
    """
    Make a deep copy of config to changes can be made without affecting the
    original

    Parameters
    ----------
    config : configparser.ConfigParser
        Configuration options

    Returns
    -------
    new_config : configparser.ConfigParser
        Deep copy of configuration options
    """

    config_string = StringIO()
    config.write(config_string)
    # We must reset the buffer to make it ready for reading.
    config_string.seek(0)
    new_config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation())
    new_config.read_file(config_string)
    return new_config


def add_config(config, package, config_file, exception=True):
    """
    Add the contents of a config file within a package to the current config
    parser

    Parameters
    ----------
    config : configparser.ConfigParser
        Configuration options

    package : str or Package
        The package where ``config_file`` is found

    config_file : str
        The name of the config file to add

    exception : bool
        Whether to raise an exception if the config file isn't found
    """
    try:
        with resources.path(package, config_file) as path:
            config.read(path)
    except (ModuleNotFoundError, FileNotFoundError):
        if exception:
            raise


def ensure_absolute_paths(config):
    """
    make sure all paths in the paths, namelists and streams sections are
    absolute paths

    Parameters
    ----------
    config : configparser.ConfigParser
        Configuration options
    """
    for section in ['paths', 'namelists', 'streams', 'executables']:
        for option, value in config.items(section):
            value = os.path.abspath(value)
            config.set(section, option, value)


def get_source_file(source_path, source, config):
    """
    Get an absolute path given a tag name for that path

    Parameters
    ----------
    source_path : str
        The keyword path for a path as defined in :ref:`compass_config`_,
        a config option from
        a relative or absolute directory for the source

    source : str
        The basename or relative path of the source within the ``source_path``
        directory

    config : configparser.ConfigParser
        Configuration options used to determine the the absolute paths for the
        given ``source_path``
    """

    if config.has_option('paths', source_path):
        source_path = config.get('paths', source_path)

    source_file = '{}/{}'.format(source_path, source)
    source_file = os.path.abspath(source_file)
    return source_file
