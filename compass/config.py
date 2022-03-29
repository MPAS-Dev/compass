from configparser import RawConfigParser, ConfigParser, ExtendedInterpolation
import os
from importlib import resources
import inspect


class CompassConfigParser:
    """
    A "meta" config parser that keeps a dictionary of config parsers and their
    sources to combine when needed.  The custom config parser allows provenance
    of the source of different config options and allows the "user" config
    options to always take precedence over other config options (even if they
    are added later).
    """
    def __init__(self):
        """
        Make a new (empty) config parser
        """

        self._configs = dict()
        self._user_config = dict()
        self._combined = None
        self._sources = None

    def add_user_config(self, filename):
        """
        Add a the contents of a user config file to the parser.  These options
        take precedence over all other options.

        Parameters
        ----------
        filename : str
            The relative or absolute path to the config file
        """
        self._add(filename, user=True)

    def add_from_file(self, filename):
        """
        Add the contents of a config file to the parser.

        Parameters
        ----------
        filename : str
            The relative or absolute path to the config file
        """
        self._add(filename, user=False)

    def add_from_package(self, package, config_filename, exception=True):
        """
        Add the contents of a config file to the parser.

        Parameters
        ----------
        package : str or Package
            The package where ``config_filename`` is found

        config_filename : str
            The name of the config file to add

        exception : bool, optional
            Whether to raise an exception if the config file isn't found
        """
        try:
            with resources.path(package, config_filename) as path:
                self._add(path, user=False)
        except (ModuleNotFoundError, FileNotFoundError, TypeError):
            if exception:
                raise

    def get(self, section, option):
        """
        Get an option value for a given section.

        Parameters
        ----------
        section : str
            The name of the config section

        option : str
            The name of the config option

        Returns
        -------
        value : str
            The value of the config option
        """
        if self._combined is None:
            self._combine()
        return self._combined.get(section, option)

    def getint(self, section, option):
        """
        Get an option integer value for a given section.

        Parameters
        ----------
        section : str
            The name of the config section

        option : str
            The name of the config option

        Returns
        -------
        value : int
            The value of the config option
        """
        if self._combined is None:
            self._combine()
        return self._combined.getint(section, option)

    def getfloat(self, section, option):
        """
        Get an option float value for a given section.

        Parameters
        ----------
        section : str
            The name of the config section

        option : str
            The name of the config option

        Returns
        -------
        value : float
            The value of the config option
        """
        if self._combined is None:
            self._combine()
        return self._combined.getfloat(section, option)

    def getboolean(self, section, option):
        """
        Get an option boolean value for a given section.

        Parameters
        ----------
        section : str
            The name of the config section

        option : str
            The name of the config option

        Returns
        -------
        value : bool
            The value of the config option
        """
        if self._combined is None:
            self._combine()
        return self._combined.getboolean(section, option)

    def getlist(self, section, option, dtype=str):
        """
        Get an option value as a list for a given section.

        Parameters
        ----------
        section : str
            The name of the config section

        option : str
            The name of the config option

        dtype : {Type[str], Type[int], Type[float]}
            The type of the elements in the list

        Returns
        -------
        value : list
            The value of the config option parsed into a list
        """
        values = self.get(section, option)
        values = [dtype(value) for value in values.replace(',', ' ').split()]
        return values

    def has_option(self, section, option):
        """
        Whether the given section has the given option

        Parameters
        ----------
        section : str
            The name of the config section

        option : str
            The name of the config option

        Returns
        -------
        found : bool
            Whether the option was found in the section
        """
        if self._combined is None:
            self._combine()
        return self._combined.has_option(section, option)

    def set(self, section, option, value=None):
        """
        Set the value of the given option in the given section.  The file from
         which this function was called is also retained for provenance.

        Parameters
        ----------
        section : str
            The name of the config section

        option : str
            The name of the config option

        value : str, optional
            The value to set the option to
        """
        calling_frame = inspect.stack(context=2)[1]
        filename = os.path.abspath(calling_frame.filename)
        if filename not in self._configs:
            self._configs[filename] = RawConfigParser()
        config = self._configs[filename]
        if not config.has_section(section):
            config.add_section(section)
        config.set(section, option, value)
        self._combined = None

    def write(self, fp, include_sources=True):
        """
        Write the config options to the given file pointer.

        Parameters
        ----------
        fp : typing.TestIO
            The file pointer to write to.

        include_sources : bool, optional
            Whether to include a comment above each option indicating the
            source file where it was defined
        """
        if self._combined is None:
            self._combine()
        for section in self._combined.sections():
            section_items = self._combined.items(section=section)
            fp.write(f'[{section}]\n')
            for key, value in section_items:
                if include_sources:
                    source = self._sources[(section, key)]
                    fp.write(f'# source: {source}\n')
                value = str(value).replace('\n', '\n\t')
                fp.write(f'{key} = {value}\n')
            fp.write('\n')

    def __getitem__(self, section):
        """
        Get get the config options for a given section.

        Parameters
        ----------
        section : str
            The name of the section to retrieve.

        Returns
        -------
        section_proxy : configparser.SectionProxy
            The config options for the given section.
        """
        if self._combined is None:
            self._combine()
        return self._combined[section]

    def _add(self, filename, user):
        filename = os.path.abspath(filename)
        config = RawConfigParser()
        if not os.path.exists(filename):
            raise FileNotFoundError(f'Config file does not exist: {filename}')
        config.read(filenames=filename)
        if user:
            self._user_config = {filename: config}
        else:
            self._configs[filename] = config
        self._combined = None

    def _combine(self):
        self._combined = ConfigParser(interpolation=ExtendedInterpolation())
        configs = dict(self._configs)
        configs.update(self._user_config)
        self._sources = dict()
        for source, config in configs.items():
            for section in config.sections():
                if not self._combined.has_section(section):
                    self._combined.add_section(section)
                for key, value in config.items(section):
                    self._sources[(section, key)] = source
                    self._combined.set(section, key, value)
        self._ensure_absolute_paths()

    def _ensure_absolute_paths(self):
        """
        make sure all paths in the paths, namelists, streams, and executables
        sections are absolute paths
        """
        config = self._combined
        for section in ['paths', 'namelists', 'streams', 'executables']:
            if not config.has_section(section):
                continue
            for option, value in config.items(section):
                value = os.path.abspath(value)
                config.set(section, option, value)
