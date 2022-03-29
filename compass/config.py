from configparser import RawConfigParser, ConfigParser, ExtendedInterpolation
import os
from importlib import resources
import inspect
import sys


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
        self._comments = dict()
        self._combined = None
        self._combined_comments = None
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

    def set(self, section, option, value=None, comment=''):
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

        comment : str, optional
            A comment to include with the config option when it is written
            to a file
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
        self._combined_comments = None
        self._sources = None
        if filename not in self._comments:
            self._comments[filename] = dict()
        self._comments[filename][(section, option)] = comment

    def write(self, fp, include_sources=True, include_comments=True):
        """
        Write the config options to the given file pointer.

        Parameters
        ----------
        fp : typing.TestIO
            The file pointer to write to.

        include_sources : bool, optional
            Whether to include a comment above each option indicating the
            source file where it was defined

        include_comments : bool, optional
            Whether to include the original comments associated with each
            section or option
        """
        if self._combined is None:
            self._combine()
        for section in self._combined.sections():
            section_items = self._combined.items(section=section)
            if include_comments and section in self._combined_comments:
                fp.write(self._combined_comments[section])
            fp.write(f'[{section}]\n\n')
            for option, value in section_items:
                if include_comments:
                    fp.write(self._combined_comments[(section, option)])
                if include_sources:
                    source = self._sources[(section, option)]
                    fp.write(f'# source: {source}\n')
                value = str(value).replace('\n', '\n\t')
                fp.write(f'{option} = {value}\n\n')
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
        with open(filename) as fp:
            comments = self._parse_comments(fp, filename, comments_before=True)

        if user:
            self._user_config = {filename: config}
        else:
            self._configs[filename] = config
        self._comments[filename] = comments
        self._combined = None
        self._combined_comments = None
        self._sources = None

    def _combine(self):
        self._combined = ConfigParser(interpolation=ExtendedInterpolation())
        configs = dict(self._configs)
        configs.update(self._user_config)
        self._sources = dict()
        self._combined_comments = dict()
        for source, config in configs.items():
            for section in config.sections():
                if section in self._comments[source]:
                    self._combined_comments[section] = \
                        self._comments[source][section]
                if not self._combined.has_section(section):
                    self._combined.add_section(section)
                for option, value in config.items(section):
                    self._sources[(section, option)] = source
                    self._combined.set(section, option, value)
                    self._combined_comments[(section, option)] = \
                        self._comments[source][(section, option)]
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

    @staticmethod
    def _parse_comments(fp, filename, comments_before=True):
        """ Parse the comments in a config file into a dictionary """
        comments = dict()
        current_comment = ''
        section_name = None
        option_name = None
        indent_level = 0
        for line_number, line in enumerate(fp, start=1):
            value = line.strip()
            is_comment = value.startswith('#')
            if is_comment:
                current_comment = current_comment + line
            if len(value) == 0 or is_comment:
                # end of value
                indent_level = sys.maxsize
                continue

            cur_indent_level = len(line) - len(line.lstrip())
            is_continuation = cur_indent_level > indent_level
            # a section header or option header?
            if section_name is None or option_name is None or \
                    not is_continuation:
                indent_level = cur_indent_level
                # is it a section header?
                is_section = value.startswith('[') and value.endswith(']')
                if is_section:
                    if not comments_before:
                        if option_name is None:
                            comments[section_name] = current_comment
                        else:
                            comments[(section_name, option_name)] = \
                                current_comment
                    section_name = value[1:-1].strip().lower()
                    option_name = None

                    if comments_before:
                        comments[section_name] = current_comment
                    current_comment = ''
                # an option line?
                else:
                    delimiter_index = value.find('=')
                    if delimiter_index == -1:
                        raise ValueError(f'Expected to find "=" on line '
                                         f'{line_number} of {filename}')

                    if not comments_before:
                        if option_name is None:
                            comments[section_name] = current_comment
                        else:
                            comments[(section_name, option_name)] = \
                                current_comment

                    option_name = value[:delimiter_index].strip().lower()

                    if comments_before:
                        comments[(section_name, option_name)] = current_comment
                    current_comment = ''

        return comments
