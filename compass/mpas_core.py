from importlib import resources
import json


class MpasCore:
    """
    The base class for housing all the tests for a given MPAS core, such as
    ocean, landice or sw (shallow water)

    Attributes
    ----------
    name : str
        the name of the MPAS core

    test_groups : dict
        A dictionary of test groups for the MPAS core with their names as keys

    cached_files : dict
        A dictionary that maps from output file names in test cases to cached
        files in the ``compass_cache`` database for the MPAS core.  These
        file mappings are read in from ``cached_files.json`` in the MPAS core.
    """

    def __init__(self, name):
        """
        Create a new container for the test groups for a given MPAS core

        Parameters
        ----------
        name : str
            the name of the MPAS core
        """
        self.name = name

        # test groups are added with add_test_groups()
        self.test_groups = dict()

        self.cached_files = dict()
        self._read_cached_files()

    def add_test_group(self, test_group):
        """
        Add a test group to the MPAS core

        Parameters
        ----------
        test_group : compass.TestGroup
            the test group to add
        """
        self.test_groups[test_group.name] = test_group

    def _read_cached_files(self):
        """ Read in the dictionary of cached files from cached_files.json """

        package = f'compass.{self.name}'
        filename = 'cached_files.json'
        try:
            with resources.path(package, filename) as path:
                with open(path) as data_file:
                    self.cached_files = json.load(data_file)
        except FileNotFoundError:
            # no cached files for this core
            pass
