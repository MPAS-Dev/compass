#!/usr/bin/env python

import os
import re
from setuptools import setup, find_packages


def package_files(directory, prefixes, extensions):
    """ based on https://stackoverflow.com/a/36693250/7728169"""
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            parts = filename.split('.')
            prefix = parts[0]
            extension = parts[-1]
            if prefix in prefixes or extension in extensions:
                paths.append(os.path.join('..', path, filename))
    return paths


install_requires = \
    ['cartopy',
     'cmocean',
     'ipython',
     'jigsawpy==0.3.3',
     'jupyter',
     'lxml',
     'matplotlib',
     'netcdf4',
     'numpy',
     'progressbar2',
     'pyamg',
     'pyproj',
     'requests',
     'scipy>=1.8.0',
     'shapely',
     'xarray']

here = os.path.abspath(os.path.dirname(__file__))
version_path = os.path.join(here, 'compass', 'version.py')
with open(version_path) as f:
    main_ns = {}
    exec(f.read(), main_ns)
    version = main_ns['__version__']

os.chdir(here)

data_files = package_files('compass',
                           prefixes=['namelist', 'streams', 'README'],
                           extensions=['cfg', 'template', 'json', 'txt',
                                       'geojson', 'mat', 'nml'])

setup(name='compass',
      version=version,
      description='Configuration Of Model for Prediction Across Scales '
                  'Setups (COMPASS) is an automated system to set up test '
                  'cases that match the MPAS-Model repository. All '
                  'namelists and streams files begin with the default '
                  'generated from the Registry.xml file, and only the '
                  'changes relevant to the particular test case are altered '
                  'in those files.',
      url='https://github.com/MPAS-Dev/MPAS-Tools',
      author='COMPASS Developers',
      author_email='mpas-developers@googlegroups.com',
      license='BSD',
      classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          'Topic :: Scientific/Engineering',
      ],
      packages=find_packages(include=['compass', 'compass.*']),
      package_data={'': data_files},
      install_requires=install_requires,
      entry_points={'console_scripts':
                    ['compass = compass.__main__:main',
                     'create_compass_load_script=compass.load:main']})
