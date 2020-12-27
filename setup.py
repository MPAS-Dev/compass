#!/usr/bin/env python

import os
import re
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'compass', '__init__.py')) as f:
    init_file = f.read()

version = re.search(r'{}\s*=\s*[(]([^)]*)[)]'.format('__version_info__'),
                    init_file).group(1).replace(', ', '.')

os.chdir(here)

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
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Topic :: Scientific/Engineering',
      ],
      packages=find_packages(exclude=['ci', 'deploy', 'docs', 'landice',
                                      'MPAS-Mode', 'ocean', 'recipe', 'test',
                                      'utility_scripts']),
      install_requires=['matplotlib',
                        'netCDF4',
                        'numpy',
                        'progressbar2',
                        'requests',
                        'scipy',
                        'xarray'],
      entry_points={'console_scripts':
                    ['compass = compass.__main__:main']})
