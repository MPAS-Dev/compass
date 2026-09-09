"""
Shared framework code for the ISMIP7 test groups.

The ISMIP7 test groups -- ``ismip7_forcing``, ``ismip7_run`` and
``ismip7_calibration`` -- all remap data from the ISMIP7 polar stereographic
grids onto a MALI mesh, and all need the same handful of helpers to do it.
Those helpers live here rather than in any one test group, so that using them
from another does not mean importing across test-group boundaries.
"""
