.. _dev_overview:

Overview
========

``compass`` is a `python package <https://docs.python.org/3/tutorial/modules.html#packages>`_.
All of the code in the package can be accessed in one of two ways.  The first
is the command-line interface with commands like ``compass list`` and
``compass setup``.  The second way is through import commands like:

.. code-block:: python

    from compass.io import symlink


    symlink('../initial_condition/initial_condition.nc', 'init.nc')

Some of the main advantages of ``compass`` being a package instead of a group
of scripts (as was the case for :ref:`legacy_compass`) is that:

1) it is a lot easier to share code;

2) there is no need to create symlinks to individual scripts or use
   `subprocess <https://docs.python.org/3/library/subprocess.html>`_ calls to
   run one python script from within another; and

3) functions within ``compass`` modules and subpackages have relatively simple
   interfaces that are easier to document and understand that the arguments
   passed in to a script.

