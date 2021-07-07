.. _dev_troubleshooting:

Troubleshooting
===============

This section describes some common problems developers have run into and some
suggested solutions.

.. _dev_troubleshooting_conda_solver:

Solver errors when configuring conda environment
------------------------------------------------

When setting up :ref:`dev_conda_env`, by calling:

.. code-block:: bash

    ./conda/configure_compass_env.sh ...

you may run into an error like:

.. code-block:: none

    Encountered problem while solving:
      - nothing provides geos 3.5.* needed by cartopy-0.14.3-np110py27_4

    ...

    subprocess.CalledProcessError: ...

Details of teh error may vary but the message indicates in some way that there
was a problem solving for the requested combination of packages.  This likely
indicates that you have an existing compass development environment
(``dev_compass_*``) that can't be updated to be compatible with the new
set of development packages given in one of:

.. code-block:: none

    conda/compass_env/spec-file*.txt

The solution should be to recreate the environment rather than trying to
update it:

.. code-block:: bash

    ./conda/configure_compass_env.sh --recreate ...
