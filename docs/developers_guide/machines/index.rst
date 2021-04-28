.. _dev_machines:

Machines
========

One of the major advantages of ``compass`` over :ref:`legacy_compass` is that it
attempts to be aware of the capabilities of the machine it is running on.  This
is a particular advantage for so-called "supported" machines with a config file
defined for them in the ``compass`` package.  But even for "unknown" machines,
it is not difficult to set a few config options in your user config file to
describe your machine.  Then, ``compass`` can use this data to make sure test
cases are configured in a way that is appropriate for your machine.

.. _dev_supported_machines:

Supported Machines
------------------

We have activation scripts, typically for two compiler flavors, on each
supported machine.  The easiest way to load these for a developer is to run
the following in the root of the local clone of the compass repo:

.. code-block:: bash

    source ./load/load_compass_env.sh [-m <machine>] [-c <compiler>]

This will then source an appropriate script that will activate the compass
conda environment, load the compiler and MPI modules, and set environment
variables needed compile MPAS.

If you are on a login node, the script should automatically recognize what
machine you are on.  You can supply the machine name with ``-m <machine>`` if
you run into trouble with the automatic recognition (e.g. if you're setting
up test cases on a compute node).

If you do not supply a compiler set with ``-c``, you will get the default
compiler for each machine.  Currently, we only support one MPI flavor per
compiler, so you should not need to specify which MPI version to use.  This
follows automatically from the choice of compilers.

After loading this environment, you can set up test cases or test suites, and
a link ``load_compass_env.sh`` will be included in each suite or test case
work directory.  This is a link to a specific activation script for that
machine and compiler (so not a link to ``load/load_compass_env.sh``, it just
happens to be given the same name).  You can can source this file on a
compute node (e.g. in a job script) to get the right compass conda environment,
compilers, MPI libraries and environment variables for running MPAS.

Below are specifics for each supported machine
.. toctree::
   :titlesonly:

   anvil
   badger
   chrysalis
   compy
   cori
   grizzly


.. _dev_other_machines:

Other Machines
--------------

For a discussion on this, see the user's guid one :ref:`other_machines`.

