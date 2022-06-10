.. _dev_landice:

Landice core
============

The ``landice`` core is defined by the :py:class:`compass.landice.LandIce`
class. All test cases in the ``landice`` core share the following set of
default config options:

.. code-block:: cfg

    # This config file has default config options for the landice core

    # The paths section points compass to external paths
    [paths]

    # the relative or absolute path to the root of a branch where MPAS-Ocean
    # has been built
    mpas_model = MALI-Dev/components/mpas-albany-landice

    # The namelists section defines paths to example_compact namelists that will be used
    # to generate specific namelists. By default, these point to the forward and
    # init namelists in the default_inputs directory after a successful build of
    # the landice model.  Change these in a custom config file if you need a different
    # example_compact.
    [namelists]
    forward = ${paths:mpas_model}/default_inputs/namelist.landice

    # The streams section defines paths to example_compact streams files that will be used
    # to generate specific streams files. By default, these point to the forward and
    # init streams files in the default_inputs directory after a successful build of
    # the landice model. Change these in a custom config file if you need a different
    # example_compact.
    [streams]
    forward = ${paths:mpas_model}/default_inputs/streams.landice


    # The executables section defines paths to required executables. These
    # executables are provided for use by specific test cases.  Most tools that
    # compass needs should be in the conda environment, so this is only the path
    # to the MPAS-Ocean executable by default.
    [executables]
    model = ${paths:mpas_model}/landice_model


    # Options related to downloading files
    [download]

    # the path on the server for MALI
    core_path = mpas-albany-landice

The default location for MALI is in the
`git submodule <https://git-scm.com/book/en/v2/Git-Tools-Submodules>`_
``MALI-Dev`` in the directory ``components/mpas-albany-landice``.  The
``MALI-Dev`` submodule may not point to the latest update of the
`develop <https://github.com/MALI-Dev/E3SM/tree/develop>`_ branch but the plan
is to update the submodule frequently.  The current version of the submodule
should always be guaranteed to be compatible with the corresponding version of
``compass``.

To make sure the code in the submodule has been cloned and is up-to-date, you
should run

.. code-block:: bash

    git submodule update --init --recursive

in the base directory of your local clone of the compass repo.  Then, you can
``cd`` into ``MALI-Dev/components/mpas-albany-landice`` and build the code as
appropriate for whichever of the :ref:`machines` you are using.


.. toctree::
   :titlesonly:

   test_groups/index
   framework
