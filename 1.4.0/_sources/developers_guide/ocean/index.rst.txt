.. _dev_ocean:

Ocean core
==========

The ``ocean`` core is defined by the :py:class:`compass.ocean.Ocean`
class. All test cases in the ``ocean`` core share the following set of
default config options:

.. code-block:: cfg

    # This config file has default config options for the ocean core

    # The paths section points compass to external paths
    [paths]

    # the relative or absolute path to the root of a branch where MPAS-Ocean
    # has been built
    mpas_model = E3SM-Project/components/mpas-ocean

    # The namelists section defines paths to example_compact namelists that will be used
    # to generate specific namelists. By default, these point to the forward and
    # init namelists in the default_inputs directory after a successful build of
    # the ocean model.  Change these in a custom config file if you need a different
    # example_compact.
    [namelists]
    forward = ${paths:mpas_model}/default_inputs/namelist.ocean.forward
    init    = ${paths:mpas_model}/default_inputs/namelist.ocean.init

    # The streams section defines paths to example_compact streams files that will be used
    # to generate specific streams files. By default, these point to the forward and
    # init streams files in the default_inputs directory after a successful build of
    # the ocean model. Change these in a custom config file if you need a different
    # example_compact.
    [streams]
    forward = ${paths:mpas_model}/default_inputs/streams.ocean.forward
    init    = ${paths:mpas_model}/default_inputs/streams.ocean.init


    # The executables section defines paths to required executables. These
    # executables are provided for use by specific test cases.  Most tools that
    # compass needs should be in the conda environment, so this is only the path
    # to the MPAS-Ocean executable by default.
    [executables]
    model = ${paths:mpas_model}/ocean_model


    # Options related to downloading files
    [download]

    # the path on the server for MPAS-Ocean
    core_path = mpas-ocean


    # Options relate to adjusting the sea-surface height or land-ice pressure
    # below ice shelves to they are dynamically consistent with one another
    [ssh_adjustment]

    # the number of iterations of ssh adjustment to perform
    iterations = 10

The default location for MPAS-Ocean is in the
`git submodule <https://git-scm.com/book/en/v2/Git-Tools-Submodules>`_
``E3SM-Project`` in the directory ``components/mpas-ocean``.  The submodule
may not point to the latest MPAS-Ocean code in on the E3SM
`main <https://github.com/E3SM-Project/E3SM/tree/main>`_
branch but the plan is to update the submodule frequently.  The current version
of the submodule should always be guaranteed to be compatible with the
corresponding version of ``compass``.

To make sure the code in the submodule has been cloned and is up-to-date, you
should run

.. code-block:: bash

    git submodule update --init --recursive

in the base directory of your local clone of the compass repo.  Then, you can
``cd`` into ``E3SM-Project/components/mpas-ocean`` and build the code as
appropriate for whichever of the :ref:`machines` you are using.

.. toctree::
   :titlesonly:

   test_groups/index
   framework
