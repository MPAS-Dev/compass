.. _dev_landice:

Landice core
============

The ``landice`` core is defined by the :py:class:`compass.landice.Landice`
class. All test cases in the ``landice`` core share the following set of
default config options:

.. code-block:: cfg

    # This config file has default config options for the landice core

    # The paths section points compass to external paths
    [paths]

    # the relative or absolute path to the root of a branch where MPAS-Ocean
    # has been built
    mpas_model = MALI-Dev/components/mpas-albany-landice

    # The namelists section defines paths to default namelist templates used
    # to generate case-specific namelists.
    [namelists]
    forward = ${paths:mpas_model}/default_inputs/namelist.landice

    # The streams section defines paths to default stream templates used to
    # generate case-specific streams files.
    [streams]
    forward = ${paths:mpas_model}/default_inputs/streams.landice


    # The executables section defines paths to required executables. These
    # executables are provided for use by specific test cases.  Most tools that
    # compass needs should be in the deployment environment, so this is only
    # the path to the MALI executable by default.
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
