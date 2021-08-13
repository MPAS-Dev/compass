.. _design_doc_cached_outputs:

Caching outputs from compass steps
==================================

Date: 2021/07/30

Contributors: Xylar Asay-Davis

Summary
-------

We would like to have a way to download output files for ``compass`` steps from
an online cache instead of generating them each time the step runs.  The
primary motivation for this is to optionally avoid time-consuming steps for
generating meshes and initial conditions for faster regression testing with
MPAS components in "forward" mode.  Potential other uses could include cached
results as baselines for validation.  A challenge for this capability is
providing an easy way for both developers and users to control which steps in a
test case or suite are cached and which are run as normal.


Requirements
------------

.. _req_cached:

Requirement: cached outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/07/30

Contributors: Xylar Asay-Davis

Each ``compass`` step defines its output files in the ``compass.Step.outputs``
attribute. For selected steps (see :ref:`req_select`), we require a mechanism
to download cached files for each of these outputs and to use these cached
files for the outputs of the step instead of computing them.  

.. _req_select:

Requirement: selecting whether to use cached outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/07/30

Contributors: Xylar Asay-Davis

There needs to be a mechanism for developers and users to select which steps
are run as normal and which use cached outputs.  For this mechanism to be
practical, it should not be overly tedious or manual (e.g. manually setting a 
flag for each step).

.. _req_update:

Requirement: updating cached outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/07/30

Contributors: Xylar Asay-Davis

There should be a documented process for creating cached outputs for steps and
uploading them.

.. _req_unique:

Requirement: unique identifier for cached outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/07/30

Contributors: Xylar Asay-Davis

There should be a mechanism for giving each cached output file a unique 
identifier (such as a date stamp).  A given version (git hash or release) of 
``compass`` should know which cached files to download.  Older cached files
should be retained so that older versions of ``compass`` can still be used
with these cached files.  

.. note::

    It may be worthwhile to include a process for deprecating and then deleting
    old cache files.

.. _req_normal_or_cached:

Requirement: either "normal" or "cached" versions of a step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/07/30

Contributors: Xylar Asay-Davis

We **do not** require the ability to set up a "normal" and a "cached" version
of the same step within a ``compass`` test case or suite.  (If this is not the
case, it would place important constraints on the design solution.)


Design
------

.. _des_cached:

Design: cached outputs
^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/07/30

Contributors: Xylar Asay-Davis

``compass`` supports "databases" of input data files on the E3SM 
`LCRC server <https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata/>`_.
Files will be stored in a new ``compass_cache`` database within each MPAS 
core's space on that server.  If the "cached" version of a step is selected
(see :ref:`des_select`), an appropriate "input" file will be added to the test 
case where the "target" is the file on the LCRC server to be cached locally for
future use and the "filename" is the output file.  ``compass`` will know which
files on the server correspond to which output files via a python dictionary, 
as described in :ref:`des_unique`.

.. _des_select:

Design: selecting whether to use cached outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/08/03

Contributors: Xylar Asay-Davis


A ``compass`` suite can indicate cached steps in two ways.  If all steps in a
test case should have cached output, the following notation is used:

.. code-block:: none

    ocean/global_ocean/QU240/mesh
        cached
    ocean/global_ocean/QU240/PHC/init
        cached

If only some steps in a test case should have cached output, they need to be
listed explicitly, as follows:

.. code-block:: none

    ocean/global_ocean/QU240/mesh
        cached: mesh
    ocean/global_ocean/QU240/PHC/init
        cached: initial_state

Similarly, a user setting up test cases has two mechanisms for specifying which
test cases and steps should have cached outputs.  If all steps in a test case
should have cached outputs, the suffix ``c`` can be added to the test number:

.. code-block:: none

    compass setup -n 90c 91c 92 ...

This approach is efficient but does not provide any control of which steps use
cached outputs and which do not.

A much more verbose approach is required if some steps use cached outputs and
others do not within a given test case.  Each test case must be set up on its
own with the ``-t`` and ``--cached`` flags as follows:

.. code-block:: none

    compass setup -t ocean/global_ocean/QU240/mesh --cached mesh ...
    compass setup -t ocean/global_ocean/QU240/PHC/init --cached initial_state ...
    ...

These approaches assume that we always have either the "normal" or the "cached"
version of a step within a test case or test suite (see
:ref:`des_normal_or_cached`) and developers or users are free to choose between
them, as long as cache files have been stored on the LCRC server and added to
the ``cached_files.json`` database.

.. _des_update:

Design: updating cached outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/08/03

Contributors: Xylar Asay-Davis

A new ``compass cache`` command-line tool will be added.  This will only be
available on Chrysalis and Anvil, the machines where files can be placed on the
LCRC server.  This command can be run on a work directory to copy the outputs
from selected steps into the appropriate directory on the LCRC server, and to
create or update a python dictionary in a file ``cached_files.json`` (see
:ref:`des_unique`) that maps between output files in the work directory and
those on the LCRC server.  For example:

.. code-block:: bash

    compass cache -i \
        ocean/global_ocean/QU240/mesh/mesh \
        ocean/global_ocean/QU240/PHC/init/initial_state

.. _des_unique:

Design: unique identifier for cached outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/08/03

Contributors: Xylar Asay-Davis

Each cached file on the LCRC server will include a date stamp in the file name.
For example, ``culled_mesh.nc`` will become ``culled_mesh.20210730.nc`` on the
server.  When ``compass cache`` is called (see :ref:`des_update`), the date
stamp will default to the date that the call is being made but can be
overridden with a flag (e.g. ``--date 20210730``).

Each MPAS core in ``compass`` will optionally include a file
``cached_files.json`` that contains a python dictionary mapping between the
names of output files in the work directory and those in the ``compass_cache``
database for that MPAS core on the LCRC server.  For example:

.. code-block:: none

    {
        "ocean/global_ocean/QU240/mesh/mesh/culled_mesh.nc": "global_ocean/QU240/mesh/mesh/culled_mesh.210803.nc",
        "ocean/global_ocean/QU240/mesh/mesh/culled_graph.info": "global_ocean/QU240/mesh/mesh/culled_graph.210803.info",
        "ocean/global_ocean/QU240/mesh/mesh/critical_passages_mask_final.nc": "global_ocean/QU240/mesh/mesh/critical_passages_mask_final.210803.nc",
        "ocean/global_ocean/QU240/PHC/init/initial_state/initial_state.nc": "global_ocean/QU240/PHC/init/initial_state/initial_state.210803.nc",
        "ocean/global_ocean/QU240/PHC/init/initial_state/init_mode_forcing_data.nc": "global_ocean/QU240/PHC/init/initial_state/init_mode_forcing_data.210803.nc"
    }

.. _des_normal_or_cached:

Design: either "normal" or "cached" versions of a step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/07/30

Contributors: Xylar Asay-Davis

A prototype implementation of output caching had separate versions of test
cases that included cached outputs or depended on earlier test cases with
cached outputs.  This approach turned out to be very cumbersome.  It added
many "new" test cases with unique subdirectories in the work directory and
required predetermining which steps should allow caching.  But this approach
*did* allow a test suite to include a "normal" version of a step and a "cached"
version of that same step in the same work directory (and therefore in the same
test suite).

The proposed design, described in the previous sections, would allow far more
flexibility about which steps are cached and which are not.  It is not clear
to me how we achieve this flexibility without requiring that a given step
either be set up as "normal" or "cached", and not both in the same work
directory.

Implementation
--------------

The implementation is on
`this branch <https://github.com/xylar/compass/tree/cached_init>`_.

.. _imp_cached:

Implementation: cached outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/08/04

Contributors: Xylar Asay-Davis

Each step has a boolean attribute ``cached`` that defaults to ``False`` but
which can be set to ``True`` by a process described in :ref:`imp_select`.  If
``cached == True``, when inputs and outputs are being processes, the usual
inputs are ignored and instead the outputs are added as inputs.  Targets in the
``compass_cache`` database are selected using the dictionary stored in the
MPAS core's ``cached_files.json``.  Namelists and steams files are also not
generated.

.. _imp_select:

Implementation: selecting whether to use cached outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/08/04

Contributors: Xylar Asay-Davis

The implementation includes the two mechanisms for selecting cached outputs
described in :ref:`des_select`.

When setting up a test suites, a new list of lists called ``cached`` is created
along with the list of test-case paths.  By default, all test cases have an
empty list of steps with cached outputs.  Any line in a test suite file that is
``cached`` (once white space is stripped away) will indicate that all steps in
that test case should use cached outputs.  This is accomplished by adding a
special "step" named ``_all`` as the first step in the list for the given test
case.  If a line of the test suite file starts with ``cached:`` (after
stripping away white space), the remainder of the line is a space-separated
list of step names that should be set up with cached outputs.  These steps
are appended to the list of cached steps for the test case.  If a test case has
many steps with cached outputs, it may be convenient to have multiple lines
starting with ``cached:``, as in this example.

.. code-block:: none

    ocean/global_convergence/cosine_bell
      cached: QU60_mesh QU60_init QU90_mesh QU90_init QU120_mesh QU120_init
      cached: QU150_mesh QU150_init QU180_mesh QU180_init QU210_mesh QU210_init
      cached: QU240_mesh QU240_init

If a user is setting up individual test cases, they can indicate that all the
steps in a test case should have cached inputs with the suffix ``c`` after the
test number.  While there is also a flag ``--cached`` that can be used to list
steps of a single test case to use from cached outputs, this feature is likely
to be too cumbersome to be broadly useful.  Instead, developers should probably
create a test suite for test cases where users are likely to want some steps
with and others without cached outputs, as in the Cosine Bell example above.

.. _imp_update:

Implementation: updating cached outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/08/04

Contributors: Xylar Asay-Davis

The new ``compass cache`` command has been added and is defined in the
``compass.cache`` module.  It takes a list of step paths as input and optional
flags ``--dry_run`` (which doesn't copy the files to the directory on the LCRC
server) and ``--date_string``, which lets a user supply a date stamp (YYMMDD)
other than today's date.

As stated in the design, the command is only available on Chrysalis and Anvil
and should be run on a work directory.  To support caching files from multiple
MPAS cores at the same time, ``compass cache`` produces an updated database
file ``<mpas_core>_cached_files.json`` in the base of the work directory where
the command is run.  If this file already exists before ``compass cache`` is
run, the information for the specified steps will be added if it is not yet
in the database or will be updated, e.g. with new date stamps, if it does
exist.  If no ``<mpas_core>_cached_files.json`` exists, the file
``cached_files.json`` from the python module ``compass.<mpas_core>`` is used as
the starting point instead.  If this file also doesn't exist, we start with an
empty dictionary.

As an example, yesterday (8/3/2021) when I made the following call:

.. code-block:: bash

    for mesh in QU60 QU90 QU120 QU150 QU180 QU210 QU240
    do
      for step in mesh init
      do
        compass cache -i ocean/global_convergence/cosine_bell/${mesh}/${step}
      done
    done

the result was a cache file ``ocean_cached_files.json`` like this:

.. code-block:: none

    {
        "ocean/global_convergence/cosine_bell/QU60/mesh/mesh.nc": "global_convergence/cosine_bell/QU60/mesh/mesh.210803.nc",
        "ocean/global_convergence/cosine_bell/QU60/mesh/graph.info": "global_convergence/cosine_bell/QU60/mesh/graph.210803.info",
        "ocean/global_convergence/cosine_bell/QU60/init/namelist.ocean": "global_convergence/cosine_bell/QU60/init/namelist.210803.ocean",
        "ocean/global_convergence/cosine_bell/QU60/init/initial_state.nc": "global_convergence/cosine_bell/QU60/init/initial_state.210803.nc",
        "ocean/global_convergence/cosine_bell/QU90/mesh/mesh.nc": "global_convergence/cosine_bell/QU90/mesh/mesh.210803.nc",
        "ocean/global_convergence/cosine_bell/QU90/mesh/graph.info": "global_convergence/cosine_bell/QU90/mesh/graph.210803.info",
        "ocean/global_convergence/cosine_bell/QU90/init/namelist.ocean": "global_convergence/cosine_bell/QU90/init/namelist.210803.ocean",
        "ocean/global_convergence/cosine_bell/QU90/init/initial_state.nc": "global_convergence/cosine_bell/QU90/init/initial_state.210803.nc",
        ...
    }

This file should be copied back to ``compass/ocean/cached_files.json`` in
a branch of the compass repo, committed to the branch, and updated on
``master`` with a pull request as normal.


.. _imp_unique:

Implementation: unique identifier for cached outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/08/04

Contributors: Xylar Asay-Davis

A date string is appended to the end of files in the ``compass_cache`` database
on LCRC and stored in ``cached_files.json``.  The date string defaults to the
date the ``compass cache`` command is run but can be specified manually with
the ``--date_string`` flag if desired.

.. _imp_normal_or_cached:

Implementation: either "normal" or "cached" versions of a step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/08/04

Contributors: Xylar Asay-Davis

The implementation leans heavily on the assumption that a given step will
either be run with cached outputs or as normal, so that both versions are not
available in the same work directory or as part of the same test suite.

Nevertheless, if a separate "cached" version of a step were desired, it would 
be necessary to make symlinks from the cached files in the location of the 
"uncached" version of the step to the location of the "cached" version.  For 
example, if the "uncached" step is

.. code-block:: none

    ocean/global_ocean/QU240/mesh/mesh

and the "cached" version of the step is

.. code-block:: none

    ocean/global_ocean/QU240/cached/mesh/mesh

symlinks could be created on the LCRC server, e.g.

.. code-block:: none

    /lcrc/group/e3sm/public_html/mpas_standalonedata/mpas-ocean/compass_cache/global_ocean/QU240/cached/mesh/mesh/culled_mesh.210803.nc 
      -> /lcrc/group/e3sm/public_html/mpas_standalonedata/mpas-ocean/compass_cache/global_ocean/QU240/mesh/mesh/culled_mesh.210803.nc

and the ``cached`` attribute could be set to ``True`` in the constructor of the
cached version of the step.

Testing
-------

.. _test_cached:

Testing: cached outputs
^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/08/04

Contributors: Xylar Asay-Davis

I have constructed cached versions of the following steps on the LCRC server,
using test-case runs on Chrysalis.

.. code-block:: none

    ocean/global_ocean/QU240/mesh/mesh/
    ocean/global_ocean/QU240/PHC/init/initial_state/
    ocean/global_ocean/QUwISC240/mesh/mesh/
    ocean/global_ocean/QUwISC240/PHC/init/initial_state/
    ocean/global_ocean/QUwISC240/PHC/init/ssh_adjustment/
    ocean/global_ocean/EC30to60/mesh/mesh/
    ocean/global_ocean/EC30to60/PHC/init/initial_state/
    ocean/global_ocean/WC14/mesh/mesh/
    ocean/global_ocean/WC14/PHC/init/initial_state/
    ocean/global_ocean/ECwISC30to60/mesh/mesh/
    ocean/global_ocean/ECwISC30to60/PHC/init/initial_state/
    ocean/global_ocean/ECwISC30to60/PHC/init/ssh_adjustment/
    ocean/global_ocean/SOwISC12to60/mesh/mesh/
    ocean/global_ocean/SOwISC12to60/PHC/init/initial_state/
    ocean/global_ocean/SOwISC12to60/PHC/init/ssh_adjustment/
    ocean/global_convergence/cosine_bell/QU60/mesh/
    ocean/global_convergence/cosine_bell/QU60/init/
    ocean/global_convergence/cosine_bell/QU90/mesh/
    ocean/global_convergence/cosine_bell/QU90/init/
    ocean/global_convergence/cosine_bell/QU120/mesh/
    ocean/global_convergence/cosine_bell/QU120/init/
    ocean/global_convergence/cosine_bell/QU180/mesh/
    ocean/global_convergence/cosine_bell/QU180/init/
    ocean/global_convergence/cosine_bell/QU210/mesh/
    ocean/global_convergence/cosine_bell/QU210/init/
    ocean/global_convergence/cosine_bell/QU240/mesh/
    ocean/global_convergence/cosine_bell/QU240/init/
    ocean/global_convergence/cosine_bell/QU150/mesh/
    ocean/global_convergence/cosine_bell/QU150/init/

I have set up and run versions of all these steps with cached outputs, together
with forward runs (``performance_test`` in the global ocean test group, and
``forward`` steps in the ``cosine_bell`` test case)  that make use of the
cached outputs as inputs.  All tests ran successfully and were bit-for-bit with
a baseline that was used to produce the cached outputs.

.. _test_select:

Testing: selecting whether to use cached outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/08/04

Contributors: Xylar Asay-Davis

I added QUwISC240 test case to the ocean ``nightly`` test suite using cached
outputs for the ``mesh`` and ``init`` test cases:

.. code-block:: none

    ocean/global_ocean/QUwISC240/mesh
      cached
    ocean/global_ocean/QUwISC240/PHC/init
      cached
    ocean/global_ocean/QUwISC240/PHC/performance_test

I created a new test suite, ``cosine_bell_cached_init``, for the
``cosine_bell`` test case that uses cached outputs fro the ``mesh`` and
``init`` steps at each default mesh resolution:

.. code-block:: none

    ocean/global_convergence/cosine_bell
      cached: QU60_mesh QU60_init QU90_mesh QU90_init QU120_mesh QU120_init
      cached: QU150_mesh QU150_init QU180_mesh QU180_init QU210_mesh QU210_init
      cached: QU240_mesh QU240_init

I set up the remaining steps with cached outputs mentioned in
:ref:`test_cached` as follows:

.. code-block:: bash

    compass list

    compass setup -n 40c 41c 42 60c 61c 62 80c 81c 82 85c 86c 87 90c 91c 92 \
        95c 96c 97 ...

Results were bit-for-bit with the same test cases run without cached outputs.

.. _test_update:

Testing: updating cached outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/08/04

Contributors: Xylar Asay-Davis

All cached files used in the testing above sere created with ``compass cache``
on Chrysalis.  Multiple runs of this command created, then updated the local
``ocean_cached_files.json``, as expected.  The files ended up in the expected
directories on the LCRC server with the expected date strings appended to the
file basename (before the extension).

The ``--dry_run`` feature also worked as expected, updating the
``ocean_cached_files.json`` without copying files.  The ``--date_string``
flag could be used to specify an alternative suffix, as expected.

.. _test_unique:

Testing: unique identifier for cached outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/08/04

Contributors: Xylar Asay-Davis

All files in the ``compass_cache`` database have date strings appended to them
to make them unique.  No testing has been performed yet to ensure that new
cached files with new dated can be added but I don't foresee any problems.

.. _test_normal_or_cached:

Testing: either "normal" or "cached" versions of a step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/08/04

Contributors: Xylar Asay-Davis

The implementation that I tested is based on this requrements.  However, in the
future, the requirement could be relaxed if need be using the approach I
outlined in :ref:`imp_normal_or_cached`.
