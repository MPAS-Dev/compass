.. _dev_tutorial_add_param_study:

Developer Tutorial: Adding a parameter study
============================================

This tutorial presents a step-by-step guide to adding a parameter study, a
test case in which different steps run MPAS cores with different parameter
values, typically followed by an analysis step that compares the results as
functions of the parameter or parameters (see the :ref:`glossary` for
definitions of these terms).  Parameter studies differ from other test cases
in that a user will often wish to modify the list of parameters that are being
varied by setting config options *before* setting up the test case.  This is
in contrast to most config options, which can be modified either before or
after setting up the test case.  The reason for this difference is because the
steps that will be set up depend on the parameter values.

In this tutorial, I will use the :ref:`dev_ocean_global_convergence_cosine_bell`
test case as an example. Although this test case was originally developed for
:ref:`legacy_compass` and ported using an approach similar to
:ref:`dev_tutorial_porting_legacy`, we will describe it as if it were being
created from scratch.  In this example, the parameter that we are studying is
the mesh resolution for a quasi-uniform (QU) mesh.  This type of parameter
study is called a convergence study because we are analyzing how rapidly the
error in the solution converges to zero as the resolution increases.

Many of the details of creating a parameter study are similar to creating any
other test case within a test group.  Please refer to the companion tutorial
:ref:`dev_tutorial_add_test_group`, which will be referenced liberally
in this tutorial. Here, we will focus almost entirely the process that is
specific to a parameter study.

Getting started
---------------

Please see :ref:`dev_tutorial_add_test_group_getting_started` for the tutorial
on adding a new test group.  The procedure is the same for this tutorial except
that the example branch name will be `add_cosine_bell` instead of
`add_baroclinic_channel`.

Making a new test group and "cosine_bell" test case
---------------------------------------------------

If your parameter study fits well in an existing test group, you don't need
to create a new one.  If the existing test groups aren't a good fit, you will
want to follow the step for :ref:`dev_tutorial_add_test_group_make_test_group`
first, then continue here to add the new test case.  In this example, the
test group is ``global_convergence``.

Within the test group, we will create a new test case in a python package
called ``cosine_bell`` and with a class called ``CosineBell``.  We will follow
the procedure described in :ref:`dev_tutorial_add_test_group_add_default` to
get started.


Adding "mesh", "init" and "forward" steps
-----------------------------------------

Our test case will be made up of 3 steps at each resolution, followed a final
``analysis`` step that combines data from each mesh resolution.  The 3 steps
for each mesh resolution are: ``mesh``, which creates a horizontal mesh of a
given QU resolution; ``init``, which creates a vertical coordinate and an
initial condition on the given mesh; and ``forward``, which runs a simulation
forward in time.

For each step, we create a python module containing a class with the name of
the step converted to camel case (e.g. a ``Mesh`` class for the ``mesh`` step).
The details of these modules are not critical for this tutorial but you are
welcome to take a closer look:
`mesh.py <https://github.com/MPAS-Dev/compass/blob/main/compass/ocean/tests/global_convergence/cosine_bell/mesh.py>`_,
`init.py <https://github.com/MPAS-Dev/compass/blob/main/compass/ocean/tests/global_convergence/cosine_bell/init.py>`_,
and `forward.py <https://github.com/MPAS-Dev/compass/blob/main/compass/ocean/tests/global_convergence/cosine_bell/forward.py>`_.
One important detail is that each step take the resolution (i.e. the parameter
value) as an input and uses that value to give the step a unique name and
subdirectory within the test case.  For example, this is from the ``mesh``
step:

.. code-block:: python

    def __init__(self, test_case, resolution):
        """
        Create a new step
        Parameters
        ----------
        test_case : compass.ocean.tests.global_convergence.cosine_bell.CosineBell
            The test case this step belongs to
        resolution : int
            The resolution of the (uniform) mesh in km
        """
        super().__init__(test_case=test_case,
                         name='QU{}_mesh'.format(resolution),
                         subdir='QU{}/mesh'.format(resolution))

This is a general requirement of test cases that support parameter studies.
Any step or steps that are performed for each parameter value should have the
parameter value passed in as an argument to ``__init__()``, and use the
parameter value in some way to give the test case a unique name and
subdirectory.

Much of the rest of the details of creating these steps is similar to the
description in :ref:`dev_tutorial_add_test_group`, so I refer you to that
tutorial for more details.

Adding an "analysis" step
-------------------------

Many parameter studies will perform some kind of analysis that brings together
output from runs with different parameter values.  In our example, the
``analysis`` step is used to plot the error as a function of resolution.  This
requires using output from all of the ``init`` and ``forward`` steps at
different resolutions.  ``analysis`` differs from other steps in this test case
in that it takes all parameter values (in this case resolutions) as an input:

.. code-block:: python

    def __init__(self, test_case, resolutions):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_convergence.cosine_bell.CosineBell
            The test case this step belongs to

        resolutions : list of int
            The resolutions of the meshes that have been run
        """
        super().__init__(test_case=test_case, name='analysis')
        self.resolutions = resolutions

        for resolution in resolutions:
            self.add_input_file(
                filename='QU{}_namelist.ocean'.format(resolution),
                target='../QU{}/init/namelist.ocean'.format(resolution))
            self.add_input_file(
                filename='QU{}_init.nc'.format(resolution),
                target='../QU{}/init/initial_state.nc'.format(resolution))
            self.add_input_file(
                filename='QU{}_output.nc'.format(resolution),
                target='../QU{}/forward/output.nc'.format(resolution))

        ...

The remaining details of the analysis step are specific to this particular test
case so we won't go into them in this tutorial.  But feel free to have a look:
`analysis.py <https://github.com/MPAS-Dev/compass/blob/main/compass/ocean/tests/global_convergence/cosine_bell/analysis.py>`_.

Adding the steps to the test case
---------------------------------

The initial design for ``cosine_bell`` was that steps were only added to the
test case in the ``configure()`` method, which gets run when the test case gets
set up.  This design was because the particular parameter values that will be
used (the resolutions of the meshes) wasn't known at initialization, only
during setup.  A user could provide a custom config file with their own choice
of resolutions as part of setting up the test case.

However, it became clear that it wasn't possible to list the steps of the
``cosine_bell`` test case using ``compass list --verbose``.  Before test cases
are listed, the ``__init__()`` method has been called but the ``configure()``
method has not.  So the test case didn't have any steps added to it yet.  This
was confusing to developers.

The solution we decided on was to set up the steps in the test case with the
default parameters in ``__init__()``.  Then, in ``configure()``, we check if
the parameter values have been changed from the defaults.  If so, we remove the
old steps and add new ones with the new parameter values.  To do this, we use
a "private" method ``_setup_steps()``:

.. code-block:: python

    def _setup_steps(self, config):
        """ setup steps given resolutions """
        resolutions = config.get('cosine_bell', 'resolutions')
        resolutions = [int(resolution) for resolution in
                       resolutions.replace(',', ' ').split()]

        if self.resolutions is not None and self.resolutions == resolutions:
            return

        # start fresh with no steps
        self.steps = dict()
        self.steps_to_run = list()

        self.resolutions = resolutions

        for resolution in resolutions:
            self.add_step(Mesh(test_case=self, resolution=resolution))

            self.add_step(Init(test_case=self, resolution=resolution))

            self.add_step(Forward(test_case=self, resolution=resolution))

        self.add_step(Analysis(test_case=self, resolutions=resolutions))

The resolutions are parsed from the config options.  Then, if we either haven't
previously stored resolutions (i.e. we're in ``__init__()``) or we have
previous resolutions but they're different from the ones from the config
options, we start over, adding steps for the given resolutions.

Here's how this function is called from ``__init__()`` and ``configure()``:

.. code-block:: python

    def __init__(self, test_group):
        """
        Create test case for creating a global MPAS-Ocean mesh

        Parameters
        ----------
        test_group : compass.ocean.tests.cosine_bell.GlobalOcean
            The global ocean test group that this test case belongs to
        """
        super().__init__(test_group=test_group, name='cosine_bell')
        self.resolutions = None

        # add the steps with default resolutions so they can be listed
        config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())
        add_config(config, self.__module__, '{}.cfg'.format(self.name))
        self._setup_steps(config)

    def configure(self):
        """
        Set config options for the test case
        """
        config = self.config
        # set up the steps again in case a user has provided new resolutions
        self._setup_steps(config)

        ...

During ``__init__()``, the test case doesn't have any config options yet, since
these are typically only parsed when the test case is being set up (just before
``configure()`` gets called.  So the test case has to parse the default config
file for the test case manually itself and pass the config options to the
``_setup_steps()`` method.  This is a little clumsy and more time consuming
than steps we typically like to include in ``__init__()`` (because this method
gets called for every single MPAS core, test group, test case and step each
time you call any ``compass`` command-line tool).  But this seems the only
reasonable way to set up steps with the default parameter values during setup.

It is likely that other test cases supporting parameter studies will want to
mimic this behavior so that the default steps can be listed with
``compass list --verbose`` as well.

Setting the number of tasks and CPUs per task
---------------------------------------------

For some parameter studies, particularly those where resolution is the
parameter, it can be important to specify the target and minimum number of
MPI tasks for a given step as a function of the parameter.  If a given step
runs with python threading or multiprocessing instead, the number of CPUs per
task will, instead, be the important parameter (the number of tasks,
``ntasks``, is always 1).

In the following example, we set both the attribute of the steps
``step.ntasks`` and a config option (``QU<res>_ntasks``, where ``<res>`` is the
resolution) to a target number of tasks that is a heuristic function of the
resolution. Similarly, we set the minimum number of tasks (below which the step
will refuse to run) based on another heuristic function.

.. code-block:: python

    def update_cores(self):
        """ Update the number of cores and min_tasks for each forward step """

        config = self.config

        goal_cells_per_core = config.getfloat('cosine_bell',
                                              'goal_cells_per_core')
        max_cells_per_core = config.getfloat('cosine_bell',
                                             'max_cells_per_core')

        for resolution in self.resolutions:
            # a heuristic based on QU30 (65275 cells) and QU240 (10383 cells)
            approx_cells = 6e8 / resolution**2
            # ideally, about 300 cells per core
            # (make it a multiple of 4 because...it looks better?)
            ntasks = max(1,
                        4*round(approx_cells / (4 * goal_cells_per_core)))
            # In a pinch, about 3000 cells per core
            min_tasks = max(1,
                            round(approx_cells / max_cells_per_core))
            step = self.steps[f'QU{resolution}_forward']
            step.ntasks = ntasks
            step.min_tasks = min_tasks

            config.set('cosine_bell', f'QU{resolution}_ntasks', str(ntasks),
                       comment=f'Target core count for {resolution} km mesh')
            config.set('cosine_bell', f'QU{resolution}_min_tasks',
                       str(min_tasks),
                       comment=f'Minimum core count for {resolution} km mesh')

This method is called in the ``configure()`` method of the test case when it
is getting set up.  It is important to set the ``ntasks`` and ``min_tasks``
attributes of the step because this will be used as part of determining how
many cores are needed for a test suite using this test case.

Later on, when the test case gets run, we want to use the config options again
to set ``step.ntasks`` and ``step.min_tasks``, in case a user has modified
these config options before running the test case.  We do this before we run
the steps.

.. code-block:: python


    def run(self):
        """
        Run each step of the testcase
        """
        config = self.config
        for resolution in self.resolutions:
            ntasks = config.getint('cosine_bell', f'QU{resolution}_ntasks')
            min_tasks = config.getint('cosine_bell',
                                      f'QU{resolution}_min_tasks')
            step = self.steps[f'QU{resolution}_forward']
            step.ntasks = ntasks
            step.min_tasks = min_tasks

        # run the step
        super().run()

Documentation
-------------

Please document the test case within its test group as described in the
companion tutorial in its :ref:`dev_tutorial_add_test_group_docs` section.
