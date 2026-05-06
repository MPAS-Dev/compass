.. _dev_docs:

Documentation
=============

The ``compass`` documentation is generated using the
`Sphinx <https://www.sphinx-doc.org/en/master/>`_ package and is written in
`reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_
format.  We recommend this `basic guide <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_
to reStructuredText in Sphinx.

Another easy way to get started is by taking a look at the existing source
code for the documentation: https://github.com/MPAS-Dev/compass/tree/main/docs/

Each time you add a core, configuration or test case, the corresponding
documentation must be included with the pull request to add the code.  This
includes documentation for both the User's Guide and the Developer's Guide.
For examples, see:

* :ref:`ocean` in the User's Guide

* :ref:`ocean_baroclinic_channel` configuration in the User's Guide

* :ref:`dev_ocean` in the Developer's Guide

* :ref:`dev_ocean_baroclinic_channel` configuration in the Developer's Guide

Documentation for each core in the User's guide should include a label with the
name of the core:

.. code-block:: RST

    .. _ocean:

    Ocean core
    ==========

    ...

In the Developer's Guide, labels have ``dev_`` prepended to them:

.. code-block:: RST

    .. _dev_ocean:

    Ocean core
    ==========

    ...

Each configuration should have the core prepended to its label (in case
multiple cores have the same configuration name), and each test case (if
explicitly labeled) should have the core and configuration prepended to it.
Thus, in the User's guide, we have:

.. code-block:: RST

    .. _ocean_baroclinic_channel:

    baroclinic_channel
    ==================

    ...

    .. _ocean_baroclinic_channel_default:

    default
    -------

And in the Developer's guide, these become:

.. code-block:: RST

    .. _dev_ocean_baroclinic_channel:

    baroclinic_channel
    ==================

    ...

    .. _dev_ocean_baroclinic_channel_default:

    default
    -------

Documentation for a core, configuration or test case in the User's Guide
should contain information that is needed for users who set up and run the test
case, including:

* Documentation for the MPAS core itself (if any)

* A page for each configurations with a section for each test case:

  * A citation or link where the configuration is defined (if any)

  * A brief overview of the test cases within the configuration

  * An image showing typical output from one of the test cases

  * A list of (commented) config options that apply to all test cases

  * A (typically brief) description of each test case

* A description of any common framework within the core that the configuration
  or test case pages may need to refer to.  This should only include framework
  that users may need to be aware of, e.g. because of :ref:`config_files`
  or namelist options they may wish to edit.

* A description of each test suite, including which test cases are included

The Developer's guide for each core should contain:

* Relevant technical details about development specific to that core

* A page for each configuration:

  * A description of any development-specific details of that configuration

  * A description of shared config, namelist and streams files

  * A description of shared steps

  * A description of any other shared framework code for the configuration

  * A description of each test case and its steps

* Technical details on the shared framework for the core

Finally, all functions in the configuration that are part of the public API
(i.e. all functions that don't start with an underscore) should be added to
``docs/<core>/api.rst``:

.. literalinclude:: docs_example.txt
   :language: RST

The Developer's Guide also contains details on the framework shared across
``compass``, so any updates to this framework should include relevant additions
or modifications ot the documentation.

.. _dev_docstrings:

Docstrings
----------

The Developer's Guide includes a :ref:`dev_api` that is automatically generated
from the python code and the `docstrings <https://www.python.org/dev/peps/pep-0257/>`_
at the beginning of each function.  ``compass`` uses docstrings in the
`Numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`_ format.
A typical example looks like this:

.. code-block:: python

    def compute_land_ice_pressure_and_draft(ssh, modifySSHMask, ref_density):
        """
        Compute the pressure from and overlying ice shelf and the ice-shelf draft

        Parameters
        ----------
        ssh : xarray.DataArray
            The sea surface height (the ice draft)

        modifySSHMask : xarray.DataArray
            A mask that is 1 where ``landIcePressure`` can be deviate from 0

        ref_density : float
            A reference density for seawater displaced by the ice shelf

        Returns
        -------
        landIcePressure : xarray.DataArray
            The pressure from the overlying land ice on the ocean

        landIceDraft : xarray.DataArray
            The ice draft, equal to the initial ``ssh``
        """

The docstring must include a brief description of the function.  Then, it
includes a ``Parameters`` section with entries for each argument.  The argument
are always given on their own line with the type, separated by `` : `` (note
the spaces on either side of the colon).  The type should not be in code format
(i.e. not in double back-quotes) because this interferes with Sphinx's ability
to link to the documentation for the type.  In the example above, Sphinx will
automatically find the API reference to ``xarray.DataArray`` within the
``xarray`` documentation (which is also written using sphinx).  If an argument
is a keyword argument (i.e. given with ``arg=value`` in the function
declaration), the type should be followed by ``, optional``, indicating that
the argument will take on a default value if it is not supplied.

On the next lines after the argument and type, indented by 4 spaces, is a brief
description of the argument.  If the argument is optional and the default value
is not obvious (e.g. ``arg=None`` is used as an indication that ``arg`` will be
replaced by something else in the function), it should also be described. If
the default value of the argument is obvious in the function declaration (e.g.
``arg=True``), no further description is necessary.

Finally, if the function returns values, these need to be described in the same
way as the parameters, with the name of the return values followed by a colon
and the type, then a description, indented by 4 spaces.

Other sections such as ``Raises`` and ``Examples`` are optional.
