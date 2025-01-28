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

Details of the error may vary but the message indicates in some way that there
was a problem solving for the requested combination of packages.  This likely
indicates that you have an existing compass development environment
(``dev_compass_*``) that can't be updated to be compatible with the new
set of development packages given in:

.. code-block:: none

    conda/build*/spec-file*.txt

The solution should be to recreate the environment rather than trying to
update it:

.. code-block:: bash

    ./conda/configure_compass_env.sh --recreate ...

The ``--recreate`` flag will first delete the existing ``dev_compass_*`` conda
environment before creating it again with the new set of packages required for
developing with the requested compiler and MPI type.

.. _dev_troubleshooting_proxy:

Proxy on LANL Macs
------------------

If you are on a LANL Mac using the VPN and you see errors like:

.. code-block:: none

  An HTTP error occurred when trying to retrieve this URL.

this likely indicates that you need to set some proxy-related variables.

From a `LANL internal site <http://trac.lanl.gov/cgi-bin/external/trac.cgi/wiki/proxy>`_:

  Bash, and other shells do not honor the system wide proxy settings on the Mac.
  There are a couple of ways for a person to manage these settings.

  First, would be to the following will work in one's ``~/.profile`` - that way
  every time a shell is spawned with your username, these settings will be
  enabled:

  .. code-block:: none

    proxy_enable() {
        export ALL_PROXY=proxyout.lanl.gov:8080
        export all_proxy=proxyout.lanl.gov:8080
        export NO_PROXY=localhost,127.0.0.1,*.lanl.gov,lanl.gov
        export no_proxy=localhost,127.0.0.1,*.lanl.gov,lanl.gov
    }

    proxy_disable() {
        unset ALL_PROXY
        unset all_proxy
        unset NO_PROXY
        unset no_proxy
    }

    proxy_enable

If you follow this approach, you would then call ``proxy_disable`` anytime you
want to turn off the proxy (e.g. if the VPN is not running).

gethostbyname failed
--------------------

If you see errors like the following (particularly on LANL Macs):

.. code-block:: none

    Fatal error in MPI_Init: Other MPI error, error stack:
    MPIR_Init_thread(159)..............:
    MPID_Init(164).....................: channel initialization failed
    MPIDI_CH3_Init(95).................:
    MPID_nem_init(314).................:
    MPID_nem_tcp_init(173).............:
    MPID_nem_tcp_get_business_card(395):
    GetSockInterfaceAddr(369)..........: gethostbyname failed, pn2034311.lanl.gov (errno 0)

this likely indicates that MPI is having a problem finding the local host.

The solution is to set the following config option in the ``parallel`` section
of your user config file:

.. code-block:: cfg

     # The parallel section describes options related to running tests in parallel
     [parallel]

     # whether to use mpirun or srun to run the model
     parallel_executable = mpirun -host localhost

The `example config files <https://github.com/MPAS-Dev/compass/tree/main/example_configs>`_
have been updated to include this flag.
