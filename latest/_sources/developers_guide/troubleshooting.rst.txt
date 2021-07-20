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
