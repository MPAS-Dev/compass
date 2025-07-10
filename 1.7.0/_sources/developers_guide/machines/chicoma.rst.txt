Chicoma
=======

For most machine-specific details (including config options and how to
enable hyperthreading), see the User's Guide under :ref:`machine_chicoma`.

chicoma-cpu, gnu
----------------

If you've set things up for this compiler, you should be able to source a load
script similar to:

.. code-block:: bash

    source load_dev_compass_1.2.0-alpha.4_chicoma-cpu_gnu_mpich.sh

Then, you can build the MPAS model with

.. code-block:: bash

    make [DEBUG=true] gnu-cray

debug jobs
~~~~~~~~~~

In order to run jobs in the debug queue, you will need to use: 

.. code-block:: cfg

    # Config options related to creating a job script
    [job]

    # The job partition to use
    partition = debug

    # The job reservation to use (needed for debug jobs)
    reservation = debug

    # The job quality of service (QOS) to use
    qos =
