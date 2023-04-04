Perlmutter
==========

For most machine-specific details (including config options and how to
enable hyperthreading), see the User's Guide under :ref:`machine_perlmutter`.

pm-cpu, gnu
-----------

If you've set things up for this compiler, you should be able to source a load
script similar to:

.. code-block:: bash

    source load_dev_compass_1.2.0-alpha.2_pm-cpu_gnu_mpich.sh

Then, you can build the MPAS model with

.. code-block:: bash

    make [DEBUG=true] gnu-cray
