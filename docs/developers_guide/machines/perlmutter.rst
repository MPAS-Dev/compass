.. _dev_machine_perlmutter:

Perlmutter
==========

For most machine-specific details (including config options and how to
enable hyperthreading), see the User's Guide under :ref:`machine_perlmutter`.

pm-cpu, gnu
-----------

If you've set things up for this compiler, you should be able to source a load
script similar to:

.. code-block:: bash

    source load_compass_pm-cpu_gnu_mpich.sh

Then, you can build the MPAS model with

.. code-block:: bash

    make [DEBUG=true] gnu-cray

pm-gpu, gnugpu
--------------

Perlmutter's GPU nodes are supported for MALI with the GPU-enabled Albany
library.  Deploy with ``--machine pm-gpu`` and ``--with-albany``, then
source a load script similar to:

.. code-block:: bash

    source load_compass_pm-gpu_gnugpu_mpich.sh

Then, you can build MALI with

.. code-block:: bash

    make [DEBUG=true] ALBANY=true gnu-cray
