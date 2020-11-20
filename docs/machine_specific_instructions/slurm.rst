Slurm job queueing
==================

Most systems now use slurm. Here are some basic commands:

.. code-block:: bash

    salloc -N 1 -t 2:0:0 # interactive job (see machine specific versions below)
    sbatch script # submit a script
    squeue # show all jobs
    squeue -u $my_moniker # show only your jobs
    scancel jobID # cancel a job


