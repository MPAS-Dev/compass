.. _dev_building_docs:

**************************
Building the Documentation
**************************

To make a local test build of the documentation, you need to make a local
build of the ``compass`` conda package and include it in a conda environment
with some other required packages.

If you haven't installed
`Miniconda3 <https://docs.conda.io/en/latest/miniconda.html>`_, do so.  Then,
`add conda-forge <https://conda-forge.org/#about>`_ and install conda-build:

.. code-block:: bash

    miniconda=${HOME}/miniconda3
    source ${miniconda}/etc/profile.d/conda.sh
    conda activate base
    conda install conda-build

If you installed Miniconda3 somewhere other than the default location, change
``$miniconda`` both above and in the script below.

Then, run the following script to build the docs:

.. code-block:: bash

    #!/bin/bash

    miniconda=${HOME}/miniconda3

    source ${miniconda}/etc/profile.d/conda.sh

    # exit if a subprocess ends in errors
    set -e

    py=3.8
    mpi=mpich

    env=test_compass_mpi_${mpi}
    rm -rf ${miniconda}/conda-bld
    conda build -m ci/mpi_${mpi}.yaml recipe

    conda create --yes --quiet --name ${env} -c ${miniconda}/conda-bld/ \
        python=$py compass sphinx mock sphinx_rtd_theme

    conda activate $env

    version=$(python -c "import compass; print(compass.__version__)")
    echo "version: $version"
    export DOCS_VERSION="test"
    cd docs || exit 1
    rm -rf developers_guide/generated/ developers_guide/*/generated/ _build/
    make html

Finally, you can view the documentation by opening ``_build/html/index.html``.
