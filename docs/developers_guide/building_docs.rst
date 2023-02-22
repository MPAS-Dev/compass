.. _dev_building_docs:

**************************
Building the Documentation
**************************

As long as you have followed the procedure in :ref:`dev_conda_env` for setting
up your conda environment, you will already have the packages available that
you need to build the documentation.

Then, run the following script to build the docs:

.. code-block:: bash

    export DOCS_VERSION="test"
    cd docs
    rm -rf developers_guide/generated/ developers_guide/*/generated/ _build/
    make html

You may need to re-source your compass load script in the root of the compass branch
for the api docs to build successfully if you have added new modules since the load
script was last sourced.  The load script will reinstall compass into the conda
environment when it is sourced in the root of the compass branch.

You can view the documentation by opening ``_build/html/index.html``.
