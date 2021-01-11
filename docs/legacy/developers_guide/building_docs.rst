.. _legacy_dev_building_docs:

**************************
Building the Documentation
**************************

To make a local test build of the documentation, you need to set up a conda
environment with some required packages:

.. code-block:: bash

  $ conda create -y -n test_compass_docs python=3.8 sphinx mock sphinx_rtd_theme
  $ conda activate test_compass_docs

Then, to build the documentation, run:

.. code-block:: bash

  $ export DOCS_VERSION="test"
  $ cd docs
  $ make html

Then, you can view the documentation by opening ``_build/html/index.html``.
