Scripts
=======

.. _list_testcases:

list\_testcases.py
------------------

This script is used to list available test cases.

It iterates through the directory structure and prints out configuration
options to setup specific test cases. Additionally, the ``-o``, ``-c``, ``-r``, and ``-t``
flags can be used to narrow the information that this script prints. If any of
them are passed in, the script will only print test cases that match all
criteria.

Additionally, if ``-n`` is passed in to get information about a single test case,
it will only print the flags needed to setup that specific test case.

Command-line options::

    $ ./list_testcases.py -h
    usage: list_testcases.py [-h] [-o CORE] [-c CONFIG] [-r RES] [-t TEST]
                             [-n NUMBER]

    This script is used to list available test cases.

    It iterates through the directory structure and prints out configuration
    options to setup specific test cases. Additionally, the -o, -c, -r, and -t
    flags can be used to narrow the information that this script prints. If any of
    them are passed in, the script will only print test cases that match all
    criteria.

    Additionally, if -n is passed in to get information about a single test case,
    it will only print the flags needed to setup that specific test case.

    optional arguments:
      -h, --help            show this help message and exit
      -o CORE, --core CORE  Core to search for configurations within
      -c CONFIG, --configuration CONFIG
                            Configuration name to search for
      -r RES, --resolution RES
                            Resolution to search for
      -t TEST, --test TEST  Test name to search for
      -n NUMBER, --number NUMBER
                            If set, script will print the flags to use a the N'th configuration.


.. _setup_testcase:

setup\_testcase.py
------------------

This script is used to setup individual test cases. Available test cases
can be see using the :ref:`list_testcases` script.

Specifically, this script parses XML files that define cases (steps in test
cases) and driver scripts, and generates directories and scripts to run each
step in the process of creating a test case.

This script requires a setup configuration file. Configuration files are
specific to each core. Template configuration files for each core can be seen
in this directory named ``general.config.{core}`` (see :ref:`compass_config`).
Each core may have different requirements as far as what is required within a
configuration file.

Command-line options::

    $ ./setup_testcase.py -h
    usage: setup_testcase.py [-h] [-o CORE] [-c CONFIG] [-r RES] [-t TEST]
                             [-n NUM] [-f FILE] [-m FILE] [-b PATH] [-q]
                             [--no_download] [--work_dir PATH]

    This script is used to setup individual test cases. Available test cases
    can be see using the list_testcases.py script.

    Specifically, this script parses XML files that define cases (steps in test
    cases) and driver scripts, and generates directories and scripts to run each
    step in the process of creating a test case.

    This script requires a setup configuration file. Configuration files are
    specific to each core. Template configuration files for each core can be seen
    in this directory named 'general.config.{core}'. Each core may have different
    requirements as far as what is required within a configuration file.

    optional arguments:
      -h, --help            show this help message and exit
      -o CORE, --core CORE  Core that contains configurations
      -c CONFIG, --configuration CONFIG
                            Configuration to setup
      -r RES, --resolution RES
                            Resolution of configuration to setup
      -t TEST, --test TEST  Test name within a resolution to setup
      -n NUM, --case_number NUM
                            Case number to setup, as listed from list_testcases.py. Can be a comma delimited list of case numbers.
      -f FILE, --config_file FILE
                            Configuration file for test case setup
      -m FILE, --model_runtime FILE
                            Definition of how to build model run commands on this machine
      -b PATH, --baseline_dir PATH
                            Location of baseslines that can be compared to
      -q, --quiet           If set, script will not write a command_history file
      --no_download         If set, script will not auto-download base_mesh files
      --work_dir PATH       If set, script will create case directories in work_dir rather than the current directory.


.. _clean_testcase:

clean\_testcase.py
------------------

This script is used to clean one or more test cases that have already been
setup.

It will remove directories and driver scripts that were generated as part of
setting up a test case.

Command-line options::

    $ ./clean_testcase.py -h
    usage: clean_testcase.py [-h] [-o CORE] [-c CONFIG] [-r RES] [-t TEST]
                             [-n NUM] [-q] [-a] [--work_dir PATH]

    This script is used to clean one or more test cases that have already been
    setup.

    It will remove directories / driver scripts that were generated as part of
    setting up a test case.

    optional arguments:
      -h, --help            show this help message and exit
      -o CORE, --core CORE  Core that contains configurations to clean
      -c CONFIG, --configuration CONFIG
                            Configuration to clean
      -r RES, --resolution RES
                            Resolution of configuration to clean
      -t TEST, --test TEST  Test name within a resolution to clean
      -n NUM, --case_number NUM
                            Case number to clean, as listed from list_testcases.py. Can be a comma delimited list of case numbers.
      -q, --quiet           If set, script will not write a command_history file
      -a, --all             Is set, the script will clean all test cases in the work_dir.
      --work_dir PATH       If set, script will clean case directories in work_dir rather than the current directory.



.. _manage_regression_suite:

manage\_regression\_suite.py
----------------------------

This script is used to manage regression suites. A regression suite is a set of
test cases that ensure one or more features in a model meet certain criteria.

Using this script one can setup or clean a regression suite.

When setting up a regression suite, this script will generate a script to run
all tests in the suite, and additionally setup each individual test case.

When cleaning a regression suite, this script will remove any generated files
for each individual test case, and the run script that runs all test cases.

Command-line options::

    $ ./manage_regression_suite.py -h
    usage: manage_regression_suite.py [-h] -t FILE [-f FILE] [-s] [-c] [-v]
                                      [-m FILE] [-b PATH] [--work_dir PATH]

    This script is used to manage regression suites. A regression suite is a set of
    test cases that ensure one or more features in a model meet certain criteria.

    Using this script one can setup or clean a regression suite.

    When setting up a regression suite, this script will generate a script to run
    all tests in the suite, and additionally setup each individual test case.

    When cleaning a regression suite, this script will remove any generated files
    for each individual test case, and the run script that runs all test cases.

    optional arguments:
      -h, --help            show this help message and exit
      -t FILE, --test_suite FILE
                            Path to file containing a test suite to setup
      -f FILE, --config_file FILE
                            Configuration file for test case setup
      -s, --setup           Option to determine if regression suite should be setup or not.
      -c, --clean           Option to determine if regression suite should be cleaned or not.
      -v, --verbose         Use verbose output from setup_testcase.py
      -m FILE, --model_runtime FILE
                            Definition of how to build model run commands on this machine
      -b PATH, --baseline_dir PATH
                            Location of baseslines that can be compared to
      --work_dir PATH       If set, script will setup the test suite in work_dir rather in this script's location.

