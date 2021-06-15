import os
import numpy
import xarray
import re
import fnmatch


def compare_variables(test_case, variables, filename1, filename2=None,
                      l1_norm=0.0, l2_norm=0.0, linf_norm=0.0, quiet=True,
                      check_outputs=True):
    """
    Compare variables between files in the current test case and/or with the
    baseline results.  The results of the comparison are added to the
    test case's "validation" dictionary, which the framework can use later to
    log the test case results and/or to raise an exception to indicate that
    the test case has failed.

    Parameters
    ----------
    test_case : compass.TestCase
        An object describing a test case to validate

    variables : list
        A list of variable names to compare

    filename1 : str
        The relative path to a file within the ``work_dir``.  If ``filename2``
        is also given, comparison will be performed with ``variables`` in that
        file.  If a baseline directory was provided when setting up the
        test case, the ``variables`` will be compared between this test case and
        the same relative filename in the baseline version of the test case.

    filename2 : str, optional
        The relative path to another file within the ``work_dir`` if comparing
        between files within the current test case.  If a baseline directory
        was provided, the ``variables`` from this file will also be compared
        with those in the corresponding baseline file.

    l1_norm : float, optional
        The maximum allowed L1 norm difference between the variables in
        ``filename1`` and ``filename2``.  To skip L1 norm check, pass None.

    l2_norm : float, optional
        The maximum allowed L2 norm difference between the variables in
        ``filename1`` and ``filename2``.  To skip L2 norm check, pass None.

    linf_norm : float, optional
        The maximum allowed L-Infinity norm difference between the variables in
        ``filename1`` and ``filename2``.  To skip Linf norm check, pass None.

    quiet : bool, optional
        Whether to print detailed information.  If quiet is False, the norm
        tolerance values being compared against will be printed when the
        comparison is made.  This is generally desirable when using nonzero
        norm tolerance values.

    check_outputs : bool, optional
        Whether to check to make sure files are valid outputs of steps in
        the test case.  This should be set to ``False`` if comparing with an
        output of a step in another test case.
    """
    work_dir = test_case.work_dir

    if test_case.validation is not None:
        validation = test_case.validation
    else:
        validation = {'internal_pass': None,
                      'baseline_pass': None}

    path1 = os.path.abspath(os.path.join(work_dir, filename1))
    if filename2 is not None:
        path2 = os.path.abspath(os.path.join(work_dir, filename2))
    else:
        path2 = None
    if check_outputs:
        file1_found = False
        file2_found = False
        for step_name, step in test_case.steps.items():
            for output in step.outputs:
                # outputs are already absolute paths combined with the step dir
                if output == path1:
                    file1_found = True
                if output == path2:
                    file2_found = True

        if not file1_found:
            raise ValueError('{} does not appear to be an output of any step '
                             'in this test case.'.format(filename1))
        if filename2 is not None and not file2_found:
            raise ValueError('{} does not appear to be an output of any step '
                             'in this test case.'.format(filename2))

    if filename2 is not None:
        internal_pass = _compare_variables(
            variables, path1, path2, l1_norm, l2_norm, linf_norm, quiet)

        if validation['internal_pass'] is None:
            validation['internal_pass'] = internal_pass
        else:
            validation['internal_pass'] = \
                validation['internal_pass'] and internal_pass

    if test_case.baseline_dir is not None:
        baseline_root = test_case.baseline_dir
        baseline_pass = True

        result = _compare_variables(
            variables, os.path.join(work_dir, filename1),
            os.path.join(baseline_root, filename1), l1_norm=0.0, l2_norm=0.0,
            linf_norm=0.0, quiet=quiet)
        baseline_pass = baseline_pass and result

        if filename2 is not None:
            result = _compare_variables(
                variables, os.path.join(work_dir, filename2),
                os.path.join(baseline_root, filename2), l1_norm=0.0,
                l2_norm=0.0, linf_norm=0.0, quiet=quiet)
            baseline_pass = baseline_pass and result

        if validation['baseline_pass'] is None:
            validation['baseline_pass'] = baseline_pass
        else:
            validation['baseline_pass'] = \
                validation['baseline_pass'] and baseline_pass

    test_case.validation = validation


def compare_timers(timers, config, work_dir, rundir1, rundir2=None):
    """
    Compare variables between files in the current test case and/or with the
    baseline results.

    Parameters
    ----------
    timers : list
        A list of timer names to compare

    config : configparser.ConfigParser
        Configuration options for the test case

    work_dir : str
        The work directory for the test case

    rundir1 : str
        The relative path to a directory within the ``work_dir``. If
        ``rundir2`` is also given, comparison will be performed with ``timers``
        in that file.  If a baseline directory was provided when setting up the
        test case, the ``timers`` will be compared between this test case and
        the same relative directory under the baseline version of the test case.

    rundir2 : str, optional
        The relative path to another file within the ``work_dir`` if comparing
        between files within the current test case.  If a baseline directory
        was provided, the ``timers`` from this file will also be compared with
        those in the corresponding baseline directory.
    """

    if rundir2 is not None:
        _compute_timers(os.path.join(work_dir, rundir1),
                        os.path.join(work_dir, rundir2), timers)

    if config.has_option('paths', 'baseline_dir'):
        baseline_root = config.get('paths', 'baseline_dir')

        _compute_timers(os.path.join(baseline_root, rundir1),
                        os.path.join(work_dir, rundir1), timers)

        if rundir2 is not None:
            _compute_timers(os.path.join(baseline_root, rundir2),
                            os.path.join(work_dir, rundir2), timers)


def _compare_variables(variables, filename1, filename2, l1_norm, l2_norm,
                       linf_norm, quiet):
    """ compare fields in the two files """

    for filename in [filename1, filename2]:
        if not os.path.exists(filename):
            raise OSError('File {} does not exist.'.format(filename))

    ds1 = xarray.open_dataset(filename1)
    ds2 = xarray.open_dataset(filename2)

    all_pass = True

    for variable in variables:
        for ds, filename in [(ds1, filename1), (ds2, filename2)]:
            if variable not in ds:
                raise ValueError('Variable {} not in {}.'.format(
                    variable, filename))

        da1 = ds1[variable]
        da2 = ds2[variable]

        if not numpy.all(da1.dims == da2.dims):
            raise ValueError("Dimensions for variable {} don't match between "
                             "files {} and {}.".format(
                                 variable, filename1, filename2))

        for dim in da1.sizes:
            if da1.sizes[dim] != da2.sizes[dim]:
                raise ValueError("Field sizes for variable {} don't match "
                                 "files {} and {}.".format(
                                     variable, filename1, filename2))

        if not quiet:
            print("    Pass thresholds are:")
            if l1_norm is not None:
                print("       L1: {:16.14e}".format(l1_norm))
            if l2_norm is not None:
                print("       L2: {:16.14e}".format(l2_norm))
            if linf_norm is not None:
                print("       L_Infinity: {:16.14e}".format(
                    linf_norm))
        variable_pass = True
        if 'Time' in da1.dims:
            time_range = range(0, da1.sizes['Time'])
            time_str = ', '.join(['{}'.format(j) for j in time_range])
            print('{} Time index: {}'.format(variable.ljust(20), time_str))
            for time_index in time_range:
                slice1 = da1.isel(Time=time_index)
                slice2 = da2.isel(Time=time_index)
                result = _compute_norms(slice1, slice2, quiet, l1_norm,
                                        l2_norm, linf_norm,
                                        time_index=time_index)
                variable_pass = variable_pass and result

        else:
            print('{}'.format(variable))
            result = _compute_norms(da1, da2, quiet, l1_norm, l2_norm,
                                    linf_norm)
            variable_pass = variable_pass and result

        # ANSI fail text: https://stackoverflow.com/a/287944/7728169
        start_fail = '\033[91m'
        start_pass = '\033[92m'
        end = '\033[0m'
        pass_str = '{}PASS{}'.format(start_pass, end)
        fail_str = '{}FAIL{}'.format(start_fail, end)

        if variable_pass:
            print('  {} {}\n'.format(pass_str, filename1))
        else:
            print('  {} {}\n'.format(fail_str, filename1))
        print('       {}\n'.format(filename2))
        all_pass = all_pass and variable_pass

    return all_pass


def _compute_norms(da1, da2, quiet, max_l1_norm, max_l2_norm, max_linf_norm,
                   time_index=None):
    """ Compute norms between variables in two DataArrays """

    da1 = _rename_duplicate_dims(da1)
    da2 = _rename_duplicate_dims(da2)

    result = True
    diff = numpy.abs(da1 - da2).values.ravel()

    l1_norm = numpy.linalg.norm(diff, ord=1)
    l2_norm = numpy.linalg.norm(diff, ord=2)
    linf_norm = numpy.linalg.norm(diff, ord=numpy.inf)

    if time_index is None:
        diff_str = ''
    else:
        diff_str = '{:d}: '.format(time_index)

    if l1_norm is not None:
        if max_l1_norm < l1_norm:
            result = False
    diff_str = '{} l1: {:16.14e} '.format(diff_str, l1_norm)

    if l2_norm is not None:
        if max_l2_norm < l2_norm:
            result = False
    diff_str = '{} l2: {:16.14e} '.format(diff_str, l2_norm)

    if linf_norm is not None:
        if max_linf_norm < linf_norm:
            result = False
    diff_str = '{} linf: {:16.14e} '.format(diff_str, linf_norm)

    if not quiet or not result:
        print(diff_str)

    return result


def _compute_timers(base_directory, comparison_directory, timers):
    """ Find timers and compute speedup between two run directories """
    for timer in timers:
        timer1_found, timer1 = _find_timer_value(timer, base_directory)
        timer2_found, timer2 = _find_timer_value(timer, comparison_directory)

        if timer1_found and timer2_found:
            if timer2 > 0.:
                speedup = timer1 / timer2
            else:
                speedup = 1.0

            percent = (timer2 - timer1) / timer1

            print("Comparing timer {}:".format(timer))
            print("             Base: {}".format(timer1))
            print("          Compare: {}".format(timer2))
            print("   Percent Change: {}%".format(percent * 100))
            print("          Speedup: {}".format(speedup))


def _find_timer_value(timer_name, directory):
    """ Find a timer in the given directory """
    # Build a regular expression for any two characters with a space between
    # them.
    regex = re.compile(r'(\S) (\S)')

    sub_timer_name = timer_name.replace(' ', '_')

    timer = 0.0
    timer_found = False
    for file in os.listdir(directory):
        if not timer_found:
            # Compare files written using built in MPAS timers
            if fnmatch.fnmatch(file, "log.*.out"):
                timer_line_size = 6
                name_index = 1
                total_index = 2
            # Compare files written using GPTL timers
            elif fnmatch.fnmatch(file, "timing.*"):
                timer_line_size = 6
                name_index = 0
                total_index = 3
            else:
                continue

            with open(os.path.join(directory, file), "r") as stats_file:
                for block in iter(lambda: stats_file.readline(), ""):
                    new_block = regex.sub(r"\1_\2", block[2:])
                    new_block_arr = new_block.split()
                    if len(new_block_arr) >= timer_line_size:
                        if sub_timer_name.find(new_block_arr[name_index]) >= 0:
                            try:
                                timer = \
                                    timer + float(new_block_arr[total_index])
                                timer_found = True
                            except ValueError:
                                pass

    return timer_found, timer


def _rename_duplicate_dims(da):
    dims = list(da.dims)
    new_dims = list(dims)
    duplicates = False
    for index, dim in enumerate(dims):
        if dim in dims[index+1:]:
            duplicates = True
            suffix = 2
            for other_index, other in enumerate(dims[index+1:]):
                if other == dim:
                    new_dims[other_index + index + 1] = \
                        '{}_{}'.format(dim, suffix)
                    suffix += 1

    if not duplicates:
        return da

    da = xarray.DataArray(data=da.values, dims=new_dims)
    return da
