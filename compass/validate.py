import os
import numpy
import xarray
import re
import fnmatch


def compare_variables(variables, config, work_dir, filename1, filename2=None,
                      l1_norm=0.0, l2_norm=0.0, linf_norm=0.0, quiet=False):
    """
    Compare variables between files in the current testcase and/or with the
    baseline results.

    Parameters
    ----------
    variables : list
        A list of variable names to compare

    config : configparser.ConfigParser
        Configuration options for the testcase

    work_dir : str
        The work directory for the testcase

    filename1 : str
        The relative path to a file within the ``work_dir``.  If ``filename2``
        is also given, comparison will be performed with ``variables`` in that
        file.  If a baseline directory was provided when setting up the
        testcase, the ``variables`` will be compared between this testcase and
        the same relative filename in the baseline version of the testcase.

    filename2 : str, optional
        The relative path to another file within the ``work_dir`` if comparing
        between files within the current testcase.  If a baseline directory
        was provided, the ``variables`` from this file will also be compared
        with those in the corresponding baseline file.

    l1_norm : float, optional
        The maximum allowed L1 norm difference between the variables in
        ``filename1`` and ``filename2``.

    l2_norm : float, optional
        The maximum allowed L2 norm difference between the variables in
        ``filename1`` and ``filename2``.

    linf_norm : float, optional
        The maximum allowed L-Infinity norm difference between the variables in
        ``filename1`` and ``filename2``.

    quiet : bool, optional
        Whether to print

    Raises
    ------
    ValueError
        If one or more of the norms is outside the required bounds

    """

    all_pass = True
    if filename2 is not None:
        result = _compare_variables(
            variables, os.path.join(work_dir, filename1),
            os.path.join(work_dir, filename2), l1_norm, l2_norm, linf_norm,
            quiet)
        all_pass = all_pass and result

    if config.has_option('paths', 'baseline_dir'):
        baseline_root = config.get('paths', 'baseline_dir')

        result = _compare_variables(
            variables, os.path.join(work_dir, filename1),
            os.path.join(baseline_root, filename1), l1_norm=0.0, l2_norm=0.0,
            linf_norm=0.0, quiet=quiet)
        all_pass = all_pass and result

        if filename2 is not None:
            result = _compare_variables(
                variables, os.path.join(work_dir, filename2),
                os.path.join(baseline_root, filename2), l1_norm=0.0,
                l2_norm=0.0, linf_norm=0.0, quiet=quiet)
            all_pass = all_pass and result

    if not all_pass:
        raise ValueError('Comparison failed, see above.')


def compare_timers(timers, config, work_dir, rundir1, rundir2=None):
    """
    Compare variables between files in the current testcase and/or with the
    baseline results.

    Parameters
    ----------
    timers : list
        A list of timer names to compare

    config : configparser.ConfigParser
        Configuration options for the testcase

    work_dir : str
        The work directory for the testcase

    rundir1 : str
        The relative path to a directory within the ``work_dir``. If
        ``rundir2`` is also given, comparison will be performed with ``timers``
        in that file.  If a baseline directory was provided when setting up the
        testcase, the ``timers`` will be compared between this testcase and
        the same relative directory under the baseline version of the testcase.

    rundir2 : str, optional
        The relative path to another file within the ``work_dir`` if comparing
        between files within the current testcase.  If a baseline directory
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
            print("Beginning variable comparisons for all time levels "
                  "of field '{}'. Note any time levels reported are "
                  "0-based.".format(variable))
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
            for time_index in range(0, da1.sizes['Time']):
                slice1 = da1.isel(Time=time_index)
                slice2 = da2.isel(Time=time_index)
                result = _compute_norms(slice1, slice2, quiet, l1_norm,
                                        l2_norm, linf_norm,
                                        time_index=time_index)
                variable_pass = variable_pass and result

        else:
            result = _compute_norms(da1, da2, quiet, l1_norm, l2_norm,
                                    linf_norm)
            variable_pass = variable_pass and result

        if variable_pass:
            print(' ** PASS Comparison of {} between {} and\n'
                  '    {}'.format(variable, filename1, filename2))
        else:
            print(' ** FAIL Comparison of {} between {} and\n'
                  '    {}'.format(variable, filename1, filename2))
        all_pass = all_pass and variable_pass

    return all_pass


def _compute_norms(da1, da2, quiet, max_l1_norm, max_l2_norm, max_linf_norm,
                   time_index=None):
    """ Compute norms between variables in two DataArrays """

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
            print("   Percent Change: {}%%".format(percent * 100))
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
