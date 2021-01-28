from importlib import resources


def add_namelist_file(step, package, namelist, out_name=None):
    """
    Add a namelist file to the step to be parsed later with a call to
    :py:func:`compass.namelist.generate_namelist()`.

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step

    package : Package
        The package name or module object that contains ``namelist``

    namelist : str
        The name of the namelist replacements file to read from

    out_name : str, optional
        The name of the namelist file to write out, ``namelist.<core>`` by
        default
    """
    namelist_list = _get_list(step, out_name)

    namelist_list.append(dict(package=package, namelist=namelist))


def add_namelist_options(step, options, out_name=None):
    """
    Parse the replacement namelist options from the given file

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step

    options : dict
        A dictionary of options and value to replace namelist options with new
        values.

    out_name : str, optional
        The name of the namelist file to write out, ``namelist.<core>`` by
        default
    """
    namelist_list = _get_list(step, out_name)

    namelist_list.append(dict(options=options))


def generate_namelist(step, config, out_name=None, mode='forward'):
    """
    Writes out a namelist file in the work directory with new values given
    by parsing the files and dictionaries in the step's ``namelist_data``.

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step from the ``collect()``
        function, used to get the work directory and the core that the test
        case belongs to.

    config : configparser.ConfigParser
        Configuration options used determine the name of the namelist file and
        the starting template of namelist records and options

    out_name : str, optional
        The name of the namelist file (without a path), with a default value of
        ``namelist.<core>``

    mode : {'init', 'forward'}, optional
        The mode that the model will run in
    """

    step_work_dir = step['work_dir']
    core = step['core']
    if out_name is None:
        out_name = 'namelist.{}'.format(core)

    replacements = dict()

    if out_name not in step['namelist_data']:
        raise ValueError("It doesn't look like there are namelist options for "
                         "the output file name {}".format(out_name))

    for entry in step['namelist_data'][out_name]:
        if 'options' in entry:
            # this is a dictionary of replacement namelist options
            options = entry['options']
        else:
            options = _parse_replacements(entry['package'], entry['namelist'])
        replacements.update(options)

    defaults_filename = config.get('namelists', mode)
    out_filename = '{}/{}'.format(step_work_dir, out_name)

    namelist = _ingest(defaults_filename)

    namelist = _replace(namelist, replacements)

    _write(namelist, out_filename)


def update(replacements, step_work_dir, out_name):
    """
    Update an existing namelist file with additional ``replacements``.  This
    would typically be used for namelist options that are only known at
    runtime, not during setup.  For example, the number of PIO tasks and the
    stride between tasks, which are related to the number of nodes and cores.

    Parameters
    ----------
    replacements : dict
        A dictionary of options and value to replace namelist options with new
        values

    step_work_dir : str
        The path for the work directory for the step that this namelist is
        being generated for

    out_name : str
        The name of the namelist file (without a path)
    """

    filename = '{}/{}'.format(step_work_dir, out_name)

    namelist = _ingest(filename)

    namelist = _replace(namelist, replacements)

    _write(namelist, filename)


def _parse_replacements(package, namelist):
    """
    Parse the replacement namelist options from the given file

    Parameters
    ----------
    package : Package
        The package name or module object that contains ``namelist``

    namelist : str
        The name of the namelist replacements file to read from

    Returns
    -------
    replacements : dict
        A dictionary of replacement namelist options
    """

    lines = resources.read_text(package, namelist).split('\n')
    replacements = dict()
    for line in lines:
        if '=' in line:
            opt, val = line.split('=')
            replacements[opt.strip()] = val.strip()

    return replacements


def _ingest(defaults_filename):
    """ Read the defaults file """
    with open(defaults_filename, 'r') as f:
        lines = f.readlines()

    namelist = dict()
    record = None
    for line in lines:
        if '&' in line:
            record = line.strip('&').strip('\n').strip()
            namelist[record] = dict()
        elif '=' in line:
            if record is not None:
                opt, val = line.strip('\n').split('=')
                namelist[record][opt.strip()] = val.strip()

    return namelist


def _replace(namelist, replacements):
    """ Replace entries in the namelist using the replacements dict """
    new = dict(namelist)
    for record in new:
        for key in replacements:
            if key in new[record]:
                new[record][key] = replacements[key]

    return new


def _write(namelist, filename):
    """ Write the namelist out """

    with open(filename, 'w') as f:
        for record in namelist:
            f.write('&{}\n'.format(record))
            rec = namelist[record]
            for key in rec:
                f.write('    {} = {}\n'.format(key.strip(), rec[key].strip()))
            f.write('/\n')


def _get_list(step, out_name):
    if 'namelist_data' not in step:
        step['namelist_data'] = dict()
    if out_name is None:
        out_name = 'namelist.{}'.format(step['core'])

    if out_name not in step['namelist_data']:
        step['namelist_data'][out_name] = list()

    return step['namelist_data'][out_name]
