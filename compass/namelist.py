from importlib import resources


def parse_replacements(package, namelist):
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


def ingest(defaults_filename):
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


def replace(namelist, replacements):
    """ Replace entries in the namelist using the replacements dict """
    new = dict(namelist)
    for record in new:
        for key in replacements:
            if key in new[record]:
                new[record][key] = replacements[key]

    return new


def write(namelist, filename):
    """ Write the namelist out """

    with open(filename, 'w') as f:
        for record in namelist:
            f.write('&{}\n'.format(record))
            rec = namelist[record]
            for key in rec:
                f.write('    {} = {}\n'.format(key.strip(), rec[key].strip()))
            f.write('/\n')
