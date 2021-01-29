import os
import tempfile
import requests
import progressbar

from compass.config import get_source_file


def add_input_file(step, filename=None, target=None, database=None,
                   url=None):
    """
    Add an input file to the step.  The file can be local, a symlink to
    a file that will be created in another step, a symlink to a file in one
    of the databases for files cached after download, and/or come from a
    specified URL.

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step

    filename : str, optional
        The relative path of the input file within the step's work directory.
        The default is the file name (without the path) of ``target``.

    target : str, optional
        A file that will be the target of a symlink to ``filename``.  If
        ``database`` is not specified, this should be an absolute path or a
        relative path from the step's work directory.  If ``database`` is
        specified, this is a relative path within the database and the name
        of the remote file to download.

    database : str, optional
        The name of a database for caching local files.  This will be a
        subdirectory of the local cache directory for this core.  If ``url``
        is not provided, the URL for downloading the file will be determined
        by combining the base URL of the data server, the relative path for the
        core, ``database`` and ``target``.

    url : str, optional
        The base URL for downloading ``target`` (if provided, or ``filename``
        if not).  This option should be set if the file is not in a database on
        the data server. The file's URL is determined by combining ``url``
        with the filename (without the directory) from ``target`` (or
        ``filename`` if ``target`` is not provided).  ``database`` is not
        included in the file's URL even if it is provided.
    """
    if filename is None:
        if target is None:
            raise ValueError('At least one of local_name and target are '
                             'required.')
        filename = os.path.basename(target)

    step['inputs'].append(dict(filename=filename, target=target,
                               database=database, url=url))


def add_output_file(step, filename):
    """
    Add the output file to the step

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step

    filename : str
        The relative path of the output file within the step's work directory
    """
    step['outputs'].append(filename)


def download(file_name, url, config, dest_path=None, dest_option=None,
             exceptions=True):
    """
    Download a file from a URL to the given path or path name

    Parameters
    ----------
    file_name : str
        The relative path of the source file relative to ``url`` and the
        destination path relative to ``dest_path`` (or the associated config
        option)

    url : str
        The URL where ``file_name`` can be found

    config : configparser.ConfigParser
        Configuration options used to find custom paths if ``dest_path`` is
        a config option

    dest_path : str, optional
        The output path; either ``dest_path`` or ``dest_option`` should be
        specified

    dest_option : str, optional
        An option in the ``paths`` config section  defining an output path;
        either ``dest_path`` or ``dest_option`` should be specified

    exceptions : bool, optional
        Whether to raise exceptions when the download fails

    Returns
    -------
    out_file_name : str
        The resulting file name if the download was successful
    """

    if dest_option is not None:
        out_file_name = get_source_file(dest_option, file_name, config)
    elif dest_path is not None:
        out_file_name = '{}/{}'.format(dest_path, file_name)
    else:
        raise ValueError('One of "dest_option" and "dest_path" must be '
                         'specified.')

    do_download = config.getboolean('download', 'download')
    check_size = config.getboolean('download', 'check_size')
    verify = config.getboolean('download', 'verify')

    if not do_download:
        if not os.path.exists(out_file_name):
            raise OSError('File not found and downloading is disabled: '
                          '{}'.format(out_file_name))
        return out_file_name

    if not check_size and os.path.exists(out_file_name):
        return out_file_name

    session = requests.Session()
    if not verify:
        session.verify = False

    # out_file_name contains full path, so we need to make the relevant
    # subdirectories if they do not exist already
    directory = os.path.dirname(out_file_name)
    try:
        os.makedirs(directory)
    except OSError:
        pass

    url = '{}/{}'.format(url, file_name)
    try:
        response = session.get(url, stream=True)
        totalSize = response.headers.get('content-length')
    except requests.exceptions.RequestException:
        if exceptions:
            raise
        else:
            print('  {} could not be reached!'.format(url))
            return None

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if exceptions:
            raise
        else:
            print('ERROR while downloading {}:'.format(file_name))
            print(e)
            return None

    if totalSize is None:
        # no content length header
        if not os.path.exists(out_file_name):
            with open(out_file_name, 'wb') as f:
                print('Downloading {}...'.format(file_name))
                try:
                    f.write(response.content)
                except requests.exceptions.RequestException:
                    if exceptions:
                        raise
                    else:
                        print('  {} failed!'.format(file_name))
                        return None
                else:
                    print('  {} done.'.format(file_name))
    else:
        # we can do the download in chunks and use a progress bar, yay!

        totalSize = int(totalSize)
        if os.path.exists(out_file_name) and \
                totalSize == os.path.getsize(out_file_name):
            # we already have the file, so just return
            return out_file_name

        print('Downloading {} ({})...'.format(file_name,
                                              _sizeof_fmt(totalSize)))
        widgets = [progressbar.Percentage(), ' ', progressbar.Bar(),
                   ' ', progressbar.ETA()]
        bar = progressbar.ProgressBar(widgets=widgets,
                                      max_value=totalSize).start()
        size = 0
        with open(out_file_name, 'wb') as f:
            try:
                for data in response.iter_content(chunk_size=4096):
                    size += len(data)
                    f.write(data)
                    bar.update(size)
                bar.finish()
            except requests.exceptions.RequestException:
                if exceptions:
                    raise
                else:
                    print('  {} failed!'.format(file_name))
                    return None
            else:
                print('  {} done.'.format(file_name))
    return out_file_name


def symlink(target, link_name, overwrite=True):
    """
    From https://stackoverflow.com/a/55742015/7728169
    Create a symbolic link named link_name pointing to target.
    If link_name exists then FileExistsError is raised, unless overwrite=True.
    When trying to overwrite a directory, IsADirectoryError is raised.

    Parameters
    ----------
    target : str
        The file path to link to

    link_name : str
        The name of the new link

    overwrite : bool, optional
        Whether to replace an existing link if one already exists
    """

    if not overwrite:
        os.symlink(target, link_name)
        return

    # os.replace() may fail if files are on different filesystems
    link_dir = os.path.dirname(link_name)

    # Create link to target with temporary file_name
    while True:
        temp_link_name = tempfile.mktemp(dir=link_dir)

        # os.* functions mimic as closely as possible system functions
        # The POSIX symlink() returns EEXIST if link_name already exists
        # https://pubs.opengroup.org/onlinepubs/9699919799/functions/symlink.html
        try:
            os.symlink(target, temp_link_name)
            break
        except FileExistsError:
            pass

    # Replace link_name with temp_link_name
    try:
        # Preempt os.replace on a directory with a nicer message
        if not os.path.islink(link_name) and os.path.isdir(link_name):
            raise IsADirectoryError(
                f"Cannot symlink over existing directory: '{link_name}'")
        os.replace(temp_link_name, link_name)
    except BaseException:
        if os.path.islink(temp_link_name):
            os.remove(temp_link_name)
        raise


def process_step_inputs_and_outputs(step, config):
    """
    Process the inputs to and outputs from a step added with
    :py:func:`compass.io.add_input_file` and
    :py:func:`compass.io.add_output_file`.  This includes downloading files,
    making symlinks, and converting relative paths to absolute paths.

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step

    config : configparser.ConfigParser
        Configuration options used to get the server base url, core path on
        the server and cache root for databases if any files are to be
        downloaded to these locations
   """
    core = step['core']
    step_dir = step['work_dir']

    inputs = []
    for entry in step['inputs']:
        filename = entry['filename']
        target = entry['target']
        database = entry['database']
        url = entry['url']

        download_target = None
        download_path = None

        if database is not None:
            # we're downloading a file to a cache of a database (if it's not
            # already there.
            if url is None:
                base_url = config.get('download', 'server_base_url')
                core_path = config.get('download', 'core_path')
                url = '{}/{}/{}'.format(base_url, core_path, database)

            if target is None:
                target = filename

            download_target = target

            database_root = config.get('paths',
                                       '{}_database_root'.format(core))
            download_path = os.path.join(database_root, database)
        elif url is not None:
            if target is None:
                download_target = filename
                download_path = '.'
            else:
                download_path, download_target = os.path.split(target)

        if url is not None:
            download_target = download(download_target, url, config,
                                       download_path)
            if target is not None:
                # this is the absolute path that we presumably want
                target = download_target

        if target is not None:
            symlink(target, os.path.join(step_dir, filename))
            inputs.append(target)
        else:
            inputs.append(filename)

    # convert inputs and outputs to absolute paths
    step['inputs'] = [os.path.abspath(os.path.join(step_dir, filename)) for
                      filename in inputs]

    step['outputs'] = [os.path.abspath(os.path.join(step_dir, filename)) for
                       filename in step['outputs']]


# From https://stackoverflow.com/a/1094933/7728169
def _sizeof_fmt(num, suffix='B'):
    """
    Covert a number of bytes to a human-readable file size
    """
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)
