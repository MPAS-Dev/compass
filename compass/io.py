import os
import tempfile
import requests
import progressbar

from compass.config import get_source_file


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
    except:
        if os.path.islink(temp_link_name):
            os.remove(temp_link_name)
        raise


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
