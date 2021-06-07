import os
import tempfile
import requests
import progressbar
from urllib.parse import urlparse


def download(url, dest_path, config, exceptions=True):
    """
    Download a file from a URL to the given path or path name

    Parameters
    ----------
    url : str
        The URL (including file name) to download

    dest_path : str
        The path (including file name) where the downloaded file should be
        saved

    config : configparser.ConfigParser
        Configuration options used to find custom paths if ``dest_path`` is
        a config option

    exceptions : bool, optional
        Whether to raise exceptions when the download fails

    Returns
    -------
    dest_path : str
        The resulting file name if the download was successful, or None if not
    """

    in_file_name = os.path.basename(urlparse(url).path)
    dest_path = os.path.abspath(dest_path)
    out_file_name = os.path.basename(dest_path)

    do_download = config.getboolean('download', 'download')
    check_size = config.getboolean('download', 'check_size')
    verify = config.getboolean('download', 'verify')

    if not do_download:
        if not os.path.exists(dest_path):
            raise OSError('File not found and downloading is disabled: '
                          '{}'.format(dest_path))
        return dest_path

    if not check_size and os.path.exists(dest_path):
        return dest_path

    session = requests.Session()
    if not verify:
        session.verify = False

    # dest_path contains full path, so we need to make the relevant
    # subdirectories if they do not exist already
    directory = os.path.dirname(dest_path)
    try:
        os.makedirs(directory)
    except OSError:
        pass

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
            print('ERROR while downloading {}:'.format(in_file_name))
            print(e)
            return None

    if totalSize is None:
        # no content length header
        if not os.path.exists(dest_path):
            with open(dest_path, 'wb') as f:
                print('Downloading {}...'.format(in_file_name))
                try:
                    f.write(response.content)
                except requests.exceptions.RequestException:
                    if exceptions:
                        raise
                    else:
                        print('  {} failed!'.format(in_file_name))
                        return None
                else:
                    print('  {} done.'.format(in_file_name))
    else:
        # we can do the download in chunks and use a progress bar, yay!

        totalSize = int(totalSize)
        if os.path.exists(dest_path) and \
                totalSize == os.path.getsize(dest_path):
            # we already have the file, so just return
            return dest_path

        if out_file_name == in_file_name:
            file_names = in_file_name
        else:
            file_names = '{} as {}'.format(in_file_name, out_file_name)
        print('Downloading {} ({})...'.format(file_names,
                                              _sizeof_fmt(totalSize)))
        widgets = [progressbar.Percentage(), ' ', progressbar.Bar(),
                   ' ', progressbar.ETA()]
        bar = progressbar.ProgressBar(widgets=widgets,
                                      max_value=totalSize).start()
        size = 0
        with open(dest_path, 'wb') as f:
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
                    print('  {} failed!'.format(in_file_name))
                    return None
            else:
                print('  {} done.'.format(in_file_name))
    return dest_path


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
