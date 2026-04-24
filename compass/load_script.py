import os

from compass.io import symlink


def get_active_load_script():
    """
    Get the active COMPASS load script from the environment.

    ``mache.deploy`` exports ``COMPASS_LOAD_SCRIPT`` from generated load
    scripts.  ``LOAD_COMPASS_ENV`` is retained as a legacy fallback while the
    transition is underway.
    """
    return os.environ.get(
        'COMPASS_LOAD_SCRIPT', os.environ.get('LOAD_COMPASS_ENV')
    )


def symlink_load_script(work_dir):
    """
    Symlink the active load script into a work directory if one is available.
    """
    script_filename = get_active_load_script()
    if script_filename is not None:
        symlink(script_filename, os.path.join(work_dir, 'load_compass_env.sh'))
