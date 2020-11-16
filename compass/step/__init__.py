import os
import stat
from jinja2 import Template
from importlib import resources


def get_default(module):
    """
    Set up a default dictionary describing the step in the given module.  The
    dictionary contains the full name of the python module for the step, the
    name of the step (the name of the python file without the ``.py``
    extension), the subdirectory for the step (the same as the ``name``),
    the names of the ``setup()`` and ``run()`` functions within the module,
    and empty lists of ``inputs`` and ``outputs``, to be filled with the
    files required to run the step or produced by the step, respectively.

    Parameters
    ----------
    module : str
        The full name of the python module for the step, usually supplied from
        ``__name__``

    Returns
    -------
    step : dict
        A dictionary with the default information about the step, most of which
        can be modified as appropriate

    """
    name = module.split('.')[-1]
    step = {'module': module,
            'name': name,
            'subdir': name,
            'setup': 'setup',
            'run': 'run',
            'inputs': [],
            'outputs': []}
    return step


def generate_run(step):
    """
    Generate a ``run.py`` script for the given testcase or step.

    Parameters
    ----------
    step : dict
        The dictionary of information about the step, used to fill in the
        script template
    """
    step_dir = step['work_dir']

    template = Template(resources.read_text('compass.step', 'step.template'))
    script = template.render(step=step)

    run_filename = os.path.join(step_dir, 'run.py')
    with open(run_filename, 'w') as handle:
        handle.write(script)

    # make sure it has execute permission
    st = os.stat(run_filename)
    os.chmod(run_filename, st.st_mode | stat.S_IEXEC)
