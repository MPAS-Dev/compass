#!/usr/bin/env python

from jinja2 import Template


for os in ['linux', 'osx']:
    with open(f'template_{os}.yaml') as f:
        template_test = f.read()

    template = Template(template_test)
    for python in ['3.8', '3.9', '3.10', '3.11']:
        for mpi in ['nompi', 'mpich', 'openmpi']:
            script = template.render(python=python, mpi=mpi)
            filename = f'{os}_mpi_{mpi}_python{python}.yaml'
            with open(filename, 'w') as handle:
                handle.write(script)
