#!/usr/bin/env python

from jinja2 import Template


with open('template.yaml') as f:
    tempate_test = f.read()

template = Template(tempate_test)
for python in ['3.7', '3.8', '3.9', '3.10']:
    for mpi in ['nompi', 'mpich', 'openmpi']:
        script = template.render(python=python, mpi=mpi)
        filename = 'mpi_{}_python{}.yaml'.format(mpi, python)
        with open(filename, 'w') as handle:
            handle.write(script)
