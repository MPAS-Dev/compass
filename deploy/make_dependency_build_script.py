#!/usr/bin/env python
import os
import re
import stat
import argparse
import requests
from lxml import etree
from jinja2 import Template


def get_host_info(machine):
    if machine == 'cori-haswell':
        base_path = "/global/cfs/cdirs/e3sm/software/compass/cori-haswell"
        conda_base = "/global/cfs/cdirs/e3sm/software/anaconda_envs/base"
        group = "e3sm"
    elif machine == 'anvil':
        base_path = "/lcrc/soft/climate/compass/anvil"
        conda_base = "/lcrc/soft/climate/e3sm-unified/base"
        group = "climate"
    elif machine == 'chrysalis':
        base_path = "/lcrc/soft/climate/compass/chrysalis"
        conda_base = "/lcrc/soft/climate/e3sm-unified/base"
        group = "climate"
    elif machine == 'compy':
        base_path = "/share/apps/E3SM/conda_envs/compass"
        conda_base = "/share/apps/E3SM/conda_envs/base"
        group = "users"
    elif machine == 'grizzly':
        base_path = "/usr/projects/climate/SHARED_CLIMATE/compass/grizzly"
        conda_base = "/usr/projects/climate/SHARED_CLIMATE/anaconda_envs/base"
        group = "climate"
    elif machine == 'badger':
        base_path = "/usr/projects/climate/SHARED_CLIMATE/compass/badger"
        conda_base = "/usr/projects/climate/SHARED_CLIMATE/anaconda_envs/base"
        group = "climate"
    else:
        raise ValueError(
            "Unknown machine {}.  Add base_path and group for "
            "this machine to the script.".format(machine))

    return base_path, conda_base, group


def main():

    parser = argparse.ArgumentParser(
        description='build dependencies for a machine')

    parser.add_argument("-m", "--machine", dest="machine", required=True,
                        help="The name of the machine", metavar="MACH")

    parser.add_argument("-c", "--compiler", dest="compiler", required=True,
                        help="The name of the compiler", metavar="COMP")

    parser.add_argument("-i", "--mpilib", dest="mpilib", required=True,
                        help="The name of the MPI library", metavar="MPI")

    parser.add_argument("-s", "--scorpio", dest="scorpio", default="1.1.6",
                        help="The SCORPIO version to build", metavar="SCORPIO")

    parser.add_argument("-e", "--esmf", dest="esmf", default="8.1.0",
                        help="The ESMF version to build", metavar="ESMF")

    args = parser.parse_args()

    with open(os.path.join('..', 'compass', '__init__.py')) as f:
        init_file = f.read()

    compass = re.search(r'{}\s*=\s*[(]([^)]*)[)]'.format('__version_info__'),
                        init_file).group(1).replace(', ', '.')

    machine = args.machine
    compiler = args.compiler
    mpilib = args.mpilib
    scorpio = args.scorpio
    esmf = args.esmf

    base_path, conda_base, group = get_host_info(machine)

    for filename in ['config_machines.xml', 'config_compilers.xml']:
        url = 'https://raw.githubusercontent.com/E3SM-Project/E3SM/master/' \
              'cime_config/machines/{}'.format(filename)
        r = requests.get(url)
        with open(filename, 'wb') as outfile:
            outfile.write(r.content)

    root = etree.parse('config_machines.xml')
    
    machines = next(root.iter('config_machines'))

    for mach in machines:
        if mach.tag == 'machine' and mach.attrib['MACH'] == machine:
            break

    compilers = None
    for child in mach:
        if child.tag == 'COMPILERS':
            compilers = child.text.split(',')
            break

    if compiler not in compilers:
        raise ValueError('Compiler {} not found on {}. Try: {}'.format(
            compiler, machine, compilers))

    mpilibs = None
    for child in mach:
        if child.tag == 'MPILIBS':
            mpilibs = child.text.split(',')
            break

    if mpilib not in mpilibs:
        raise ValueError('MPI library {} not found on {}. Try: {}'.format(
            mpilib, machine, mpilibs))

    machine_os = None
    for child in mach:
        if child.tag == 'OS':
            machine_os = child.text
            break

    commands = []
    modules = next(mach.iter('module_system'))
    for module in modules:
        if module.tag == 'modules':
            include = True
            if 'compiler' in module.attrib and \
                    module.attrib['compiler'] != compiler:
                include = False
            if 'mpilib' in module.attrib and \
                    module.attrib['mpilib'] != mpilib and \
                    module.attrib['mpilib'] != '!mpi-serial':
                include = False
            if include:
                for command in module:
                    if command.tag == 'command':
                        text = 'module {}'.format(command.attrib['name'])
                        if command.text is not None:
                            text = '{} {}'.format(text, command.text)
                        commands.append(text)

    root = etree.parse('config_compilers.xml')
    
    compilers = next(root.iter('config_compilers'))

    mpicc = None
    mpifc = None
    mpicxx = None
    for comp in compilers:
        if comp.tag != 'compiler':
            continue
        if 'COMPILER' in comp.attrib and comp.attrib['COMPILER'] != compiler:
            continue
        if 'OS' in comp.attrib and comp.attrib['OS'] != machine_os:
            continue
        if 'MACH' in comp.attrib and comp.attrib['MACH'] != machine:
            continue

        # okay, this is either a "generic" compiler section or one for this
        # machine

        for child in comp:
           mpi_match = False
           if 'MPILIB' in child.attrib:
               mpi = child.attrib['MPILIB']
               if mpi[0] == '!':
                   mpi_match = mpi[1:] != mpilib
               else:
                   mpi_match = mpi == mpilib
           else:
               mpi_match = True

           if not mpi_match:
               continue

           if child.tag == 'MPICC':
               mpicc = child.text.strip()
           elif child.tag == 'MPICXX':
               mpicxx = child.text.strip()
           elif child.tag == 'MPIFC':
               mpifc = child.text.strip()
        
    scorpio_path = '{}/compass-{}/scorpio-{}/{}/{}'.format(
        base_path, compass, scorpio, compiler, mpilib)

    esmf_path = '{}/compass-{}/esmf-{}/{}/{}'.format(
        base_path, compass, esmf, compiler, mpilib)

    esmf_branch = 'ESMF_{}'.format(esmf.replace('.', '_'))

    env_vars = []

    if 'intel' in compiler:
        esmf_compilers = '    export ESMF_COMPILER=intel'
    elif compiler == 'pgi':
        esmf_compilers = '    export ESMF_COMPILER=pgi\n' \
                         '    export ESMF_F90={}\n' \
                         '    export ESMF_CXX={}'.format(mpifc, mpicxx)
    else:
        esmf_compilers = '    export ESMF_F90={}\n' \
                         '    export ESMF_CXX={}'.format(mpifc, mpicxx)

    if 'intel' in compiler and machine == 'anvil':
        env_vars.extend(['export I_MPI_CC=icc',
                         'export I_MPI_CXX=icpc',
                         'export I_MPI_F77=ifort',
                         'export I_MPI_F90=ifort'])

    if mpilib == 'mvapich':
        esmf_comm = 'mvapich2'
        env_vars.extend(['export MV2_ENABLE_AFFINITY=0',
                         'export MV2_SHOW_CPU_BINDING=1'])
    elif mpilib == 'mpich':
        esmf_comm = 'mpich3'
    elif mpilib == 'impi':
        esmf_comm = 'intelmpi'
    else:
        esmf_comm = mpilib

    if machine == 'grizzly':
        esmf_netcdf = \
            '    export ESMF_NETCDF="split"\n' \
            '    export ESMF_NETCDF_INCLUDE=$NETCDF_C_PATH/include\n' \
            '    export ESMF_NETCDF_LIBPATH=$NETCDF_C_PATH/lib64'
    elif machine == 'badger':
        esmf_netcdf = \
            '    export ESMF_NETCDF="split"\n' \
            '    export ESMF_NETCDF_INCLUDE=$NETCDF_C_PATH/include\n' \
            '    export ESMF_NETCDF_LIBPATH=$NETCDF_C_PATH/lib64'
    else:
        esmf_netcdf = '    export ESMF_NETCDF="nc-config"'

    if 'cori' in machine:
        netcdf_paths = 'export NETCDF_C_PATH=$NETCDF_DIR\n' \
                       'export NETCDF_FORTRAN_PATH=$NETCDF_DIR\n' \
                       'export PNETCDF_PATH=$PNETCDF_DIR'
        mpas_netcdf_paths = 'export NETCDF=$NETCDF_DIR\n' \
                            'export NETCDFF=$NETCDF_DIR\n' \
                            'export PNETCDF=$PNETCDF_DIR'
    else:
        netcdf_paths = \
            'export NETCDF_C_PATH=$(dirname $(dirname $(which nc-config)))\n' \
            'export NETCDF_FORTRAN_PATH=$(dirname $(dirname $(which nf-config)))\n' \
            'export PNETCDF_PATH=$(dirname $(dirname $(which pnetcdf-config)))'
        mpas_netcdf_paths = \
            'export NETCDF=$(dirname $(dirname $(which nc-config)))\n' \
            'export NETCDFF=$(dirname $(dirname $(which nf-config)))\n' \
            'export PNETCDF=$(dirname $(dirname $(which pnetcdf-config)))'


    suffix = 'scorpio_{}_esmf_{}_{}_{}_{}'.format(
        scorpio, esmf, machine, compiler, mpilib)
    script_filename = 'build_{}.bash'.format(suffix)

    build_dir = suffix

    with open('build.template', 'r') as f:
        template = Template(f.read())

    script = template.render(modules='\n'.join(commands), scorpio=scorpio,
                             scorpio_path=scorpio_path, mpicc=mpicc,
                             mpifc=mpifc, mpicxx=mpicxx, group=group,
                             esmf_path=esmf_path, esmf_branch=esmf_branch,
                             esmf_compilers=esmf_compilers, esmf_comm=esmf_comm,
                             esmf_netcdf=esmf_netcdf, base_path=base_path,
                             build_dir=build_dir, netcdf_paths=netcdf_paths)
    print('Writing {}'.format(script_filename))
    with open(script_filename, 'w') as handle:
        handle.write(script)

    # make sure it has execute permission
    st = os.stat(script_filename)
    os.chmod(script_filename, st.st_mode | stat.S_IEXEC)


    with open('machine.template', 'r') as f:
        template = Template(f.read())

    script = template.render(conda_base=conda_base, compass_version=compass,
                             modules='\n'.join(commands),
                             scorpio_path=scorpio_path, esmf_path=esmf_path,
                             env_vars='\n'.join(env_vars),
                             netcdf_paths=mpas_netcdf_paths)
    script_filename = '../machines/{}_{}_{}.sh'.format(
        machine, compiler, mpilib)

    print('Writing {}'.format(script_filename))
    with open(script_filename, 'w') as handle:
        handle.write(script)



if __name__ == '__main__':
    main()

