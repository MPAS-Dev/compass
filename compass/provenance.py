import os
import subprocess
import sys


def write(work_dir, test_cases, config=None):
    """
    Write a file with provenance, such as the git version, conda packages,
    command, and test cases, to the work directory

    Parameters
    ----------
    work_dir : str
        The path to the work directory where the test cases will be set up

    test_cases : dict
        A dictionary describing all of the test cases and their steps

    config : compass.config.CompassConfigParser
        Configuration options for this test case, a combination of user configs
        and the defaults for the machine and MPAS core
    """
    compass_git_version = None
    if os.path.exists('.git'):
        try:
            args = ['git', 'describe', '--tags', '--dirty', '--always']
            compass_git_version = subprocess.check_output(args).decode('utf-8')
            compass_git_version = compass_git_version.strip('\n')
        except subprocess.CalledProcessError:
            pass

    if config is None:
        # this is a call to clean and we don't need to document the MPAS
        # version
        mpas_git_version = None
    else:
        mpas_git_version = _get_mpas_git_version(config)

    package_list_name = 'pixi list'
    try:
        package_list = subprocess.check_output(
            ['pixi', 'list']
        ).decode('utf-8')
    except (subprocess.CalledProcessError, FileNotFoundError):
        package_list = None

    calling_command = ' '.join(sys.argv)

    try:
        os.makedirs(work_dir)
    except OSError:
        pass

    provenance_path = '{}/provenance'.format(work_dir)
    if os.path.exists(provenance_path):
        provenance_file = open(provenance_path, 'a')
        provenance_file.write('\n')
    else:
        provenance_file = open(provenance_path, 'w')

    provenance_file.write('**************************************************'
                          '*********************\n')
    if compass_git_version is not None:
        provenance_file.write('compass git version: {}\n\n'.format(
            compass_git_version))
    if mpas_git_version is not None:
        provenance_file.write('MPAS git version: {}\n\n'.format(
            mpas_git_version))
    provenance_file.write('command: {}\n\n'.format(calling_command))
    provenance_file.write('test cases:\n')

    for path, test_case in test_cases.items():
        prefix = '  '
        lines = list()
        to_print = {'path': test_case.path,
                    'name': test_case.name,
                    'MPAS core': test_case.mpas_core.name,
                    'test group': test_case.test_group.name,
                    'subdir': test_case.subdir}
        for key in to_print:
            key_string = '{}: '.format(key).ljust(15)
            lines.append('{}{}{}'.format(prefix, key_string, to_print[key]))
        lines.append('{}steps:'.format(prefix))
        for step in test_case.steps.values():
            if step.name == step.subdir:
                lines.append('{} - {}'.format(prefix, step.name))
            else:
                lines.append('{} - {}: {}'.format(prefix, step.name,
                                                  step.subdir))
        lines.append('')
        print_string = '\n'.join(lines)

        provenance_file.write('{}\n'.format(print_string))

    if package_list is not None and package_list_name is not None:
        provenance_file.write(f'{package_list_name}:\n')
        provenance_file.write('{}\n'.format(package_list))

    provenance_file.write('**************************************************'
                          '*********************\n')
    provenance_file.close()


def _get_mpas_git_version(config):

    mpas_model_path = config.get('paths', 'mpas_model')

    if not os.path.exists(mpas_model_path):
        return None

    cwd = os.getcwd()
    os.chdir(mpas_model_path)

    try:
        args = ['git', 'describe', '--tags', '--dirty', '--always']
        mpas_git_version = subprocess.check_output(args).decode('utf-8')
        mpas_git_version = mpas_git_version.strip('\n')
    except subprocess.CalledProcessError:
        mpas_git_version = None
    os.chdir(cwd)

    return mpas_git_version
