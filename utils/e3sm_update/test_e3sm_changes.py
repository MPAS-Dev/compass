#!/usr/bin/env python

import argparse
import configparser
import os
import subprocess


def submit_e3sm_tests(submodule, repo_url, ocean_strings, landice_strings,
                      framework_strings, current, new, work_base, mpas_subdir,
                      make_command, setup_command, load_script):
    """
    The driver function for calling ``git bisect`` to find the first "bad"
    commit between a known "good" and "bad" E3SM commit.

    The function uses ``git bisect run`` to call
    ``utils/bisect/bisect_step.py`` repeatedly to test whether a given commit
    is good or bad.

    Parameters
    ----------
    submodule : str
        submodule to update
    repo_url : str
        the base URL of the repo for the submodule
    ocean_strings : list of str
        list of strings that identify MPAS-Ocean branches
    landice_strings : list of str
        list of strings that identify MALI branches
    framework_strings : list of str
        list of strings that identify MPAS-Framework branches
    current : str
        The hash or tag of the current E3SM-Project submodule
    new : str
        The hash or tag of the new E3SM-Project submodule
    work_base : str
        the absolute or relative path for test results (subdirectories will be
        created within this path for each git hash)
    mpas_subdir : str
        path within the E3SM worktree to the MPAS model you want to build
    make_command : str
        the make command to run to build the MPAS model
    setup_command : str
        the command to set up one or more test cases or a test suite.
        Note: the mpas model, baseline and work directories will be appended
        automatically so don't include -p, -b or -w flags
    load_script : bool
        the absolute or relative path to the load script use to activate the
        compass environment
    """
    print('\n')
    print(72*'-')
    print(f'Testing to update {submodule} submodule')
    print(72*'-')

    load_script = os.path.abspath(load_script)

    commands = f'git submodule update --init'
    print_and_run(commands)

    current_submodule = f'{submodule}-current'
    new_submodule = f'{submodule}-new'

    print(f'Setting up current submodule\n')

    setup_worktree(submodule, worktree=current_submodule, hash=current)

    print(f'\nSetting up new submodule\n')

    setup_worktree(submodule, worktree=new_submodule, hash=new)

    # get a list of merges between the current and new hashes
    commands = f'cd {new_submodule}; ' \
               f'git log --oneline --first-parent --ancestry-path ' \
               f'{current}..{new}'

    all_commits = print_and_run(commands, get_output=True)

    pull_requests = []
    for line in all_commits.split('\n')[::-1]:
        component = None
        if any([string in line for string in ocean_strings]):
            component = 'ocn'
        elif any([string in line for string in landice_strings]):
            component = 'mali'
        elif any([string in line for string in framework_strings]):
            component = 'fwk'
        if component is not None:
            if '#' in line:
                hash = line.split(' ')[0]
                end = line.split('#')[-1].replace('(', ' ').replace(')', ' ')
                pull_request = end.split()[0]
                index = len(pull_requests)
                worktree = f'{index + 1:02d}_{pull_request}'
                pull_requests.append({'hash': hash,
                                      'component': component,
                                      'pull_request': pull_request,
                                      'worktree': worktree})
            else:
                print(f'Warning: skipping commit with no "#" for PR:\n{line}')

    print('Merge commits of interest: hash PR (component):')
    print(f'00: {current} {current_submodule} (current)')
    for index, data in enumerate(pull_requests):
        hash = data['hash']
        pull_request = data['pull_request']
        component = data['component']
        print(f'{index+1:02d}: {hash} {pull_request} ({component})')
    print(f'{len(pull_requests)+1:02d}: {new} {new_submodule} (new)')
    print('\n')

    print(f'Setting up worktrees of all commits of interest\n')

    for index, data in enumerate(pull_requests):
        hash = data['hash']
        worktree = data['worktree']
        setup_worktree(submodule, worktree=worktree, hash=hash)

    print(f'Building each worktree and submitting comparison tests\n')

    print('00: current\n')
    baseline = f'{work_base}/00_current'
    build_model(load_script=load_script, worktree=current_submodule,
                mpas_subdir=mpas_subdir, make_command=make_command)
    setup_and_submit(load_script=load_script, setup_command=setup_command,
                     worktree=current_submodule, mpas_subdir=mpas_subdir,
                     workdir=baseline)
    previous = 'current'

    for index, data in enumerate(pull_requests):
        worktree = data['worktree']
        pull_request = data['pull_request']
        workdir = f'{work_base}/{index+1:02d}_{pull_request}_{previous}'
        print(f'{index+1:02d}: {pull_request}\n')
        build_model(load_script=load_script, worktree=worktree,
                    mpas_subdir=mpas_subdir, make_command=make_command)
        setup_and_submit(load_script=load_script, setup_command=setup_command,
                         worktree=worktree,
                         mpas_subdir=mpas_subdir, workdir=workdir,
                         baseline=baseline)
        baseline = workdir
        previous = pull_request

    index = len(pull_requests)
    print(f'{index+1:02d}: new\n')
    workdir = f'{work_base}/{index+1:02d}_new_{previous}'
    build_model(load_script=load_script, worktree=new_submodule,
                mpas_subdir=mpas_subdir, make_command=make_command)
    setup_and_submit(load_script=load_script, setup_command=setup_command,
                     worktree=new_submodule, mpas_subdir=mpas_subdir,
                     workdir=workdir, baseline=baseline)

    print_pr_description(submodule, repo_url, current, new, pull_requests)


def setup_worktree(submodule, worktree, hash):
    if not os.path.exists(worktree):
        commands = f'cd {submodule}; ' \
                   f'git worktree add ../{worktree}'
        print_and_run(commands)

    commands = f'cd {worktree}; ' \
               f'git reset --hard {hash}; ' \
               f'git submodule update --init --recursive >& submodule.log'
    print_and_run(commands)


def build_model(load_script, worktree, mpas_subdir, make_command):
    commands = f'source {load_script}; ' \
               f'cd {worktree}/{mpas_subdir}; ' \
               f'{make_command} &> make.log'
    print_and_run(commands)


def setup_and_submit(load_script, setup_command, worktree, mpas_subdir,
                     workdir, baseline=None):

    if ' -t ' in setup_command:
        split = setup_command.split()
        index = split.index('-t')
        suite = split[index+1]
    elif '--test_suite' in setup_command:
        split = setup_command.split()
        index = split.index('--test_suite')
        suite = split[index+1]
    else:
        suite = 'custom'

    full_setup = f'{setup_command} -p {worktree}/{mpas_subdir} -w {workdir}'
    if baseline is not None:
        full_setup = f'{full_setup} -b {baseline}'

    commands = f'source {load_script}; ' \
               f'{full_setup}'
    print_and_run(commands)

    commands = f'cd {workdir}; ' \
               f'sbatch job_script.{suite}.sh'
    print_and_run(commands)


def print_and_run(commands, get_output=False):
    print('\nRunning:')
    print_commands = commands.replace('; ', '\n  ')
    print(f'  {print_commands}\n\n')
    if get_output:
        output = subprocess.check_output(commands, shell=True)
        output = output.decode('utf-8').strip('\n')
    else:
        subprocess.run(commands, check=True, shell=True)
        output = None

    return output


def print_pr_description(submodule, repo_url, current, new, pull_requests):
    print('')
    print(72*'-')
    print(f'Pull Request Description Text')
    print(72*'-')
    print('')

    print(f'This merge updates the {submodule} submodule from '
          f'[{current}]({repo_url}/tree/{current}) '
          f'to [{new}]({repo_url}/tree/{new}).\n')

    print('This update includes the following MPAS-Ocean and MPAS-Frameworks '
          'PRs (check mark indicates bit-for-bit with previous PR in the '
          'list):')
    for data in pull_requests:
        pull_request = data['pull_request']
        component = data['component']
        print(f'- [ ]  ({component}) '
              f'{repo_url}/pull/{pull_request}')
    print('\n')


def string_to_list(string):
    return string.replace(',', ' ').split()


def main():
    parser = argparse.ArgumentParser(
        description='Test changes in E3SM before an update to the E3SM-Project '
                    'submodule')
    parser.add_argument("-f", "--config_file", dest="config_file",
                        required=True,
                        help="Configuration file",
                        metavar="FILE")

    args = parser.parse_args()

    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation())
    config.read(args.config_file)

    section = config['e3sm_update']

    submit_e3sm_tests(
        submodule=section['submodule'], repo_url=section['repo_url'],
        ocean_strings=string_to_list(section['ocean_strings']),
        landice_strings=string_to_list(section['landice_strings']),
        framework_strings=string_to_list(section['framework_strings']),
        current=section['current'], new=section['new'],
        work_base=section['work_base'], mpas_subdir=section['mpas_subdir'],
        make_command=section['make_command'],
        setup_command=section['setup_command'],
        load_script=section['load_script'])


if __name__ == '__main__':
    main()
