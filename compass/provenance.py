import os
import sys
import subprocess


def write(work_dir, testcases):
    try:
        args = ['git', 'describe', '--tags', '--dirty', '--always']
        git_version = subprocess.check_output(args).decode('utf-8')
        git_version = git_version.strip('\n')
    except subprocess.CalledProcessError:
        git_version = None

    try:
        args = ['conda', 'list']
        conda_list = subprocess.check_output(args).decode('utf-8')
    except subprocess.CalledProcessError:
        conda_list = None

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
    if git_version is not None:
        provenance_file.write('git_version: {}\n\n'.format(git_version))
    provenance_file.write('command: {}\n\n'.format(calling_command))
    provenance_file.write('testcases:\n')

    for module, test in testcases.items():
        prefix = '  '
        lines = list()
        for key in ['path', 'description', 'name', 'core',
                    'configuration', 'subdir', 'module']:
            key_string = '{}: '.format(key).ljust(15)
            lines.append('{}{}{}'.format(prefix, key_string, test[key]))
        lines.append('{}steps:'.format(prefix))
        for step in test['steps']:
            lines.append('{} - {}'.format(prefix, step))
        lines.append('')
        print_string = '\n'.join(lines)

        provenance_file.write('{}\n'.format(print_string))

    if conda_list is not None:
        provenance_file.write('conda list:\n')
        provenance_file.write('{}\n'.format(conda_list))

    provenance_file.write('**************************************************'
                          '*********************\n')
    provenance_file.close()
