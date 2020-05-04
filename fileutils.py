import os

def remove_files(config):
    if os.path.isfile(config['generator-matrix']):
        os.remove(config['generator-matrix'])
    if os.path.isfile(config['parity-check-matrix']):
        os.remove(config['parity-check-matrix'])
    if os.path.isfile(config['encode-message']):
        os.remove(config['encode-message'])
    if os.path.isfile(config['public-key']):
        os.remove(config['public-key'])
    if os.path.isfile(config['private-key']):
        os.remove(config['private-key'])


def print_files(config, filename, matrix):
    with open(config[filename], 'a') as file:
        for item in matrix:
            for j in item:
                file.write('{0}'.format(j))
            file.write('\n')

        file.close()