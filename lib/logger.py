# A simple torch style logger
# code borrowed from OANet repository: https://github.com/zjhthu/OANet/blob/master/core/logger.py
# (C) Wei YANG 2017

from __future__ import absolute_import
import os
import sys
import numpy as np
import logging
from datetime import datetime
import coloredlogs
import git
import subprocess

__all__ = ['Logger', 'LoggerMonitor', 'savefig']

_logger = logging.getLogger()


def print_info(config, log_dir=None):
    """ Logs source code configuration

        Code adapted from RPMNet repository: https://github.com/yewzijian/RPMNet/
    """
    _logger.info('Command: {}'.format(' '.join(sys.argv)))

    # Print commit ID
    try:
        repo = git.Repo(search_parent_directories=True)
        git_sha = repo.head.object.hexsha
        git_date = datetime.fromtimestamp(repo.head.object.committed_date).strftime('%Y-%m-%d')
        git_message = repo.head.object.message
        _logger.info('Source is from Commit {} ({}): {}'.format(git_sha[:8], git_date, git_message.strip()))

        # Also create diff file in the log directory
        if log_dir is not None:
            with open(os.path.join(log_dir, 'compareHead.diff'), 'w') as fid:
                subprocess.run(['git', 'diff'], stdout=fid)

    except git.exc.InvalidGitRepositoryError:
        pass

    # Arguments
    arg_str = []

    for k_id, k_val in config.items():
        for key in k_val:
            arg_str.append("{}_{}: {}".format(k_id, key, k_val[key]))

    arg_str = ', '.join(arg_str)
    _logger.info('Arguments: {}'.format(arg_str))


def prepare_logger(config, log_path = None):
    """Creates logging directory, and installs colorlogs 
    Args:
        opt: Program arguments, should include --dev and --logdir flag.
             See get_parent_parser()
        log_path: Logging path (optional). This serves to overwrite the settings in
                 argparse namespace
    Returns:
        logger (logging.Logger)
        log_path (str): Logging directory

    Code borrowed from RPMNet repository: https://github.com/yewzijian/RPMNet/
    """

    logdir = config['misc']['log_dir']

    if log_path is None:
        datetime_str = datetime.now().strftime('%y%m%d_%H%M%S')
        log_path = os.path.join(logdir, config['method']['descriptor_module'] if config['method']['descriptor_module'] else 'No_Desc' + '_' +  
                                        config['method']['filter_module'] if config['method']['filter_module'] else 'No_Filter', datetime_str)

    else:
        log_path = os.path.join(logdir, log_path)
    
    os.makedirs(log_path, exist_ok=True)

    logger = logging.getLogger()
    coloredlogs.install(level='INFO', logger=logger)
    file_handler = logging.FileHandler('{}/log.txt'.format(log_path))
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    print_info(config, log_path)
    logger.info('Output and logs will be saved to {}'.format(log_path))

    return logger, log_path



if __name__ == '__main__':
    # # Example
    # logger = Logger('test.txt')
    # logger.set_names(['Train loss', 'Valid loss','Test loss'])

    # length = 100
    # t = np.arange(length)
    # train_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
    # valid_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
    # test_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1

    # for i in range(0, length):
    #     logger.append([train_loss[i], valid_loss[i], test_loss[i]])
    # logger.plot()

    # Example: logger monitor
    paths = {
    'resadvnet20':'/home/wyang/code/pytorch-classification/checkpoint/cifar10/resadvnet20/log.txt', 
    'resadvnet32':'/home/wyang/code/pytorch-classification/checkpoint/cifar10/resadvnet32/log.txt',
    'resadvnet44':'/home/wyang/code/pytorch-classification/checkpoint/cifar10/resadvnet44/log.txt',
    }

    field = ['Valid Acc.']

    monitor = LoggerMonitor(paths)
    monitor.plot(names=field)
    savefig('test.eps')