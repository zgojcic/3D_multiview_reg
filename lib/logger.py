import os
import sys
import numpy as np
import logging
from datetime import datetime
import coloredlogs
import git
import subprocess


_logger = logging.getLogger()


def print_info(cfg, log_dir=None):
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

    for k_id, k_val in cfg.items():
        for key in k_val:
            arg_str.append("{}_{}: {}".format(k_id, key, k_val[key]))

    arg_str = ', '.join(arg_str)
    _logger.info('Arguments: {}'.format(arg_str))


def prepare_logger(cfg, log_path = None):
    """Creates logging directory, and installs colorlogs 
    Args:
        cfg (dict): config parmaters
        log_path (str): Logging path (optional). This serves to overwrite the settings in cfg

    Returns:
        logger (logging.Logger): logger instance
        log_path (str): Logging directory

    Code borrowed from RPMNet repository: https://github.com/yewzijian/RPMNet/
    """

    logdir = cfg['misc']['log_dir']

    if log_path is None:
        datetime_str = datetime.now().strftime('%y%m%d_%H%M%S')
        log_path = os.path.join(logdir, cfg['method']['descriptor_module'] if cfg['method']['descriptor_module'] else 'No_Desc' + '_' +  
                                        cfg['method']['filter_module'] if cfg['method']['filter_module'] else 'No_Filter', datetime_str)

    os.makedirs(log_path, exist_ok=True)

    logger = logging.getLogger()
    coloredlogs.install(level='INFO', logger=logger)
    file_handler = logging.FileHandler('{}/log.txt'.format(log_path))
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    print_info(cfg, log_path)
    logger.info('Output and logs will be saved to {}'.format(log_path))

    return logger, log_path


