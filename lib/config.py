
from lib import pairwise
import torch

method_dict = {
    'pairwise': pairwise,
}


def get_model(cfg):
    ''' 
    Gets the model instance based on the input paramters.

    Args:
        cfg (dict): config dictionary
    
    Returns:
        model (nn.Module): torch model initialized with the input params
    '''

    method = cfg['method']['task']
    device = torch.device('cuda' if (torch.cuda.is_available() and cfg['misc']['use_gpu']) else 'cpu') 

    model = method_dict[method].config.get_model(cfg, device=device)

    return model

def get_trainer(cfg, model, optimizer, logger):
    ''' 
    Returns a trainer instance.

    Args:
        cfg (dict): config dictionary
        model (nn.Module): the model used for training
        optimizer (optimizer): pytorch optimizer
        logger (logger instance): logger used to output info to the consol

    Returns:
        trainer (trainer instance): trainer instance used to train the network
    '''
    
    method = cfg['method']['task']
    device = torch.device('cuda' if (torch.cuda.is_available() and cfg['misc']['use_gpu']) else 'cpu') 

    trainer = method_dict[method].config.get_trainer(cfg, model, optimizer, logger, device)

    return trainer
