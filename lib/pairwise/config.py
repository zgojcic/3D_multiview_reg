import torch
from lib.descriptor import descriptor_dict
from lib.filtering import filtering_dict
from lib.pairwise import training
from lib import pairwise

def get_model(cfg, device):
    ''' 
    Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device

    Returns: 
        model (nn.Module): instance of the selected model class initialized based on the paramaters
    '''

    # Shortcuts
    sampling_type = cfg['train']['samp_type']
    correspondence_type = cfg['train']['corr_type']
    connectivity_info = None
    tgt_num_points = cfg['data']['max_num_points'] 
    st_grad = cfg['train']['st_grad_flag']

    # Get individual components
    filtering_module = get_filter(cfg, device)
    descriptor_module = get_descriptor(cfg, device)
    

    model = pairwise.PairwiseReg(descriptor_module=descriptor_module, filtering_module=filtering_module, 
                        device=device, samp_type=sampling_type, corr_type = correspondence_type,
                        connectivity_info=connectivity_info,tgt_num_points=tgt_num_points,
                        straight_through_gradient=st_grad)

    return model


def get_descriptor(cfg, device):
    descriptor_module = cfg['method']['descriptor_module']

    if descriptor_module:
        # We always keep the default parameters of FCGF
        descriptor_module = descriptor_dict[descriptor_module]().to(device)
    else:
        descriptor_module = None

    return descriptor_module


def get_filter(cfg, device):
    filter_module = cfg['method']['filter_module']
    
    if filter_module:
        filter_module = filtering_dict[filter_module](cfg).to(device)
    else:
        filter_module = None

    return filter_module


def get_trainer(cfg, model, optimizer, logger, device):
    ''' 
    Returns a pairwise registration trainer instance.

    Args:
        cfg (dict): configuration paramters
        model (nn.Module): PairwiseReg model
        optimizer (optimizer): PyTorch optimizer
        logger (logger instance): logger used to output info to the consol
        device (device): PyTorch device

    Return
        trainer (trainer instace): Trainer used to train the pairwise registration model
    '''

    trainer = training.Trainer(cfg, model, optimizer, logger, device)

    return trainer

