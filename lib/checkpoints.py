"""
A collection of function used for saving and loading the model weights. 
Based on the Occupany Networks repository: https://github.com/autonomousvision/occupancy_networks

If you use in your project pelase consider also citing: http://www.cvlibs.net/publications/Mescheder2019CVPR.pdf
"""

import os
import urllib
import torch
from torch.utils import model_zoo

class CheckpointIO(object):
    ''' CheckpointIO class.
    It handles saving and loading checkpoints.
    Args:
        checkpoint_dir (str): path where checkpoints are saved
    '''
    def __init__(self, checkpoint_dir='./chkpts', initialize_from=None,
                 initialization_file_name='model_best.pt', **kwargs):
        self.module_dict = kwargs
        self.checkpoint_dir = checkpoint_dir
        self.initialize_from = initialize_from
        self.initialization_file_name = initialization_file_name
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def register_modules(self, **kwargs):
        ''' Registers modules in current module dictionary.
        '''
        self.module_dict.update(kwargs)

    def save(self, filename, **kwargs):
        ''' Saves the current module dictionary.
        Args:
            filename (str): name of output file
        '''
        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        outdict = kwargs
        for k, v in self.module_dict.items():
            outdict[k] = v.state_dict()
        torch.save(outdict, filename)

    def load(self, filename):
        '''Loads a module dictionary from local file or url.
        Args:
            filename (str): name of saved module dictionary
        '''
        if is_url(filename):
            return self.load_url(filename)
        else:
            return self.load_file(filename)

    def load_file(self, filename):
        '''Loads a module dictionary from file.
        Args:
            filename (str): name of saved module dictionary
        '''

        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        if os.path.exists(filename):
            print(filename)
            print('=> Loading checkpoint from local file...')
            state_dict = torch.load(filename)
            scalars = self.parse_state_dict(state_dict)
            return scalars
        else:
            if self.initialize_from is not None:
                self.initialize_weights()
            raise FileExistsError

    def load_url(self, url):
        '''Load a module dictionary from url.
        Args:
            url (str): url to saved model
        '''
        print(url)
        print('=> Loading checkpoint from url...')
        state_dict = model_zoo.load_url(url, progress=True)
        scalars = self.parse_state_dict(state_dict)
        return scalars

    def parse_state_dict(self, state_dict):
        '''Parse state_dict of model and return scalars.
        Args:
            state_dict (dict): State dict of model
    '''

        for k, v in self.module_dict.items():
            if k in state_dict:
                v.load_state_dict(state_dict[k])
            else:
                print('Warning: Could not find %s in checkpoint!' % k)
        scalars = {k: v for k, v in state_dict.items()
                    if k not in self.module_dict}
        return scalars

    def initialize_weights(self):
        ''' Initializes the model weights from another model file.
        '''

        print('Intializing weights from model %s' % self.initialize_from)
        filename_in = os.path.join(
                    self.initialize_from, self.initialization_file_name)

        model_state_dict = self.module_dict.get('model').state_dict()
        model_dict = self.module_dict.get('model').state_dict()
        model_keys = set([k for (k, v) in model_dict.items()])

        init_model_dict = torch.load(filename_in)['model']
        init_model_k = set([k for (k, v) in init_model_dict.items()])

        for k in model_keys:
            if ((k in init_model_k) and (model_state_dict[k].shape ==
                                         init_model_dict[k].shape)):
                model_state_dict[k] = init_model_dict[k]
        self.module_dict.get('model').load_state_dict(model_state_dict)


def is_url(url):
    ''' Checks if input is url.'''
    scheme = urllib.parse.urlparse(url).scheme
    return scheme in ('http', 'https')