import torch
import torch.nn as nn
import MinkowskiEngine as ME
from lib.layers import Soft_NN, Sampler
from lib.utils import extract_overlaping_pairs, extract_mutuals, construct_filtering_input_data
from lib.pairwise import (
    config, training
)

__all__ = [
    config, training
]


class PairwiseReg(nn.Module):
    """
    Pairwise registration class.

    It cobmines a feature descriptor with a filtering network and differentiable Kabsch algorithm to estimate
    the transformation parameters of two point clouds.

    Args:
        descriptor_module (nn.Module): feature descriptor network
        filtering_module (nn.Module): filtering (outlier detection) network
        corr_type (string): type of the correspondences to be used (hard, soft, Gumble-Softmax)
        device (device): torch device 
        mutuals_flag (bool): if mutual nearest neighbors should be used

    Returns:


    """
    
    def __init__(self, descriptor_module, 
                filtering_module, device, samp_type='fps', 
                corr_type = 'soft', mutuals_flag=False, 
                connectivity_info=None, tgt_num_points=2000, 
                straight_through_gradient=True, train_descriptor=False):
        super().__init__()

        self.device = device
        self.samp_type = samp_type
        self.corr_type = corr_type
                
        self.mutuals = mutuals_flag
        self.connectivity_info = connectivity_info
        self.train_descriptor = train_descriptor
        
        self.descriptor_module = descriptor_module


        # If the descriptor module is not specified, precomputed descriptor data should be used
        if self.descriptor_module:    
            self.sampler = Sampler(samp_type=self.samp_type, targeted_num_points=tgt_num_points)
            self.feature_matching = Soft_NN(corr_type=self.corr_type, st=straight_through_gradient)
            self.precomputed_desc = False
        else:
            self.precomputed_desc = True

        self.filtering_module = filtering_module
    
    def forward(self, data):

        filtering_input, f_0, f_1 = self.compute_descriptors(input_dict=data)
        
        registration_outputs = self.filter_correspondences(filtering_input)
            
        return filtering_input, f_0, f_1, registration_outputs




    def compute_descriptors(self, input_dict):
        ''' 
        If not precomputed it infers the feature descriptors and returns the established correspondences
        together with the ground truth transformation parameters and inlier labels.

        Args:
            input_dict (dict): input data

        '''

        if not self.precomputed_desc:

            xyz_down = input_dict['pcd0'].to(self.device)

            sinput0 = ME.SparseTensor(
                input_dict['sinput0_F'], coords=input_dict['sinput0_C']).to(self.device)

            F0 = self.descriptor_module(sinput0).F

            test = torch.any(torch.isnan(F0))
            # If the FCGF descriptor should be trained with the FCGF loss (need also corresponding desc.)
            if self.train_descriptor:
                sinput1 = ME.SparseTensor(
                    input_dict['sinput1_F'], coords=input_dict['sinput1_C']).to(self.device)

                F1 = self.descriptor_module(sinput1).F 
            else:
                F1 = torch.empty(F0.shape[0], 0).to(self.device)


            # Sample the points
            xyz_batch, f_batch = self.sampler(xyz_down, F0, input_dict['pts_list'].to(self.device))

            # Build point cloud pairs for the inference
            xyz_s, xyz_t, f_s, f_t = extract_overlaping_pairs(xyz_batch, f_batch, self.connectivity_info)

            # Compute nearest neighbors in feature space
            nn_C_s_t = self.feature_matching(f_s, f_t, xyz_t) # NNs of the source points in the target point cloud
            nn_C_t_s = self.feature_matching(f_t, f_s, xyz_s) # NNs of the target points in the source point cloud
            

            if self.mutuals:
                mutuals = extract_mutuals(xyz_s, xyz_t, nn_C_s_t, nn_C_t_s)
            else:
                mutuals = None

            # Prepare the input for the filtering block
            filtering_input = construct_filtering_input_data(xyz_s, nn_C_s_t, input_dict, self.mutuals)

        else:
            filtering_input = input_dict
            F0 = None
            F1 = None

        return filtering_input, F0, F1



    def filter_correspondences(self, input_dict):
        '''
        Return the infered weights together with the pairwise rotation matrices nad translation vectors. 

        Args:
            input_dict (dict): input data

        '''

        registration_outputs = self.filtering_module(input_dict)

        return registration_outputs
