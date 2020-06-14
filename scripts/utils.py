"""
Contains the util function used in the provided scripts. 
Some of the functions are borrowed from FCGF repository (https://github.com/chrischoy/FCGF/tree/440034846e9c27e4faba44346885e4cca51e9753)
"""

import os
import re
from os import listdir
from os.path import isfile, join, isdir, splitext
import numpy as np
import torch
import MinkowskiEngine as ME
from lib.data import PrecomputedPairwiseEvalDataset, collate_fn, get_folder_list,get_file_list
from lib.utils import read_trajectory

def read_txt(path):
    """
    Reads the text file into lines.

    Args:
        path (str): path to the file

    Returns:
        lines (list): list of the lines from the input text file
    """
    with open(path) as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    
    return lines


def ensure_dir(path):
    """
    Creates dir if it does not exist.

    Args:
        path (str): path to the folder
    """
    if not os.path.exists(path):
        os.makedirs(path, mode=0o755)


def extract_features(model,
                     xyz,
                     rgb=None,
                     normal=None,
                     voxel_size=0.05,
                     device=None,
                     skip_check=False,
                     is_eval=True):
    """
    Extracts FCGF features.

    Args:
        model (FCGF model instance): model used to inferr the features
        xyz (torch tensor): coordinates of the point clouds [N,3]
        rgb (torch tensor): colors, must be in range (0,1) [N,3]
        normal (torch tensor): normal vectors, must be in range (-1,1) [N,3]
        voxel_size (float): voxel size for the generation of the saprase tensor
        device (torch device): which device to use, cuda or cpu
        skip_check (bool): if true skip rigorous check (to speed up)
        is_eval (bool): flag for evaluation mode

    Returns:
        return_coords (torch tensor): return coordinates of the points after the voxelization [m,3] (m<=n)
        features (torch tensor): per point FCGF features [m,c]
    """


    if is_eval:
        model.eval()

    if not skip_check:
        assert xyz.shape[1] == 3

        N = xyz.shape[0]
        if rgb is not None:
            assert N == len(rgb)
            assert rgb.shape[1] == 3
            if np.any(rgb > 1):
                raise ValueError('Invalid color. Color must range from [0, 1]')

        if normal is not None:
            assert N == len(normal)
            assert normal.shape[1] == 3
            if np.any(normal > 1):
                raise ValueError('Invalid normal. Normal must range from [-1, 1]')

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    feats = []
    if rgb is not None:
        # [0, 1]
        feats.append(rgb - 0.5)

    if normal is not None:
        # [-1, 1]
        feats.append(normal / 2)

    if rgb is None and normal is None:
        feats.append(np.ones((len(xyz), 1)))

    feats = np.hstack(feats)

    # Voxelize xyz and feats
    coords = np.floor(xyz / voxel_size)
    inds = ME.utils.sparse_quantize(coords, return_index=True)
    coords = coords[inds]
    # Convert to batched coords compatible with ME
    coords = ME.utils.batched_coordinates([coords])
    return_coords = xyz[inds]

    feats = feats[inds]

    feats = torch.tensor(feats, dtype=torch.float32)
    coords = torch.tensor(coords, dtype=torch.int32)

    stensor = ME.SparseTensor(coords=coords, feats=feats).to(device)

    return return_coords, model(stensor).F


def transform_point_cloud(x1, R, t, data_type='numpy'):
    """
    Transforms the point cloud using the giver transformation paramaters
    
    Args:
        x1  (np array): points of the point cloud [n,3]
        R   (np array): estimated rotation matrice [3,3]
        t   (np array): estimated translation vectors [3,1]
    Returns:
        x1_t (np array): points of the transformed point clouds [n,3]
    """
    assert data_type in ['numpy', 'torch']

    if data_type == 'numpy':
        x1_t = (np.matmul(R, x1.transpose()) + t).transpose()
        
    elif data_type =='torch':
        x1_t = (torch.matmul(R, x1.transpose(1,0)) + t).transpose(1,0)


    return x1_t


def make_pairwise_eval_data_loader(args):
    """
    Prepares the data loader for the pairwise evaluation
    
    Args:
        args (dict): configuration parameters

    Returns:
    loader (torch data loader): data loader for the evaluation data
    scene_info (dict): metadate of the scenes
    """

    dset = PrecomputedPairwiseEvalDataset(args)

    batch_size = 1 if args.mutuals else args.batch_size

    # Extract the number of examples per scene
    scene_names = get_folder_list(os.path.join(args.source_path,'correspondences'))
    scene_info = {}
    nr_examples = 0
    save_path = os.path.join(args.source_path, 'results', args.method)
    save_path += '/mutuals/' if args.mutuals else '/all/'

    for folder in scene_names:
        curr_scene_name = folder.split('/')[-1]
        if os.path.exists(os.path.join(save_path, curr_scene_name, 'traj.txt')) and not args.overwrite:
            pass
        else:
            if args.only_gt_overlaping:
                gt_pairs, gt_traj = read_trajectory(os.path.join(args.source_path,'raw_data', curr_scene_name, "gt.log"))
                examples_per_scene = len(gt_pairs)
                scene_info[curr_scene_name] = [nr_examples * 4, (nr_examples + examples_per_scene) * 4 ]
                nr_examples += examples_per_scene

            else:
                examples_per_scene = len(get_file_list(folder))
                scene_info[curr_scene_name] = [nr_examples * 4, (nr_examples + examples_per_scene) * 4 ]
                nr_examples += examples_per_scene

    # Save the total number of examples
    scene_info['nr_examples'] = nr_examples

    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False)

    return loader, scene_info