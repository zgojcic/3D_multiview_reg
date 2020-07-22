"""
This script is used to showcase the pairwise registration. Given two point clouds it 
estimates the relative transformation parameters that align the given two point clouds. 

"""

import sys
import os 
import logging 
import argparse
import coloredlogs
import time
import torch
import copy
cwd = os.getcwd()
sys.path.append(cwd)

import numpy as np
import open3d as o3d
import MinkowskiEngine as ME

import lib.config as config
from lib.utils import load_config, extract_overlaping_pairs, transformation_residuals, write_trajectory
from lib.checkpoints import CheckpointIO


# Set the random seeds for reproducibility
np.random.seed(41)
torch.manual_seed(41)
if torch.cuda.is_available():
    torch.cuda.manual_seed(41)


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def to_tensor(x):
    """
    Maps the numpy arrays to torch tenors. In torch tenor is used as input
    it simply returns it.

    Args:
    x (numpy array): numpy array input data
    
    Returns:
    x (torch tensor): input converted to a torch tensor
    """

    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    else:
        raise ValueError(f'Can not convert to torch tensor, {x}')

def prepare_data(files, voxel_size, device='cuda'):
    """
    Loads the data and prepares the input for the pairwise registration demo. 

    Args:
        files (list): paths to the point cloud files
    """

    feats = []
    xyz = []
    coords = []
    n_pts = []

    for pc_file in files:
        pcd0 = o3d.io.read_point_cloud(pc_file)
        xyz0 = np.array(pcd0.points)

        # Voxelization
        sel0 = ME.utils.sparse_quantize(xyz0 / voxel_size, return_index=True)

        # Make point clouds using voxelized points
        xyz0 = xyz0[sel0,:]

        # Get features
        npts0 = xyz0.shape[0]

        xyz.append(to_tensor(xyz0))
        n_pts.append(npts0)
        feats.append(np.ones((npts0, 1)))
        coords.append(np.floor(xyz0 / voxel_size))


    coords_batch0, feats_batch0 = ME.utils.sparse_collate(coords, feats)

    data = {'pcd0': torch.cat(xyz, 0).float(), 'sinput0_C': coords_batch0, 
            'sinput0_F': feats_batch0.float(), 'pts_list': torch.tensor(n_pts)}
    
    return data


def main(cfg, args, logger):
    """
    Main function of this software. After preparing the model, carry out the pairwise point cloud registration.

    Args:
        cfg (dict): current configuration paramaters
    """

    # Get model
    model = config.get_model(cfg)
    model.eval()
    
    # Load pre-trained model if existing
    kwargs = {
    'model': model
    }

    checkpoint_io = CheckpointIO('', initialize_from='./pretrained/',
                                 initialization_file_name=args.model, **kwargs)
    
    try:
        load_dict = checkpoint_io.load()
    except FileExistsError:
        load_dict = dict()

    # Print model parameters and model graph
    nparameters = sum(p.numel() for p in model.parameters())
    logger.info('Total number of model parameters: {}'.format(nparameters))

    # Prepare the output file name
    target_base = './data/demo/pairwise/results'
    id_0 = args.source_pc.split(os.sep)[-1].split('_')[-1].split('.')[0]
    id_1 = args.target_pc.split(os.sep)[-1].split('_')[-1].split('.')[0]
    metadata = np.array([[id_0, id_1, 'True']])

    if not os.path.exists(target_base):
        os.makedirs(target_base)

    target_path = os.path.join(target_base, 'est_T.log')

    with torch.no_grad():
        # Load the point clouds and prepare the input
        start_time_pipeline = time.time()

        point_cloud_files = [args.source_pc, args.target_pc]
        data = prepare_data(point_cloud_files, cfg['misc']['voxel_size'])

        # Extract the descriptors perform the NN search and prepare the data for the reg. blocks
        start_time_features = time.time()
        filtering_data, _, _ = model.compute_descriptors(data)
        end_time_features = time.time()

        if args.verbose:
            logger.info('Feature computation and sampling took {:.3f}s'.format(end_time_features - start_time_features))

        # Filter the putative correspondences and estimate the relative transformation parameters
        start_time_filtering = time.time()
        est_data = model.filter_correspondences(filtering_data)
        end_time_filtering = time.time()
        
        if args.verbose:
            logger.info('Filtering the correspondences and estimation of paramaters took {:.3f}s'.format(end_time_filtering - start_time_filtering))

        est_T = np.eye(4)
        est_T[0:3,0:3] = est_data['rot_est'][-1].cpu().numpy()
        est_T[0:3,3:4] = est_data['trans_est'][-1].cpu().numpy()
        
        end_time_pipeline = time.time()
        
        # Save the results
        write_trajectory(np.expand_dims(est_T,0), metadata, target_path)

        if args.verbose:
            logger.info('Estimation of the pairwise transformation parameters completed in {:.3f}s'.format(end_time_pipeline - start_time_pipeline))
            logger.info('Estimated parameters were saved in {}.'.format(target_path))

        if args.visualize:
            pcd_1 = o3d.io.read_point_cloud(args.source_pc)
            pcd_2 = o3d.io.read_point_cloud(args.target_pc)

            # First plot both point clouds in their original reference frame
            draw_registration_result(pcd_1, pcd_2, np.identity(4))

            # Plot the point clouds after applying the estimated transformation parameters
            draw_registration_result(pcd_1, pcd_2, est_T)

if __name__ == "__main__":
    
    # Initialize the logger
    logger = logging.getLogger()
    coloredlogs.install(level='INFO', logger=logger)
    logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')


    parser = argparse.ArgumentParser()
    parser.add_argument('config', default='./configs/pairwise_registration/demo/config.yaml', type=str, help='config file')
    parser.add_argument('--source_pc', default='./data/demo/pairwise/raw_data/cloud_bin_0.ply', type=str, help='source point cloud')
    parser.add_argument('--target_pc', default='./data/demo/pairwise/raw_data/cloud_bin_1.ply', type=str, help='target point cloud')
    parser.add_argument('--model', default='pairwise_reg.pt', type=str, help= 'Name of the pretrained model.')
    parser.add_argument('--verbose', action='store_true', help='Write out the intermediate results and timings')
    parser.add_argument('--visualize', action='store_true', help='Visualize the point cloud and the results.')

    args = parser.parse_args()
    cfg = load_config(args.config)

    # Ensure that all the necessary inputs were provided
    assert args.source_pc is not None, "Source pc path argument is missing!"
    assert args.target_pc is not None, "Target pc path argument is missing!"
    assert args.model is not None, "Model path argument is missing!"



    # Loging the settings
    logger.info("Torch version: {}".format(torch.__version__))
    logger.info("Demo pairwise registration started.")

    main(cfg, args, logger)
