"""
Source code used to extract FCGF features, pairwise correspondences and training that can be used to train our network
without computing the FCGF descriptors on the fly. This way of training greatly eases the sampling of the point cloud pairs and
can at least be used to pretrain the filtering network, confidence estimation block and transformation synchronization.

"""

import argparse
import os
import open3d as o3d
import torch 
import logging
import numpy as np
import sys
import coloredlogs
import torch
import easydict
import multiprocessing as mp
import sys

cwd = os.getcwd()
sys.path.append(cwd)

from sklearn.neighbors import NearestNeighbors
from functools import partial

from lib.descriptor.fcgf import FCGFNet
from scripts.utils import extract_features, transform_point_cloud
from lib.utils import compute_overlap_ratio, load_point_cloud, ensure_dir, get_file_list, get_folder_list

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)



def extract_features_batch(model, source_path, target_path, dataset, voxel_size, device):
    """
    Extracts the per point features in the FCGF feature space and saves them to the predefined path

    Args:
        model (FCGF model instance): model used to inferr the descriptors
        source_path (str): path to the raw files
        target path (str): where to save the extracted data
        dataset (float): name of the dataset
        voxel_size (float): voxel sized used to create the sparse tensor
        device (pytorch device): cuda or cpu
    """

    source_path = os.path.join(source_path,dataset,'raw_data')
    target_path = os.path.join(target_path,dataset,'features')

    ensure_dir(target_path)

    folders = get_folder_list(source_path)

    assert len(folders) > 0, 'Could not find {} folders under {}'.format(dataset, source_path)

    logging.info(folders)
    list_file = os.path.join(target_path, 'list.txt')
    f = open(list_file, 'w')
    model.eval()

    for fo in folders:
        scene_name = fo.split()
        files = get_file_list(fo, '.ply')
        fo_base = os.path.basename(fo)
        ensure_dir(os.path.join(target_path, fo_base))

        f.write('%s %d\n' % (fo_base, len(files)))
        for i, fi in enumerate(files):
            save_fn = '%s_%03d' % (fo_base, i)
            if os.path.exists(os.path.join(target_path, fo_base, save_fn + '.npz')):
                print('Correspondence file already exits moving to the next example.')
            else:
                # Extract features from a file
                pcd = o3d.io.read_point_cloud(fi)
                
                if i % 100 == 0:
                    logging.info(f'{i} / {len(files)}: {save_fn}')

                xyz_down, feature = extract_features(
                    model,
                    xyz=np.array(pcd.points),
                    rgb=None,
                    normal=None,
                    voxel_size=voxel_size,
                    device=device,
                    skip_check=True)

                np.savez_compressed(
                    os.path.join(target_path, fo_base, save_fn),
                    points=np.array(pcd.points),
                    xyz=xyz_down,
                    feature=feature.detach().cpu().numpy())

    f.close()

def extract_correspondences(dataset,source_path,target_path, n_correspondences):
    """
    Prepares the arguments and runs the correspondence extration in parallel mode

    Args:
        dataset (str): name of the dataset
        source_path (str): path to the raw files
        target_path (str): path to where the extracted data will be saved
        n_correspondences (int): number of points to sample

    """
    source_path = os.path.join(source_path,dataset,'raw_data')


    scene_paths = get_folder_list(source_path)
    idx = list(range(len(scene_paths)))


    pool = mp.Pool(processes=6)
    func = partial(run_correspondence_extraction, dataset, source_path, target_path, n_correspondences)
    pool.map(func, idx)
    pool.close()
    pool.join()


def run_correspondence_extraction(dataset,source_path, target_path, n_correspondences, idx):
    """
    Computes the correspondences in the FCGF space together with the mutuals and ratios side information

    Args:
        dataset (str): name of the dataset
        source_path (str): path to the raw data
        target_path (str): path to where the extracted data will be saved
        n_correspondences (int): number of points to sample
        idx (int): index of the scene, used for parallel processing

    """

    # Initialize all the paths
    features_path = os.path.join(target_path,dataset,'features')
    target_path = os.path.join(target_path,dataset,'correspondences')

    fo = get_folder_list(source_path)[idx]
    fo_base = os.path.basename(fo)
    files = get_file_list(os.path.join(features_path, fo_base), '.npz')

    ensure_dir(os.path.join(target_path, fo_base))      

    # Loop over all fragment pairs and compute the training data
    for idx_1 in range(len(files)):
        for idx_2 in range(idx_1+1, len(files)):
            if os.path.exists(os.path.join(target_path, fo_base,'{}_{}_{}.npz'.format(fo_base,str(idx_1).zfill(3), str(idx_2).zfill(3)))):
                logging.info('Correspondence file already exits moving to the next example.')

            else:
                pc_1_data = np.load(os.path.join(features_path, fo_base, fo_base + '_{}.npz'.format(str(idx_1).zfill(3))))
                pc_1_features = pc_1_data['feature']
                pc_1_keypoints = pc_1_data['xyz']

                pc_2_data = np.load(os.path.join(features_path, fo_base, fo_base + '_{}.npz'.format(str(idx_2).zfill(3))))
                pc_2_features = pc_2_data['feature']
                pc_2_keypoints = pc_2_data['xyz']

                # Sample with replacement if less then n_correspondences points are in the point cloud
                if pc_1_features.shape[0] >= n_correspondences:
                    inds_1 = np.random.choice(pc_1_features.shape[0], n_correspondences, replace=False)
                else:
                    inds_1 = np.random.choice(pc_1_features.shape[0], n_correspondences, replace=True)

                if pc_2_features.shape[0] >=  n_correspondences:
                    inds_2 = np.random.choice(pc_2_features.shape[0], n_correspondences, replace=False)
                else:
                    inds_2 = np.random.choice(pc_2_features.shape[0], n_correspondences, replace=True)


                pc_1_features = pc_1_features[inds_1,:]
                pc_2_features = pc_2_features[inds_2, :]
                pc_1_key = pc_1_keypoints[inds_1,:]
                pc_2_key = pc_2_keypoints[inds_2,:]

                # find the correspondence using nearest neighbor search in the feature space (two way)
                nn_search = NearestNeighbors(n_neighbors=1, metric='minkowski', p=2)
                nn_search.fit(pc_2_features)
                nn_dists, nn_indices = nn_search.kneighbors(X=pc_1_features, n_neighbors=2, return_distance=True)
    

                nn_search.fit(pc_1_features)
                nn_dists_1, nn_indices_1 = nn_search.kneighbors(X=pc_2_features, n_neighbors=2, return_distance=True)

                ol_nn_ids = np.where((nn_indices[nn_indices_1[:, 0], 0] - np.arange(pc_1_features.shape[0])) == 0)[0]

                # Initialize mutuals and ratios
                mutuals = np.zeros((n_correspondences, 1))
                mutuals[ol_nn_ids] = 1
                ratios = nn_dists[:, 0] / nn_dists[:, 1]

                # Concatenate the correspondence coordinates
                xs = np.concatenate((pc_1_key[nn_indices_1[:, 0], :], pc_2_key), axis=1)

                np.savez_compressed(
                os.path.join(target_path, fo_base, '{}_{}_{}.npz'.format(fo_base, str(idx_1).zfill(3), str(idx_2).zfill(3))),
                x=xs,
                mutuals=mutuals,
                ratios=ratios)


def extract_precomputed_training_data(dataset, source_path, target_path, voxel_size, inlier_threshold):
    """
    Prepares the data for training the filtering networks with precomputed correspondences (without FCGF descriptor) 

    Args:
        dataset (str): name of the dataset
        source_path (str): path to the raw data
        target_path (str): path to where the extracted data will be saved
        voxel_size (float): voxel size that was used to compute the features
        inlier_threshold (float): threshold to determine if a correspondence is an inlier or outlier
    """
    source_path = os.path.join(source_path,dataset,'raw_data')
    features_path = os.path.join(target_path,dataset,'features')
    correspondence_path = os.path.join(target_path,dataset,'correspondences')
    target_path = os.path.join(target_path,dataset,'training_data')

    
    ensure_dir(target_path)

    # Check that the GT global transformation matrices are available and that the FCGF features are computed
    folders = get_folder_list(source_path)

    assert len(folders) > 0, 'Could not find {} folders under {}'.format(dataset, source_path)

    logging.info('Found {} scenes from the {} dataset!'.format(len(folders),dataset))

    for fo in folders:

        scene_name = fo.split()
        fo_base = os.path.basename(fo)
        ensure_dir(os.path.join(target_path, fo_base))
        
        pc_files = get_file_list(fo, '.ply')
        trans_files = get_file_list(fo, '.txt')
        assert len(pc_files) <= len(trans_files), 'The number of point cloud files does not equal the number of GT trans parameters!'
        
        feat_files = get_file_list(os.path.join(features_path,fo_base), '.npz')
        assert len(pc_files) == len(feat_files), 'Features for scene {} are either not computed or some are missing!'.format(fo_base)

        coor_files = get_file_list(os.path.join(correspondence_path,fo_base), '.npz')

        assert len(coor_files) == int((len(feat_files) * (len(feat_files)-1))/2), 'Correspondence files for the scene {} are missing. First run the correspondence extraction!'.format(fo_base)

        # Loop over all fragment pairs and compute the training data
        for idx_1 in range(len(pc_files)):
            for idx_2 in range(idx_1+1, len(pc_files)):
                if os.path.exists(os.path.join(target_path, fo_base,'{}_{}_{}.npz'.format(fo_base,str(idx_1).zfill(3), str(idx_2).zfill(3)))):
                    logging.info('Training file already exits moving to the next example.')

        
                data = np.load(os.path.join(correspondence_path, fo_base,'{}_{}_{}.npz'.format(fo_base,str(idx_1).zfill(3), str(idx_2).zfill(3))))
                xs = data['xs']
                mutuals = data['mutuals']
                ratios = data['ratios']

                # Get the GT transformation parameters
                t_1 = np.genfromtxt(os.path.join(source_path, fo_base, 'cloud_bin_{}.info.txt'.format(idx_1)), skip_header=1)
                t_2 = np.genfromtxt(os.path.join(source_path, fo_base, 'cloud_bin_{}.info.txt'.format(idx_2)), skip_header=1)


                # Get the GT transformation parameters
                pc_1 = load_point_cloud(os.path.join(source_path,fo_base, 'cloud_bin_{}.ply'.format(idx_1)), data_type='numpy')
                pc_2 = load_point_cloud(os.path.join(source_path,fo_base, 'cloud_bin_{}.ply'.format(idx_2)), data_type='numpy')

                pc_1_tr = transform_point_cloud(pc_1, t_1[0:3,0:3], t_1[0:3,3].reshape(-1,1))
                pc_2_tr = transform_point_cloud(pc_2, t_2[0:3,0:3], t_2[0:3,3].reshape(-1,1))

                overlap_ratio = compute_overlap_ratio(pc_1_tr, pc_2_tr, np.eye(4), method = 'FCGF', voxel_size=voxel_size)

                # Estimate pairwise transformation parameters
                t_3 = np.matmul(np.linalg.inv(t_2), t_1)

                r_matrix = t_3[0:3, 0:3]
                t_vector = t_3[0:3, 3]

                # Transform the keypoints of the first point cloud
                pc_1_key_tr = transform_point_cloud(xs[:,0:3], r_matrix, t_vector.reshape(-1,1))
                
                # Compute the residuals after the transformation
                y_s = np.sqrt(np.sum(np.square(pc_1_key_tr - xs[:,3:6]), axis=1))

                # Inlier percentage
                inlier_ratio = np.where(y_s < inlier_threshold)[0].shape[0] / y_s.shape[0]
                inlier_ratio_mutuals = np.where(y_s[mutuals.astype(bool).reshape(-1)] < inlier_threshold)[0].shape[0] / np.sum(mutuals)

                np.savez_compressed(os.path.join(target_path,fo_base,'cloud_{}_{}.npz'.format(str(idx_1).zfill(3), str(idx_2).zfill(3))),
                            R=r_matrix, t=t_vector, x=xs, y=y_s, mutuals=mutuals, inlier_ratio=inlier_ratio, inlier_ratio_mutuals=inlier_ratio_mutuals,
                            ratios=ratios, overlap=overlap_ratio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source_path', default=None, type=str, help='path to the raw files')
    parser.add_argument(
        '--target_path', default=None, type=str, help='path to where the extracted data will be saved')
    parser.add_argument(
        '--dataset', default=None, type=str, help='name of the dataset')
    parser.add_argument(
        '-m',
        '--model',
        default=None,
        type=str,
        help='path to latest checkpoint (default: None)')
    parser.add_argument(
        '--voxel_size',
        default=0.025,
        type=float,
        help='voxel size to preprocess point cloud')
    parser.add_argument(
        '--n_correspondences',
        default=10000,
        type=int,
        help='number of points to be sampled in the correspondence estimation')
    parser.add_argument(
        '--inlier_threshold',
        default=0.05,
        type=float,
        help='threshold to determine if the correspondence is an inlier or outlier')   
    parser.add_argument('--extract_features', action='store_true')
    parser.add_argument('--extract_correspondences', action='store_true')
    parser.add_argument('--extract_precomputed_training_data', action='store_true')
    parser.add_argument('--refine', action='store_true')
    parser.add_argument('--with_cuda', action='store_true')


    args = parser.parse_args()

    # Prepare the logger
    logger = logging.getLogger()
    coloredlogs.install(level='INFO', logger=logger)
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')

    device = torch.device('cuda' if args.with_cuda and torch.cuda.is_available() else 'cpu')

    if args.extract_features:
        assert args.model is not None

        checkpoint = torch.load(args.model)
        config = checkpoint['config']

        num_feats = 1
        model = FCGFNet(
            num_feats,
            config.model_n_out,
            bn_momentum=0.05,
            normalize_feature=config.normalize_feature,
            conv1_kernel_size=config.conv1_kernel_size,
            D=3)

        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        model = model.to(device)

        with torch.no_grad():
            if args.extract_features:
                logger.info('Starting feature extraction')
                extract_features_batch(model, args.source_path, args.target_path, args.dataset, config.voxel_size,
                                        device)
                logger.info('Feature extraction completed')

    if args.extract_correspondences:
        logger.info('Starting establishing pointwise correspondences in the feature space')
        extract_correspondences(args.dataset,args.source_path, args.target_path, args.n_correspondences)
        logger.info('Pointwise correspondences in the features space established')

    if args.extract_precomputed_training_data:
        logger.info('Starting establishing pointwise correspondences in the feature space')
        extract_precomputed_training_data(args.dataset,args.source_path,
                                          args.target_path, args.inlier_threshold, args.voxel_size)
        logger.info('Pointwise correspondences in the features space established')

            