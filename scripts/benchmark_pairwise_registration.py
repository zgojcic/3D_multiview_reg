"""
Script for benchmarking the pairwise registration algorithm on 3DMatch and Redwood datasets.
The script expects that the correspondences (any feature descriptor) are precomputed
and benchmarks the transformation estimation algorithms. Other datasets can easily be added
provided that they are expressed in the same data formats.

NOTE: The results might deviate little from the official benchmarking code that is implemented 
in matlab (https://github.com/andyzeng/3dmatch-toolbox). The reason being different RANSAC
implementation and overlap estimation (official one is also implemented here but is slower).

Code is partially borrowed from the Chris Choy's FCGF repository (https://github.com/chrischoy/FCGF)

Author: Zan Gojcic
"""

import os
import glob
import sys
import numpy as np
import argparse
import logging
import open3d as o3d
from collections import defaultdict 
import torch
import coloredlogs
from matplotlib import pyplot as plt
cwd = os.getcwd()
sys.path.append(cwd)

from lib.utils import ensure_dir, read_trajectory, write_trajectory, read_trajectory_info, get_folder_list, \
            kabsch_transformation_estimation, Timer, run_ransac, rotation_error, load_config, \
            translation_error, evaluate_registration, compute_overlap_ratio, extract_corresponding_trajectors

from scripts.utils import make_pairwise_eval_data_loader
from lib.checkpoints import CheckpointIO
import lib.config as config

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

SHORT_NAMES = {}
SHORT_NAMES['3d_match'] = {'kitchen':'Kitchen',
                           'sun3d-home_at-home_at_scan1_2013_jan_1':'Home 1',
                           'sun3d-home_md-home_md_scan9_2012_sep_30':'Home 2',
                           'sun3d-hotel_uc-scan3':'Hotel 1',
                           'sun3d-hotel_umd-maryland_hotel1':'Hotel 2',
                           'sun3d-hotel_umd-maryland_hotel3':'Hotel 3',
                           'sun3d-mit_76_studyroom-76-1studyroom2':'Study',
                           'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika':'MIT Lab'}

SHORT_NAMES['redwood'] = {'iclnuim-livingroom1':'livingroom1',
                          'iclnuim-livingroom2':'livingroom2',
                          'iclnuim-office1':'office1',
                          'iclnuim-office2':'office2'}


def estimate_trans_param_RANSAC(eval_data, source_path, dataset, scene_info, method, mutuals, save_data=False, overlap_method='FCGF'):
    """
    Estimates the pairwise transformation parameters from the provided correspondences using RANSAC and
    saves the results in the trajectory file that can be used to estimate the registration precision and recall.
    
    Args:
    eval_data (torch dataloader): dataloader with the evaluation data
    source_path (numpy array): coordinates of the correspondences from the second point cloud [n,3]
    dataset (str): name of the dataset
    scene_info (dict): metadata of the individual scenes from the dataset
    method (str): method used for estimating the pairwise transformation paramaters
    mutuals (bool): if True only mutually closest neighbors are used
    save_data (bool): if True transformation parameters are saved in npz files
    overlap_method (str): method for overlap computation [FCGF, 3DMatch]

    """

    # Initialize the transformation matrix 
    num_pairs = scene_info['nr_examples']
    est_trans_param = np.tile(np.eye(4),reps=[num_pairs,1])

    # Configure the base save path
    save_path = os.path.join(source_path,'results', method)
    save_path += '/mutuals/' if mutuals else '/all/'
    ensure_dir(save_path)

    reg_metadata = []

    logging.info('Starting RANSAC based registration estimation for {} pairs!'.format(num_pairs))
    avg_timer, full_timer, overlap_timer = Timer(), Timer(), Timer()
    full_timer.tic()
    
    overlap_threshold = 0.3 if dataset =='3d_match' else 0.23

    for batch in eval_data:
        for idx in range(batch['xs'].shape[0]):
            
            data = batch['xs'][idx,0,:,:].numpy()
            pc_1 = batch['xyz1'][idx][0]
            pc_2 = batch['xyz2'][idx][0]
            meta = batch['metadata'][idx]
            pair_idx = int(batch['idx'][idx].numpy().item())

            avg_timer.tic()
            T_est = run_ransac(data[:,0:3], data[:,3:])
            avg_time = avg_timer.toc()
            T_est = np.linalg.inv(T_est)

            overlap_timer.tic()
            overlap_ratio = compute_overlap_ratio(pc_1, pc_2, T_est, method=overlap_method)
            avg_time_overlap = overlap_timer.toc()

            overlap_flag = True if overlap_ratio >= overlap_threshold else False
            est_trans_param[4*pair_idx: 4*pair_idx + 4, :] = T_est

            reg_metadata.append([str(int(meta[1])), str(int(meta[2])), overlap_flag])

            if save_data:
                np.savez_compressed(
                    os.path.join(save_path, meta[0], 'cloud_{}_{}.npz'.format(str(int(meta[1])), str(int(meta[2])))),
                    t_est=T_est[0:3,3],
                    R_est=T_est[0:3,0:3],
                    overlap=overlap_flag)


    if len(eval_data) != 0:
        logging.info('RANSAC based registration estimation is complete!')
        logging.info('{} pairwise registration parameters estimated in {:.3f}s'.format(num_pairs,full_timer.toc(average=False)))
        logging.info('Transformation estimation run time {:.4f}s per pair'.format(avg_time))
        logging.info('Overlap computation run time {:.4f}s per pair'.format(avg_time_overlap))

    # Loop through the transformation matrix and save results to trajectory files
    for key in scene_info:
        if key != 'nr_examples':
            scene_idx = scene_info[key]   
            ensure_dir(os.path.join(save_path, key))
            trans_par = est_trans_param[scene_idx[0]:scene_idx[1],:].reshape(-1,4,4) 
            write_trajectory(trans_par,reg_metadata[scene_idx[0]//4:scene_idx[1]//4], os.path.join(save_path, key, 'traj.txt'))



def infer_transformation_parameters(eval_data, source_path, dataset, scene_info, method, model_path, mutuals, save_data=False, overlap_method='FCGF', refine=False):
    """
    Estimates the pairwise transformation parameters from the provided correspondences using a deep learning model
    and saves the results in the trajectory file that can be used to estimate the registration precision and recall.
    
    Args:
    eval_data (torch dataloader): dataloader with the evaluation data
    source_path (numpy array): coordinates of the correspondences from the second point cloud [n,3]
    dataset (str): name of the dataset
    scene_info (dict): metadata of the individual scenes from the dataset
    method (str): method used for estimating the pairwise transformation paramaters
    model_path (str): path to the model
    mutuals (bool): if True only mutually closest neighbors are used
    save_data (bool): if True transformation parameters are saved in npz files
    overlap_method (str): method for overlap computation [FCGF, 3DMatch]
    refine (bool): if the RANSAC should be applied after the network filtering on the inliers (similar to 2D filtering networks)

    """
    # Model initialization
    logging.info("Using the method {}".format(method))
    
    # Load config file 
    cfg = load_config(os.path.join('./configs/pairwise_registration/eval', method + '.yaml'))
    model = config.get_model(cfg)

    # Load pre-trained model
    model_name = model_path.split('/')[-1]
    model_path = '/'.join(model_path.split('/')[0:-1])

    kwargs = {'model': model}

    checkpoint_io = CheckpointIO(model_path, initialize_from=None, 
                                    initialization_file_name=None, **kwargs)

    load_dict = checkpoint_io.load(model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Initialize the transformation matrix 
    num_pairs = scene_info['nr_examples']
    est_trans_param = np.tile(np.eye(4),reps=[num_pairs,1])

    save_path = os.path.join(source_path,'results', method)
    save_path += '/mutuals/' if args.mutuals else '/all/'

    ensure_dir(save_path)
    reg_metadata = []

    logging.info('Starting {} based registration estimation for {} pairs!'.format(method, num_pairs))
    avg_timer, full_timer = Timer(), Timer()
    full_timer.tic()
    T_est = np.eye(4)

    overlap_threshold = 0.3 if dataset =='3d_match' else 0.23
    logging.info('Using overlap threshold {} for the dataset {}.'.format(overlap_threshold, dataset))
    for batch in eval_data:

        # Filter the correspondences and estimate the pairwise transformation parameters
        avg_timer.tic()
        filtered_output = model.filter_correspondences(batch)


        avg_time = avg_timer.toc()

        # We still have to loop through the batch for the overlap estimation (point cloud not the same size)
        for idx in range(batch['xs'].shape[0]):

            pc_1 = batch['xyz1'][idx][0]
            pc_2 = batch['xyz2'][idx][0]
            meta = batch['metadata'][idx]
            pair_idx = int(batch['idx'][idx].numpy().item())
            if refine:
                data = batch['xs'][idx,0,:,:].numpy()
                inliers = (filtered_output['scores'][-1][idx].cpu().numpy() > 0.5)
                T_est = run_ransac(data[inliers,0:3], data[inliers,3:])

            else:
                T_est[0:3,0:3] = filtered_output['rot_est'][-1][idx].cpu().numpy()
                T_est[0:3,3] = filtered_output['trans_est'][-1][idx].cpu().numpy().reshape(-1)
            T_est = np.linalg.inv(T_est)

            overlap_ratio = compute_overlap_ratio(pc_1, pc_2, T_est, method=overlap_method)
            overlap_flag = True if overlap_ratio >= overlap_threshold else False

            est_trans_param[4*pair_idx: 4*pair_idx + 4, :] = T_est

            reg_metadata.append([str(int(meta[1])), str(int(meta[2])), overlap_flag])

            if save_data:
                np.savez_compressed(
                    os.path.join(save_path, meta[0], 'cloud_{}_{}.npz'.format(str(int(meta[1])), str(int(meta[2])))),
                    t_est=T_est[0:3,3],
                    R_est=T_est[0:3,0:3],
                    overlap=overlap_flag)


    if len(eval_data) != 0:
        logging.info('RANSAC based registration estimation is complete!')
        logging.info('{} pairwise registration parameters estimated in {:.3f}s'.format(num_pairs,full_timer.toc(average=False)))
        logging.info('Pure run time {:.4f}s per pair'.format(avg_time/eval_data.batch_size))

    # Loop through the transformation matrix and save results to trajectory files
    for key in scene_info:
        if key != 'nr_examples':
            scene_idx = scene_info[key]   
            ensure_dir(os.path.join(save_path, key))
            trans_par = est_trans_param[scene_idx[0]:scene_idx[1],:].reshape(-1,4,4) 
            write_trajectory(trans_par,reg_metadata[scene_idx[0]//4:scene_idx[1]//4], os.path.join(save_path, key, 'traj.txt'))




def evaluate_registration_performance(eval_data, source_path,dataset, scene_info, method, model, mutuals=False, save_data = False, overlap_method='FCGF', refine=False):
    """
    Evaluates the pairwise registration performance of the selected method on the selected dataset.
    
    Args:
    eval_data (torch dataloader): dataloader with the evaluation data
    source_path (numpy array): coordinates of the correspondences from the second point cloud [n,3]
    dataset (str): name of the dataset
    scene_info (dict): metadata of the individual scenes from the dataset
    method (str): method used for estimating the pairwise transformation paramaters
    model (str): path to the model
    mutuals (bool): if True only mutually closest neighbors are used
    save_data (bool): if True transformation parameters are saved in npz files
    overlap_method (str): method for overlap computation [FCGF, 3DMatch]
    refine (bool): if the RANSAC should be applied after the network filtering on the inliers (similar to 2D filtering networks)

    """
    # Prepare the variables

    re_per_scene = defaultdict(list)
    te_per_scene = defaultdict(list)
    re_all, te_all, precision, recall = [], [], [], []
    re_medians, te_medians = [], []

    # Estimate the transformation parameters (if the trajectory files are note existing yet)
    if method == 'RANSAC':
        estimate_trans_param_RANSAC(eval_data, source_path, dataset, scene_info, method, mutuals, save_data, overlap_method)
    else:
        infer_transformation_parameters(eval_data, source_path, dataset, scene_info, method, model, mutuals, save_data, overlap_method, refine)

    logging.info("Results of {} on {} dataset!".format(method, dataset))
    logging.info("--------------------------------------------")
    logging.info("{:<12} ¦ prec. ¦ rec.  ¦   re  ¦   te  ¦".format('Scene'))
    logging.info("--------------------------------------------")
    scenes = get_folder_list(os.path.join(source_path,'correspondences'))
    scenes = [scene.split('/')[-1] for scene in scenes]

    for idx, scene in enumerate(scenes):
        # Extract the values from the gt trajectory and trajectory information files
        gt_pairs, gt_traj = read_trajectory(os.path.join(source_path,'raw_data', scene, "gt.log"))
        n_fragments, gt_traj_cov = read_trajectory_info(os.path.join(source_path,'raw_data', scene, "gt.info"))
        assert gt_traj.shape[0] > 0, "Empty trajectory file"

        # Extract the estimated transformation matrices
        if mutuals:
            if method == 'RANSAC':
                est_pairs, est_traj = read_trajectory(os.path.join(source_path, 'results',
                                                method, 'mutuals', scene, "traj.txt"))
            else:
                est_pairs, est_traj = read_trajectory(os.path.join(source_path, 'results', 
                                                method, 'mutuals', scene, "traj.txt"))
        else:
            if method == 'RANSAC':
                est_pairs, est_traj = read_trajectory(os.path.join(source_path, 'results', 
                                                method, 'all', scene, "traj.txt"))
            else:
                est_pairs, est_traj = read_trajectory(os.path.join(source_path, 'results', 
                                                method, 'all', scene, "traj.txt"))
               

        temp_precision, temp_recall = evaluate_registration(n_fragments, est_traj, est_pairs, gt_pairs, gt_traj, gt_traj_cov)
        
        # Filter out only the transformation matrices that are in the GT and EST
        ext_traj_est, ext_traj_gt = extract_corresponding_trajectors(est_pairs,gt_pairs,est_traj, gt_traj)

        re = rotation_error(torch.from_numpy(ext_traj_gt[:,0:3,0:3]), torch.from_numpy(ext_traj_est[:,0:3,0:3])).cpu().numpy()
        te = translation_error(torch.from_numpy(ext_traj_gt[:,0:3,3:4]), torch.from_numpy(ext_traj_est[:,0:3,3:4])).cpu().numpy()

        re_per_scene['mean'].append(np.mean(re))
        re_per_scene['median'].append(np.median(re))
        re_per_scene['min'].append(np.min(re))
        re_per_scene['max'].append(np.max(re))
        

        te_per_scene['mean'].append(np.mean(te))
        te_per_scene['median'].append(np.median(te))
        te_per_scene['min'].append(np.min(te))
        te_per_scene['max'].append(np.max(te))


        re_all.extend(re.reshape(-1).tolist())
        te_all.extend(te.reshape(-1).tolist())

        precision.append(temp_precision)
        recall.append(temp_recall)
        re_medians.append(np.median(re))
        te_medians.append(np.median(te))

        logging.info("{:<12} ¦ {:.3f} ¦ {:.3f} ¦ {:.3f} ¦ {:.3f} ¦".format(SHORT_NAMES[dataset][scene], temp_precision, temp_recall, np.median(re), np.median(te)))
    
    logging.info("--------------------------------------------")
    logging.info("Mean precision: {:.3f} +- {:.3f}".format(np.mean(precision),np.std(precision)))
    logging.info("Mean recall: {:.3f} +- {:.3f}".format(np.mean(recall),np.std(recall)))
    logging.info("Mean ae: {:.3f} +- {:.3f} [deg]".format(np.mean(re_medians),np.std(re_medians)))
    logging.info("Mean te: {:.3f} +- {:.3f} [m]".format(np.mean(te_medians),np.std(te_medians)))
    logging.info("--------------------------------------------")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--source_path', default='./data/eval_data/', type=str, help='path to dataset')
    parser.add_argument(
            '--dataset', default='3d_match', type=str, help='path to dataset')
    parser.add_argument(
            '--method', default='OANet', type=str, help='Which method should be used [RANSAC, RegBlock, Joint]')
    parser.add_argument(
            '--model',
            default=None,
            type=str,
            help='path to latest checkpoint (default: None)')

    parser.add_argument(
            '--batch_size',
            type=int,
            default=32,
            help='Batch size (if mutuals are selected batch size will be 1).')

    parser.add_argument(
            '--mutuals',
            action='store_true',
            help='If only mutually closest NN should be used (reciprocal matching).')

    parser.add_argument(
            '--save_data',
            action='store_true',
            help='If the intermediate data should be saved to npz files.')
    
    parser.add_argument(
            '--overwrite',
            action='store_true',
            help='If results for this method and dataset exist they will be overwritten, otherwised they will be loaded and used')

    parser.add_argument(
            '--overlap_method',
            type=str,
            default='FCGF',
            help='Method to compute the overlap ratio (FCGF or 3DMatch) FCGF is slightly faster than official 3DMatch')

    parser.add_argument(
            '--only_gt_overlaping',
            action='store_true',
            help='Transformation matrices will be computed only for the GT overlaping pairs. Does not change \
                  registration recall that is typically reported in the papers but it is almost 10x faster.')

    parser.add_argument(
            '--refine',
            action='store_true',
            help='The results of the deep methods are refined by the subsequent RANSAC using only the inliers')


    args = parser.parse_args()

    # Prepare the logger
    logger = logging.getLogger()
    coloredlogs.install(level='INFO', logger=logger)
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')


    # Ensure that the source and the target folders were provided
    assert args.source_path is not None

    # Adapt source path
    args.source_path = os.path.join(args.source_path, args.dataset)

    # Prepare the data loader
    eval_data, scene_info = make_pairwise_eval_data_loader(args)
    

    with torch.no_grad():
        evaluate_registration_performance(eval_data, 
                            source_path=args.source_path,
                            dataset=args.dataset, 
                            scene_info=scene_info,
                            method=args.method, 
                            model=args.model,
                            mutuals=args.mutuals,
                            save_data=args.save_data,
                            overlap_method=args.overlap_method,
                            refine=args.refine)