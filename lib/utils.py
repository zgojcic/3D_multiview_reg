import math
import os 
import copy
import torch
import re
import numpy as np
import torch.nn.functional as F
import open3d as o3d
import nibabel.quaternions as nq
import logging
import yaml
import time

from torch import optim
from itertools import combinations
from torch.distributions import normal
from sklearn.neighbors import NearestNeighbors

def load_config(path):
    """
    Loads config file:

    Args:
        path (str): path to the config file

    Returns: 
        config (dict): dictionary of the configuration parameters

    """
    with open(path,'r') as f:
        cfg = yaml.safe_load(f)

    return cfg


def load_point_cloud(file, data_type='numpy'):
    """
    Loads the point cloud coordinates from the '*.ply' file.

    Args: 
        file (str): path to the '*.ply' file
        data_type (str): data type to be reurned (default: numpy)

    Returns:
        ae (torch tensor): Rotation error in angular degreees [b,1]

    """
    temp_pc = o3d.io.read_point_cloud(file)
    
    assert data_type in ['numpy', 'open3d'], 'Wrong data type selected when loading the ply file.' 
    
    if data_type == 'numpy':
        return np.asarray(temp_pc.points)
    else:         
        return temp_pc

def sorted_alphanum(file_list_ordered):
    """
    Sorts the list alphanumerically

    Args:
        file_list_ordered (list): list of files to be sorted

    Return:
        sorted_list (list): input list sorted alphanumerically
    """
    def convert(text):
        return int(text) if text.isdigit() else text

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    sorted_list = sorted(file_list_ordered, key=alphanum_key)

    return sorted_list



def get_file_list(path, extension=None):
    """
    Build a list of all the files in the provided path

    Args:
        path (str): path to the directory 
        extension (str): only return files with this extension

    Return:
        file_list (list): list of all the files (with the provided extension) sorted alphanumerically
    """
    if extension is None:
        file_list = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    else:
        file_list = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and os.path.splitext(f)[1] == extension
        ]
    file_list = sorted_alphanum(file_list)

    return file_list


def get_folder_list(path):
    """
    Build a list of all the files in the provided path

    Args:
        path (str): path to the directory 
        extension (str): only return files with this extension

    Returns:
        file_list (list): list of all the files (with the provided extension) sorted alphanumerically
    """
    folder_list = [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    folder_list = sorted_alphanum(folder_list)
    
    return folder_list




def rotation_error(R1, R2):
    """
    Torch batch implementation of the rotation error between the estimated and the ground truth rotatiom matrix. 
    Rotation error is defined as r_e = \arccos(\frac{Trace(\mathbf{R}_{ij}^{T}\mathbf{R}_{ij}^{\mathrm{GT}) - 1}{2})

    Args: 
        R1 (torch tensor): Estimated rotation matrices [b,3,3]
        R2 (torch tensor): Ground truth rotation matrices [b,3,3]

    Returns:
        ae (torch tensor): Rotation error in angular degreees [b,1]

    """
    R_ = torch.matmul(R1.transpose(1,2), R2)
    e = torch.stack([(torch.trace(R_[_, :, :]) - 1) / 2 for _ in range(R_.shape[0])], dim=0).unsqueeze(1)

    # Clamp the errors to the valid range (otherwise torch.acos() is nan)
    e = torch.clamp(e, -1, 1, out=None)

    ae = torch.acos(e)
    pi = torch.Tensor([math.pi])
    ae = 180. * ae / pi.to(ae.device).type(ae.dtype)

    return ae


def translation_error(t1, t2):
    """
    Torch batch implementation of the rotation error between the estimated and the ground truth rotatiom matrix. 
    Rotation error is defined as r_e = \arccos(\frac{Trace(\mathbf{R}_{ij}^{T}\mathbf{R}_{ij}^{\mathrm{GT}) - 1}{2})

    Args: 
        t1 (torch tensor): Estimated translation vectors [b,3,1]
        t2 (torch tensor): Ground truth translation vectors [b,3,1]

    Returns:
        te (torch tensor): translation error in meters [b,1]

    """
    return torch.norm(t1-t2, dim=(1, 2))


def kabsch_transformation_estimation(x1, x2, weights=None, normalize_w = True, eps = 1e-7, best_k = 0, w_threshold = 0):
    """
    Torch differentiable implementation of the weighted Kabsch algorithm (https://en.wikipedia.org/wiki/Kabsch_algorithm). Based on the correspondences and weights calculates
    the optimal rotation matrix in the sense of the Frobenius norm (RMSD), based on the estimate rotation matrix is then estimates the translation vector hence solving
    the Procrustes problem. This implementation supports batch inputs.

    Args:
        x1            (torch array): points of the first point cloud [b,n,3]
        x2            (torch array): correspondences for the PC1 established in the feature space [b,n,3]
        weights       (torch array): weights denoting if the coorespondence is an inlier (~1) or an outlier (~0) [b,n]
        normalize_w   (bool)       : flag for normalizing the weights to sum to 1
        best_k        (int)        : number of correspondences with highest weights to be used (if 0 all are used)
        w_threshold   (float)      : only use weights higher than this w_threshold (if 0 all are used)
    Returns:
        rot_matrices  (torch array): estimated rotation matrices [b,3,3]
        trans_vectors (torch array): estimated translation vectors [b,3,1]
        res           (torch array): pointwise residuals (Eucledean distance) [b,n]
        valid_gradient (bool): Flag denoting if the SVD computation converged (gradient is valid)

    """
    if weights is None:
        weights = torch.ones(x1.shape[0],x1.shape[1]).type_as(x1).to(x1.device)

    if normalize_w:
        sum_weights = torch.sum(weights,dim=1,keepdim=True) + eps
        weights = (weights/sum_weights)

    weights = weights.unsqueeze(2)

    if best_k > 0:
        indices = np.argpartition(weights.cpu().numpy(), -best_k, axis=1)[0,-best_k:,0]
        weights = weights[:,indices,:]
        x1 = x1[:,indices,:]
        x2 = x2[:,indices,:]

    if w_threshold > 0:
        weights[weights < w_threshold] = 0


    x1_mean = torch.matmul(weights.transpose(1,2), x1) / (torch.sum(weights, dim=1).unsqueeze(1) + eps)
    x2_mean = torch.matmul(weights.transpose(1,2), x2) / (torch.sum(weights, dim=1).unsqueeze(1) + eps)

    x1_centered = x1 - x1_mean
    x2_centered = x2 - x2_mean

    weight_matrix = torch.diag_embed(weights.squeeze(2))

    cov_mat = torch.matmul(x1_centered.transpose(1, 2),
                           torch.matmul(weight_matrix, x2_centered))

    try:
        u, s, v = torch.svd(cov_mat)
    except Exception as e:
        r = torch.eye(3,device=x1.device)
        r = r.repeat(x1_mean.shape[0],1,1)
        t = torch.zeros((x1_mean.shape[0],3,1), device=x1.device)

        res = transformation_residuals(x1, x2, r, t)

        return r, t, res, True

    tm_determinant = torch.det(torch.matmul(v.transpose(1, 2), u.transpose(1, 2)))

    determinant_matrix = torch.diag_embed(torch.cat((torch.ones((tm_determinant.shape[0],2),device=x1.device), tm_determinant.unsqueeze(1)), 1))

    rotation_matrix = torch.matmul(v,torch.matmul(determinant_matrix,u.transpose(1,2)))

    # translation vector
    translation_matrix = x2_mean.transpose(1,2) - torch.matmul(rotation_matrix,x1_mean.transpose(1,2))

    # Residuals
    res = transformation_residuals(x1, x2, rotation_matrix, translation_matrix)

    return rotation_matrix, translation_matrix, res, False


def transformation_residuals(x1, x2, R, t):
    """
    Computer the pointwise residuals based on the estimated transformation paramaters
    
    Args:
        x1  (torch array): points of the first point cloud [b,n,3]
        x2  (torch array): points of the second point cloud [b,n,3]
        R   (torch array): estimated rotation matrice [b,3,3]
        t   (torch array): estimated translation vectors [b,3,1]
    Returns:
        res (torch array): pointwise residuals (Eucledean distance) [b,n,1]
    """
    x2_reconstruct = torch.matmul(R, x1.transpose(1, 2)) + t 

    res = torch.norm(x2_reconstruct.transpose(1, 2) - x2, dim=2)

    return res

def transform_point_cloud(x1, R, t):
    """
    Transforms the point cloud using the giver transformation paramaters
    
    Args:
        x1  (np array): points of the point cloud [b,n,3]
        R   (np array): estimated rotation matrice [b,3,3]
        t   (np array): estimated translation vectors [b,3,1]
    Returns:
        x1_t (np array): points of the transformed point clouds [b,n,3]
    """
    x1_t = (torch.matmul(R, x1.transpose(0,2,1)) + t).transpose(0,2,1)

    return x1_t


def knn_point(k, pos1, pos2):
    '''
    Performs the k nearest neighbors search with CUDA support

    Args:
        k (int): number of k in k-nn search
        pos1: (torch tensor) input points [b,n,c]
        pos2: (torch tensor) float32 array, query points [b,m,c]

    Returns:
        val: (torch tensor)  squared L2 distances [b,m,k]
        idx: (torch tensor)  indices of the k nearest points [b,m,1]

    '''

    B, N, C = pos1.shape
    M = pos2.shape[1]

    pos1 = pos1.view(B,1,N,-1).repeat(1,M,1,1)
    pos2 = pos2.view(B,M,1,-1).repeat(1,1,N,1)

    dist = torch.sum(-(pos1 - pos2)**2,  -1)

    val, idx = dist.topk(k=k,dim = -1)

    return -val, idx

def axis_angle_to_rot_mat(axes, thetas):
    """
    Computer a rotation matrix from the axis-angle representation using the Rodrigues formula.
    \mathbf{R} = \mathbf{I} + (sin(\theta)\mathbf{K} + (1 - cos(\theta)\mathbf{K}^2), where K = \mathbf{I} \cross \frac{\mathbf{K}}{||\mathbf{K}||}

    Args:
    axes (numpy array): array of axes used to compute the rotation matrices [b,3]
    thetas (numpy array): array of angles used to compute the rotation matrices [b,1]

    Returns:
    rot_matrices (numpy array): array of the rotation matrices computed from the angle, axis representation [b,3,3]

    """

    R = []
    for k in range(axes.shape[0]):
        K = np.cross(np.eye(3), axes[k,:]/np.linalg.norm(axes[k,:]))
        R.append( np.eye(3) + np.sin(thetas[k])*K + (1 - np.cos(thetas[k])) * np.matmul(K,K))

    rot_matrices = np.stack(R)
    return rot_matrices


def sample_random_trans(pcd, randg=None, rotation_range=360):
    """
    Samples random transformation paramaters with the rotaitons limited to the rotation range

    Args:
    pcd (numpy array): numpy array of coordinates for which the transformation paramaters are sampled [n,3]
    randg (numpy random generator): numpy random generator

    Returns:
    T (numpy array): sampled transformation paramaters [4,4]

    """
    if randg == None:
        randg = np.random.default_rng(41)

    # Create 3D identity matrix
    T = np.zeros((4,4))
    idx = np.arange(4)
    T[idx,idx] = 1
    
    axes = np.random.rand(1,3) - 0.5

    angles = rotation_range * np.pi / 180.0 * (np.random.rand(1,1) - 0.5)

    R = axis_angle_to_rot_mat(axes, angles)

    T[:3, :3] = R
    T[:3, 3]  = np.matmul(R,-np.mean(pcd, axis=0))

    return T


def augment_precomputed_data(x,R,t, max_angle=360.0):
    """
    Function used for data augmention (random transformation) in the training process. It transforms the point from PC1 with a randomly sampled
    transformation matrix and updates the ground truth rotation and translation, respectively.

    Args:
    x (np.array): coordinates of the correspondences [n,6]
    R (np.array): gt rotation matrix [3,3]
    t (np.array): gt translation vector [3,1]
    max_angle (float): maximum angle that should be used to sample the rotation matrix

    Returns:
    t_data (numpy array): augmented coordinates of the correspondences [n, 6]
    t_rs (numpy array): augmented rotation matrix [3,3]
    t_ts (numpy array): augmented translation vector [3,1]
    """

    # Sample random transformation matrix for each example in the batch
    T_rand = sample_random_trans(x[:, 0:3], np.random.RandomState(), max_angle)

    # Compute the updated ground truth transformation paramaters R_n = R_gt*R_s^-1, t_n = t_gt - R_gt*R_s^-1*t_s
    rotation_matrix_inv = T_rand[:3,:3].transpose()
    t_rs = np.matmul(R,rotation_matrix_inv)
    t_ts = t - np.matmul(R, np.matmul(rotation_matrix_inv, T_rand[:3,2:3].reshape(-1,1)))

    # Transform the coordinates of the first point cloud with the sampled transformation parmaters
    t_xs = (np.matmul(T_rand[:3,:3], x[:, 0:3].transpose()) + T_rand[:3,2:3].reshape(-1,1)).transpose()

    t_data = np.concatenate((t_xs,x[:, 3:6]),axis=-1)
    

    return t_data, t_rs, t_ts


def add_jitter(x, R, t, std=0.01, clip=0.025):
    """
    Function used to add jitter to the coordinates of the correspondences in the training process. 

    Args:
    x (np.array): coordinates of the correspondences [n,6]
    R (np.array): gt rotation matrix [3,3]
    t (np.array): gt translation vector [3,1]
    std (float): standard deviation of the normal distribution used to sample the jitter
    clip (float): cut-off value for the jitter

    Returns:
    x (np.array): coordinates of the correspondences with added jitter [n,6]
    y (np.array): gt residuals of the correspondences aftter jitter [n]
    """

    jitter = np.clip(np.random.normal(0.0, scale=std, size=(x.shape[0], x.shape[1])),
                                      a_min=-clip, a_max=clip)

    x += jitter  # Add noise to xyz

    # Compute new ys
    temp_x = (np.matmul(R,x[:,0:3].transpose()) + t).transpose()
    y = np.sqrt(np.sum((x[:,3:6]-temp_x)**2,1))

    return x, y


class ClippedStepLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, step_size, min_lr, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.min_lr = min_lr
        self.gamma = gamma
        super(ClippedStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * self.gamma ** (self.last_epoch // self.step_size), self.min_lr)
                for base_lr in self.base_lrs]



def ensure_dir(path):
    """
        Creates the directory specigied by the input if it does not yet exist. 
    """
    if not os.path.exists(path):
        os.makedirs(path, mode=0o755)

def read_trajectory(filename, dim=4):
    """
    Function that reads a trajectory saved in the 3DMatch/Redwood format to a numpy array. 
    Format specification can be found at http://redwood-data.org/indoor/fileformat.html
    
    Args:
    filename (str): path to the '.txt' file containing the trajectory data
    dim (int): dimension of the transformation matrix (4x4 for 3D data)

    Returns:
    final_keys (dict): indices of pairs with more than 30% overlap (only this ones are included in the gt file)
    traj (numpy array): gt pairwise transformation matrices for n pairs[n,dim, dim] 
    """

    with open(filename) as f:
        lines = f.readlines()

        # Extract the point cloud pairs
        keys = lines[0::(dim+1)]
        temp_keys = []
        for i in range(len(keys)):
            temp_keys.append(keys[i].split('\t')[0:3])

        final_keys = []
        for i in range(len(temp_keys)):
            final_keys.append([temp_keys[i][0].strip(), temp_keys[i][1].strip(), temp_keys[i][2].strip()])


        traj = []
        for i in range(len(lines)):
            if i % 5 != 0:
                curr_line = '\t'.join(lines[i].split())
                traj.append(curr_line.split('\t')[0:dim])

        traj = np.asarray(traj, dtype=np.float).reshape(-1,dim,dim)
        
        final_keys = np.asarray(final_keys)

        return final_keys, traj



def write_trajectory(traj, metadata, filename, dim=4):
    """
    Writes the trajectory into a '.txt' file in 3DMatch/Redwood format. 
    Format specification can be found at http://redwood-data.org/indoor/fileformat.html

    Args:
    traj (numpy array): trajectory for n pairs[n,dim, dim] 
    metadata (numpy array): file containing metadata about fragment numbers [n,3]
    filename (str): path where to save the '.txt' file containing trajectory data
    dim (int): dimension of the transformation matrix (4x4 for 3D data)
    """

    with open(filename, 'w') as f:
        for idx in range(traj.shape[0]):
            # Only save the transfromation parameters for which the overlap threshold was satisfied
            if metadata[idx][2] == 'True':
                p = traj[idx,:,:].tolist()
                f.write('\t'.join(map(str, metadata[idx])) + '\n')
                f.write('\n'.join('\t'.join(map('{0:.12f}'.format, p[i])) for i in range(dim)))
                f.write('\n')


def read_trajectory_info(filename, dim=6):
    """
    Function that reads the trajectory information saved in the 3DMatch/Redwood format to a numpy array.
    Information file contains the variance-covariance matrix of the transformation paramaters. 
    Format specification can be found at http://redwood-data.org/indoor/fileformat.html
    
    Args:
    filename (str): path to the '.txt' file containing the trajectory information data
    dim (int): dimension of the transformation matrix (4x4 for 3D data)

    Returns:
    n_frame (int): number of fragments in the scene
    cov_matrix (numpy array): covariance matrix of the transformation matrices for n pairs[n,dim, dim] 
    """

    with open(filename) as fid:
        contents = fid.readlines()
    n_pairs = len(contents) // 7
    assert (len(contents) == 7 * n_pairs)
    info_list = []
    n_frame = 0

    for i in range(n_pairs):
        frame_idx0, frame_idx1, n_frame = [int(item) for item in contents[i * 7].strip().split()]
        info_matrix = np.concatenate(
            [np.fromstring(item, sep='\t').reshape(1, -1) for item in contents[i * 7 + 1:i * 7 + 7]], axis=0)
        info_list.append(info_matrix)
    
    cov_matrix = np.asarray(info_list, dtype=np.float).reshape(-1,dim,dim)
    
    return n_frame, cov_matrix

def extract_corresponding_trajectors(est_pairs,gt_pairs, est_traj, gt_traj):
    """
    Extract only those transformation matrices from the estimated trajectory that are also in the GT trajectory.
    
    Args:
    est_pairs (numpy array): indices of point cloud pairs with enough estimated overlap [m, 3]
    gt_pairs (numpy array): indices of gt overlaping point cloud pairs [n,3]
    est_traj (numpy array): 3d array of the estimated transformation parameters [m,4,4]
    gt_traj (numpy array): 3d array of the gt transformation parameters [n,4,4]

    Returns:
    ext_traj_est (numpy array): extracted est transformation parameters for the point cloud pairs from  [k,4,4] 
    ext_traj_gt (numpy array): extracted gt transformation parameters for the point cloud pairs from est_pairs [k,4,4] 
    """
    ext_traj_est = []
    ext_traj_gt = []

    est_pairs = est_pairs[:,0:2]
    gt_pairs = gt_pairs[:,0:2]

    for gt_idx, pair in enumerate(gt_pairs):
        est_idx = np.where((est_pairs == pair).all(axis=1))[0]
        if est_idx.size:
            ext_traj_gt.append(gt_traj[gt_idx,:,:])
            ext_traj_est.append(est_traj[est_idx[0],:,:])

    return np.stack(ext_traj_est, axis=0), np.stack(ext_traj_gt, axis=0)

def computeTransformationErr(trans, info):
    """
    Computer the transformation error as an approximation of the RMSE of corresponding points.
    More informaiton at http://redwood-data.org/indoor/registration.html
    
    Args:
    trans (numpy array): transformation matrices [n,4,4]
    info (numpy array): covariance matrices of the gt transformation paramaters [n,4,4]

    Returns:
    p (float): transformation error
    """
    
    t = trans[:3, 3]
    r = trans[:3, :3]
    q = nq.mat2quat(r)
    er = np.concatenate([t, q[1:]], axis=0)
    p = er.reshape(1, 6) @ info @ er.reshape(6, 1) / info[0, 0]
    
    return p.item()


def evaluate_registration(num_fragment, result, result_pairs, gt_pairs, gt, gt_info, err2=0.2):
    """
    Evaluates the performance of the registration algorithm according to the evaluation protocol defined
    by the 3DMatch/Redwood datasets. The evaluation protocol can be found at http://redwood-data.org/indoor/registration.html
    
    Args:
    num_fragment (int): path to the '.txt' file containing the trajectory information data
    result (numpy array): estimated transformation matrices [n,4,4]
    result_pairs (numpy array): indices of the point cloud for which the transformation matrix was estimated (m,3)
    gt_pairs (numpy array): indices of the ground truth overlapping point cloud pairs (n,3)
    gt (numpy array): ground truth transformation matrices [n,4,4]
    gt_cov (numpy array): covariance matrix of the ground truth transfromation parameters [n,6,6]
    err2 (float): threshold for the RMSE of the gt correspondences (default: 0.2m)

    Returns:
    precision (float): mean registration precision over the scene (not so important because it can be increased see papers)
    recall (float): mean registration recall over the scene (deciding parameter for the performance of the algorithm)
    """

    err2 = err2 ** 2
    gt_mask = np.zeros((num_fragment, num_fragment), dtype=np.int)



    for idx in range(gt_pairs.shape[0]):
        i = int(gt_pairs[idx,0])
        j = int(gt_pairs[idx,1])

        # Only non consecutive pairs are tested
        if j - i > 1:
            gt_mask[i, j] = idx

    n_gt = np.sum(gt_mask > 0)

    good = 0
    n_res = 0
    for idx in range(result_pairs.shape[0]):
        i = int(result_pairs[idx,0])
        j = int(result_pairs[idx,1])
        pose = result[idx,:,:]

        if j - i > 1:
            n_res += 1
            if gt_mask[i, j] > 0:
                gt_idx = gt_mask[i, j]
                p = computeTransformationErr(np.linalg.inv(gt[gt_idx,:,:]) @ pose, gt_info[gt_idx,:,:])
                if p <= err2:
                    good += 1
    if n_res == 0:
        n_res += 1e6
    precision = good * 1.0 / n_res
    recall = good * 1.0 / n_gt

    return precision, recall




def do_single_pair_RANSAC_reg(xyz_i, xyz_j, pc_i, pc_j, voxel_size=0.025,method='3DMatch'):
    """
    Runs a RANSAC registration pipeline for a single pair of point clouds.
    
    Args:
    xyz_i (numpy array): coordinates of the correspondences from the first point cloud [n,3]
    xyz_j (numpy array): coordinates of the correspondences from the second point cloud [n,3]
    pc_i (numpy array): coordinates of all the points from the first point cloud [N,3]
    pc_j (numpy array): coordinates of all the points from the second point cloud [N,3]
    method (str): name of the method used for the overlap computation [3DMatch, FCGF]

    Returns:
    overlap_flag (bool): flag denoting if overlap of the point cloud after aplying the estimated trans paramaters if more than a threshold
    trans (numpy array): transformation parameters that trnasform point of point cloud 2 to the coordinate system of point cloud 1
    """


    trans = run_ransac(xyz_j, xyz_i)


    ratio = compute_overlap_ratio(pc_i, pc_j, trans, method, voxel_size)
    
    overlap_flag = True if ratio > 0.3 else False

    
    return [overlap_flag, trans]



def run_ransac(xyz_i, xyz_j):
    """
    Ransac based estimation of the transformation paramaters of the congurency transformation. Estimates the
    transformation parameters thtat map xyz0 to xyz1. Implementation is based on the open3d library
    (http://www.open3d.org/docs/release/python_api/open3d.registration.registration_ransac_based_on_correspondence.html)
    
    Args:
    xyz_i (numpy array): coordinates of the correspondences from the first point cloud [n,3]
    xyz_j (numpy array): coordinates of the correspondences from the second point cloud [n,3]

    Returns:
    trans_param (float): mean registration precision over the scene (not so important because it can be increased see papers)
    recall (float): mean registration recall over the scene (deciding parameter for the performance of the algorithm)
    """

    # Distance threshold as specificed by 3DMatch dataset
    distance_threshold = 0.05

    # Convert the point to an open3d PointCloud object
    xyz0 = o3d.geometry.PointCloud()
    xyz1 = o3d.geometry.PointCloud()
    
    xyz0.points = o3d.utility.Vector3dVector(xyz_i)
    xyz1.points = o3d.utility.Vector3dVector(xyz_j)

    # Correspondences are already sorted
    corr_idx = np.tile(np.expand_dims(np.arange(len(xyz0.points)),1),(1,2))
    corrs = o3d.utility.Vector2iVector(corr_idx)

    result_ransac = o3d.registration.registration_ransac_based_on_correspondence(
        source=xyz0, target=xyz1,corres=corrs, 
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        criteria=o3d.registration.RANSACConvergenceCriteria(50000, 2500))

    trans_param = result_ransac.transformation
    
    return trans_param

  

def compute_overlap_ratio(pc_i, pc_j, trans, method = '3DMatch', voxel_size=0.025):
    """
    Computes the overlap percentage of the two point clouds using the estimateted transformation paramaters and based on the selected method.
    Available methods are 3DMatch/Redwood as defined in the oficial dataset and the faster FCGF method that first downsamples the point clouds.
    Method 3DMatch slightly deviates from the original implementation such that we take the max of the overlaps to check if it is above the threshold
    where as in the original implementation only the overlap relative to PC1 is used. 

    Args:
    pc_i (numpy array): coordinates of all the points from the first point cloud [N,3]
    pc_j (numpy array): coordinates of all the points from the second point cloud [N,3]
    trans (numpy array): estimated transformation paramaters [4,4]
    method (str): name of the method for overlap computation to be used ['3DMatch', 'FCGF']
    voxel size (float): voxel size used to downsample the point clouds when 'FCGF' method is selected

    Returns:
    overlap (float): max of the computed overlap ratios relative to the PC1 and PC2

    """
    neigh = NearestNeighbors(n_neighbors=1,algorithm='kd_tree')
    trans_inv = np.linalg.inv(trans)

    if method == '3DMatch':
        pc_i_t = (np.matmul(trans_inv[0:3, 0:3], pc_i.transpose()) + trans_inv[0:3, 3].reshape(-1, 1)).transpose()
        pc_j_t = (np.matmul(trans[0:3, 0:3], pc_j.transpose()) + trans[0:3, 3].reshape(-1, 1)).transpose()

        neigh.fit(pc_j_t)
        dist01, _ = neigh.kneighbors(pc_i, return_distance=True)
        matching01 = np.where(dist01 < 0.05)[0].shape[0]

        neigh.fit(pc_i_t)
        dist10, _ = neigh.kneighbors(pc_j, return_distance=True)
        matching10 = np.where(dist10 < 0.05)[0].shape[0]

        overlap0 = matching01 / pc_i.shape[0]
        overlap1 = matching10 / pc_j.shape[0]

    elif method == 'FCGF':
        # Convert the point to an open3d PointCloud object
        pcd0 = o3d.geometry.PointCloud()
        pcd1 = o3d.geometry.PointCloud()
        pcd0.points = o3d.utility.Vector3dVector(pc_i)
        pcd1.points = o3d.utility.Vector3dVector(pc_j)

        pcd0_down = pcd0.voxel_down_sample(voxel_size)
        pcd1_down = pcd1.voxel_down_sample(voxel_size)

        pc_i = np.array(pcd0_down.points)
        pc_j = np.array(pcd1_down.points)

        pc_i_t = (np.matmul(trans_inv[0:3, 0:3], pc_i.transpose()) + trans_inv[0:3, 3].reshape(-1, 1)).transpose()
        pc_j_t = (np.matmul(trans[0:3, 0:3], pc_j.transpose()) + trans[0:3, 3].reshape(-1, 1)).transpose()

        neigh.fit(pc_j_t)
        dist01, _ = neigh.kneighbors(pc_i, return_distance=True)
        matching01 = np.where(dist01 < 3*voxel_size)[0].shape[0]

        neigh.fit(pc_i_t)
        dist10, _ = neigh.kneighbors(pc_j, return_distance=True)
        matching10 = np.where(dist10 < 3*voxel_size)[0].shape[0]

        overlap0 = matching01 / pc_i.shape[0]
        overlap1 = matching10 / pc_j.shape[0]

        # matching01 = get_matching_indices(pcd0_down, pcd1_down, np.linalg.inv(trans), search_voxel_size = 3*voxel_size, K=1)
        # matching10 = get_matching_indices(pcd1_down, pcd0_down, trans,
        #                             search_voxel_size = 3*voxel_size, K=1)
        # overlap0 = len(matching01) / len(pcd0_down.points)
        # overlap1 = len(matching10) / len(pcd1_down.points)

    else:
        logging.error("Wrong overlap computation method was selected.")

    
    return max(overlap0, overlap1)


def get_matching_indices(pc_i, pc_j, trans, search_voxel_size=0.025, K=None, method = 'FCGF'):   
    """
    Helper function for the point cloud overlap computation. Based on the estimated transformation parameters transforms the point cloud
    and searches for the neares neighbor in the other point cloud.

    Args:
    pc_i (numpy array): coordinates of all the points from the first point cloud [N,3]
    pc_j (numpy array): coordinates of all the points from the second point cloud [N,3]
    trans (numpy array): estimated transformation paramaters [4,4]
    search_voxel_size (float): threshold used to determine if a point has a correspondence given the estimated trans parameters
    K (int): number of nearest neighbors to be returned

    Returns:
    match_inds (list): indices of points that have a correspondence withing the search_voxel_size

    """

    pc_i_copy = copy.deepcopy(pc_i)
    pc_j_copy = copy.deepcopy(pc_j)
    pc_i_copy.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(pc_j_copy)

    match_inds = []
    for i, point in enumerate(pc_i_copy.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
            
        for j in idx:
            match_inds.append((i, j))

    return match_inds

def extract_mutuals(x1, x2, x1_soft_matches, x2_soft_matches, threshold=0.05):
    '''
    Returns a flag if the two point are mutual nearest neighbors in the feature space.
    In a softNN formulation a distance threshold has to be used.

    Args:
        x1 (torch tensor): source point cloud [b,n,3]
        x2 (torch tensor): target point cloud [b,n,3]
        x1_soft_matches (torch tensor): coordinates of the (soft) correspondences for points x1 in x2 [b,n,3]
        x2_soft_matches (torch tensor): coordinates of the (soft) correspondences for points x2 in x1 [b,n,3]

    Returns:
        mutuals (torch tensor): mutual nearest neighbors flag (1 if mutual NN otherwise 0) [b,n]

    '''

    B, N, C = x1.shape

    _, idx = knn_point(k=1, pos1=x2, pos2=x1_soft_matches)

    delta = x1 - torch.gather(x2_soft_matches,index=idx.expand(-1,-1,C),dim=1)
    dist = torch.pow(delta,2).sum(dim=2)

    mutuals = torch.zeros((B,N))
    mutuals[dist < threshold**2] = 1

    return mutuals

def extract_overlaping_pairs(xyz, feat, conectivity_info=None):   
    """
    Build the point cloud pairs based either on the provided conectivity information or sample 
    all n choose 2 posibilities

    Args:
    xyz (torch tensor): coordinates of the sampled points [b,n,3]
    feat (torch tensor): features of the sampled points [b,n,c]
    conectivity_info (torch tensor): conectivity information (indices of overlapping pairs) [m,2]

    Returns:
    xyz_s (torch tensor): coordinates of the points in the source point clouds [B, n, 3] (B != b)
    xyz_t (torch tensor): coordinates of the points in the target point clouds [B, n, 3] (B != b)
    f_s (torch tensor): features of the points in the source point clouds [B, n, 3] (B != b)
    f_t (torch tensor): features of the points in the target point clouds [B, n, 3] (B != b)

    """

    if not conectivity_info:

        pairs = []

        # If no conectivity information is provided sample n choose 2 pairs
        for comb in list(combinations(range(xyz.shape[0]), 2)):
            pairs.append(torch.tensor([int(comb[0]), int(comb[1])]))

        conectivity_info = torch.stack(pairs, dim=0).to(xyz.device).long()

    # Build the point cloud pairs based on the conectivity information
    xyz_s = torch.index_select(xyz, dim=0, index=conectivity_info[:, 0])
    xyz_t = torch.index_select(xyz, dim=0, index=conectivity_info[:, 1])

    f_s = torch.index_select(feat, dim=0, index=conectivity_info[:, 0])
    f_t = torch.index_select(feat, dim=0, index=conectivity_info[:, 1])

    return xyz_s, xyz_t, f_s, f_t


def construct_filtering_input_data(xyz_s, xyz_t, data, overlapped_pair_tensors, dist_th=0.05, mutuals_flag=None):
    """
    Prepares the input dictionary for the filtering network 
    
    Args:
    xyz_s (torch tensor): coordinates of the sampled points in the source point cloud [b,n,3]
    xyz_t (torch tensor): coordinates of the correspondences from the target point cloud [b,n,3]
    data (dict): input data from the data loader
    dist_th (float): distance threshold to determine if the correspondence is an inlier or an outlier
    mutuals (torch tensor): torch tensor of the mutually nearest neighbors (can be used as side information to the filtering network)

    Returns:
    filtering_data (dict): input data for the filtering network

    """

    filtering_data = {}

    if 'T_global_0' in data:
        Rs, ts = extract_transformation_matrices(data['T_global_0'], overlapped_pair_tensors)
        ys = transformation_residuals(xyz_s, xyz_t, Rs, ts)
    
    else: 
        ys = torch.zeros(xyz_s.shape[0], xyz_s.shape[1], 1)
        Rs = torch.eye(3).unsqueeze(0).repeat(xyz_s.shape[0],1,1)
        ts = torch.zeros(xyz_s.shape[0], 3, 1)

    xs = torch.cat((xyz_s,xyz_t),dim=-1) # [b, n, 6]

    if mutuals_flag is not None:
        xs = torch.cat((xs,mutuals_flag.reshape(-1,1)), dim=-1) # [b, n, 7]


    # Threshold ys based on the distance threshol
    ys_binary = (ys < dist_th).type(xs.type())


    # Construct the data dictionary
    filtering_data['xs'] = xs.unsqueeze(1)
    filtering_data['ys'] = ys
    filtering_data['ts'] = ts
    filtering_data['Rs'] = Rs 


    return filtering_data


def extract_transformation_matrices(T0, indices):
    """
    Compute the relative transformation matrices for the overlaping pairs from their global transformation matrices
    
    Args:
    T0 (torch tensor): global transformation matrices [4*b,4]
    indices (torch tensor): indices of the overlaping point couds [B,2]

    Returns:
    rots (torch tensor): pairwise rotation matrices [B,3,3] (B!=b)
    trans (torch tensor): pairwise translation parameters [B,3,1] (B!=b)

    """

    indices = indices.detach().cpu().numpy()
    T0 = T0.detach().cpu().numpy()

    rot_matrices = []
    trans_vectors = []

    for row in indices:

        temp_trans_matrix = T0[4*row[0]:4*(row[0]+1), :] @ np.linalg.inv(T0[4*row[1]:4*(row[1]+1), :])

        rot_matrices.append(temp_trans_matrix[0:3,0:3])
        trans_vectors.append(temp_trans_matrix[0:3,3])

    rots = torch.from_numpy(np.asarray(rot_matrices)).to(T0)
    trans = torch.from_numpy(np.asarray(trans_vectors)).unsqueeze(-1).to(T0)
    
    return rots, trans 


def pairwise_distance(src, dst, normalized_feature=False):
    """Calculates squared Euclidean distance between each two points.

    Args:
        src (torch tensor): source data, [b, n, c]
        dst (torch tensor): target data, [b, m, c]
        normalized_feature (bool): distance computation can be more efficient 

    Returns:
        dist (torch tensor): per-point square distance, [b, n, m]
    """

    B, N, _ = src.shape
    _, M, _ = dst.shape
    
    # Minus such that smaller value still means closer 
    dist = -torch.matmul(src, dst.permute(0, 2, 1))

    # If features are normalized the distance is related to inner product
    if not normalized_feature:
        dist = 2 * dist
        dist += torch.sum(src ** 2, dim=-1)[:, :, None]
        dist += torch.sum(dst ** 2, dim=-1)[:, None, :]

    return dist




if __name__ == '__main__':

    # Weighted Kabsch algorithm example

    pc_1 = torch.rand(1,5000,3)

    T = sample_random_trans(pc_1)

    R = T[:,:3,:3]
    t = T[:,3:4,3]

    pc_2_t = torch.matmul(R,pc_1.transpose(1,2)).transpose(1,2) + t
    
    rotation_matrix, translation_vector, res, _ = kabsch_transformation_estimation(pc_1, pc_2_t)

    print('Input rotation matrix: {}'.format(R))
    print('Estimated rotation matrix: {}'.format(rotation_matrix))

    print('Input translation vector: {}'.format(t))
    print('Estimated translation vector: {}'.format(translation_vector))


class Timer(object):
  """A simple timer."""

  def __init__(self):
    self.total_time = 0.
    self.calls = 0
    self.start_time = 0.
    self.diff = 0.
    self.avg = 0.

  def reset(self):
    self.total_time = 0
    self.calls = 0
    self.start_time = 0
    self.diff = 0
    self.avg = 0

  def tic(self):
    self.start_time = time.time()

  def toc(self, average=True):
    self.diff = time.time() - self.start_time
    self.total_time += self.diff
    self.calls += 1
    self.avg = self.total_time / self.calls
    if average:
      return self.avg
    else:
      return self.diff