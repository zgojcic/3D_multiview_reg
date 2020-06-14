import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import math
from sklearn.metrics import precision_recall_fscore_support
from lib.utils import rotation_error, transformation_residuals

def _hash(arr, M):
  if isinstance(arr, np.ndarray):
    N, D = arr.shape
  else:
    N, D = len(arr[0]), len(arr)

  hash_vec = np.zeros(N, dtype=np.int64)
  for d in range(D):
    if isinstance(arr, np.ndarray):
      hash_vec += arr[:, d] * M**d
    else:
      hash_vec += arr[d] * M**d

  return hash_vec



class DescriptorLoss():
    """
    Descriptor loss class. Creates a DescriptorLoss object that is used to train the FCGF feature descriptor. The loss is defined the same as in 
    the original FCGF paper.

    Args:
        cfg (dict): configuration parameters

    """
    def __init__(self, cfg):  
        self.w_desc_loss = cfg['loss']['loss_desc']

        # For FCGF loss we keep the parameters the same as in the original paper
        self.pos_thresh = 1.4
        self.neg_thresh = 0.1
        self.batch_size = cfg['train']['batch_size']
        self.num_pos_per_batch = 1024
        self.num_hn_samples_per_batch = 256

    def contrastive_hardest_negative_loss(self,
                                        F0,
                                        F1,
                                        positive_pairs,
                                        num_pos=5192,
                                        num_hn_samples=2048,
                                        pos_thresh=None,
                                        neg_thresh=None):
        """
        Computes the harderst contrastive loss as defined in the Fully Convolutional Geometric Features (Choy et al. ICCV 2019) paper
        
        https://node1.chrischoy.org/data/publications/fcgf/fcgf.pdf. 

        Args:
            F0 (torch tensor)
            F1 (torch tensor)
            positive_pairs (torch tensor): indices of positive pairs
            num_pos (int): maximum number of positive pairs to be used
            num_hn_samples (int): Number of harderst negative samples to be used
            pos_thresh (): margain for positive pairs
            neg_thresh (): margain for negative pairs
        Returns:
            pos_loss (torch tensor): loss based on the positive examples
            neg_loss (torch tensor): loss based on the negative examples
        """

        N0, N1 = len(F0), len(F1)
        N_pos_pairs = len(positive_pairs)
        hash_seed = max(N0, N1)
        sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)
        sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)

        if N_pos_pairs > num_pos:
            pos_sel = np.random.choice(N_pos_pairs, num_pos, replace=False)
            sample_pos_pairs = positive_pairs[pos_sel]
        else:
            sample_pos_pairs = positive_pairs

        # Find negatives for all F1[positive_pairs[:, 1]]
        subF0, subF1 = F0[sel0], F1[sel1]

        pos_ind0 = sample_pos_pairs[:, 0].long()
        pos_ind1 = sample_pos_pairs[:, 1].long()
        posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

        D01 = torch.sum((posF0.unsqueeze(1) - posF1.unsqueeze(0)).pow(2), 2)
        D10 = torch.sum((posF1.unsqueeze(1) - posF0.unsqueeze(0)).pow(2), 2)

        D01min, D01ind = D01.min(1)
        D10min, D10ind = D10.min(1)

        if not isinstance(positive_pairs, np.ndarray):
            positive_pairs = np.array(positive_pairs, dtype=np.int64)

        pos_keys = _hash(positive_pairs, hash_seed)

        D01ind = sel1[D01ind.cpu().numpy()]
        D10ind = sel0[D10ind.cpu().numpy()]
        neg_keys0 = _hash([pos_ind0.numpy(), D01ind], hash_seed)
        neg_keys1 = _hash([D10ind, pos_ind1.numpy()], hash_seed)

        mask0 = torch.from_numpy(
            np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
        mask1 = torch.from_numpy(
            np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
        pos_loss = F.relu((posF0 - posF1).pow(2).sum(1) - pos_thresh)
        neg_loss0 = F.relu(neg_thresh - D01min[mask0]).pow(2)
        neg_loss1 = F.relu(neg_thresh - D10min[mask1]).pow(2)

        return pos_loss.mean(), (neg_loss0.mean() + neg_loss1.mean()) / 2


    def evaluate(self, F0, F1, pos_pairs):
        """
        Evaluates the hardest contrastive FCGF loss given current data

        Args:
            F0 (torch tensor): features of the source points [~b*n,c]
            F1 (torch tensor): features of the target points [~b*n,c]
            pos_pairs (torch tensor): indices of the positive pairs

        Returns:
            loss (torch tensor): mean value of the HC FCGF loss

        """
        pos_loss, neg_loss = self.contrastive_hardest_negative_loss(F0, F1, pos_pairs,
                                              num_pos=self.num_pos_per_batch *
                                                      self.batch_size,
                                              num_hn_samples=self.num_hn_samples_per_batch *
                                                             self.batch_size,
                                              pos_thresh=self.pos_thresh,
                                              neg_thresh=self.neg_thresh)

        loss = pos_loss + neg_loss

        return loss


class ClassificationLoss():
    """
    Classification loss class. Creates a ClassificationLoss object that is used to supervise the inlier/outlier classification of the putative correspondences.

    Args:
        cfg (dict): configuration parameters

    """
    def __init__(self, cfg):  
        self.w_class = cfg['loss']['loss_class']
        self.w_class = cfg['loss']['loss_class']
        self.compute_stats = cfg['train']['compute_precision']
        self.device = torch.device('cuda' if (torch.cuda.is_available() and cfg['misc']['use_gpu']) else 'cpu') 


    def class_loss(self, predicted, target):
        """
        Binary classification loss per putative correspondence.

        Args: 
            predicted (torch tensor): predicted weight per correspondence [b,n,1]
            target (torch tensor): ground truth label per correspondence (0 - outlier, 1 - inlier) [b,n,1]

        Return:
            class_loss (torch tensor): binary cross entropy loss [b]
        """

        loss = nn.BCELoss(reduction='none')  # Binary Cross Entropy loss, expects that the input was passed through the sigmoid
        sigmoid = nn.Sigmoid()
        
        predicted_labels = sigmoid(predicted).flatten().to(self.device)

        class_loss = loss(predicted_labels, target.flatten()).reshape(predicted.shape[0],-1)

        # Computing weights for compensating the class imbalance

        is_pos = (target.squeeze(-1) < 0.5).type(target.type())
        is_neg = (target.squeeze(-1) > 0.5).type(target.type())
        
        num_pos = torch.relu(torch.sum(is_pos, dim=1) - 1.0) + 1.0
        num_neg = torch.relu(torch.sum(is_neg, dim=1) - 1.0) + 1.0
        class_loss_p = torch.sum(class_loss * is_pos, dim=1)
        class_loss_n = torch.sum(class_loss * is_neg, dim=1)
        class_loss = class_loss_p * 0.5 / num_pos + class_loss_n * 0.5 / num_neg

        return class_loss

    def evaluate(self, predicted, target, scores=None):
        """
        Evaluates the binary cross entropy classification loss

        Args: 
            predicted (torch tensor): predicted logits per correspondence [b,n]
            target (torch tensor): ground truth label per correspondence (0 - outlier, 1 - inlier) [b,n,1]
            scores (torch tensor): predicted score (weight) per correspondence (0 - outlier, 1 - inlier) [b,n]

        Return:
            loss (torch tensor): mean binary cross entropy loss 
            precision (numpy array): Mean classification precision (inliers)
            recall (numpy array): Mean classification recall (inliers)
        """
        predicted = predicted.to(self.device)
        target = target.to(self.device)

        class_loss = self.class_loss(predicted, target)

        loss = torch.tensor([0.]).to(self.device)

        if self.w_class > 0:
            loss += torch.mean(self.w_class * class_loss)


        if self.compute_stats:
            assert scores != None, "If precision and recall should be computed, scores cannot be None!"

            y_predicted = scores.detach().cpu().numpy().reshape(-1)
            y_gt = target.detach().cpu().numpy().reshape(-1)

            precision, recall, f_measure, _ = precision_recall_fscore_support(y_gt, y_predicted.round(), average='binary')

            return loss, precision, recall

        else:
            return loss, None, None


class TransformationLoss():
    """
    Transformation loss class. Creates a TransformationLoss object that is used to supervise the rotation and translation estimation part of the network.

    Args:
        cfg (dict): configuration parameters

    """

    def __init__(self, cfg):  
        self.trans_loss_type = cfg['loss']['trans_loss_type']
        self.trans_loss_iter = cfg['loss']['trans_loss_iter']
        self.w_trans = cfg['loss']['loss_trans']
        self.device = torch.device('cuda' if (torch.cuda.is_available() and cfg['misc']['use_gpu']) else 'cpu')
        self.trans_loss_margin = cfg['misc']['trans_loss_margin']
        self.inlier_threshold = cfg['loss']['inlier_threshold']

    def trans_loss(self, x_in, rot_est, trans_est, gt_rot_mat, gt_t_vec):
        """
        Loss function on the transformation parameter. Based on the selected type of the loss computes either:
        0 - Vector distance between the point reconstructed using the EST transformation paramaters and the putative correspondence
        1 - Frobenius norm on the rotation matrix and L2 norm on the translation vector
        2 - L2 distance between the points reconstructed using the estimated and GT transformation paramaters
        3 - L1 distance between the points reconstructed using the estimated and GT transformation paramaters

        Args: 
            x_in (torch tensor): coordinates of the input point [b,1,n,6]
            rot_est (torch tensor): currently estimated rotation matrices [b,3,3]
            trans_est (torch tensor): currently estimated translation vectors [b,3,1]
            gt_rot_mat (torch tensor): ground truth rotation matrices [b,3,3]
            gt_t_vec (torch tensor): ground truth translation vectors [b,3,1]

        Return:
            r_loss (torch tensor): transformation loss if type 0 or 2 else Frobenius norm of the rotation matrices [b,1]
            t_loss (torch tensor): 0 if type 0, 2 or 3 else L2 norm of the translation vectors [b,1]
        """
        if self.trans_loss_type == 0:

            x2_reconstruct = torch.matmul(rot_est, x_in[:, 0, :, 0:3].transpose(1, 2)) + trans_est
            r_loss = torch.mean(torch.mean(torch.norm(x2_reconstruct.transpose(1,2) - x_in[:, :, :, 3:6], dim=(1)), dim=1))
            t_loss = torch.zeros_like(r_loss)

        elif self.trans_loss_type == 1:
            r_loss = torch.norm(gt_rot_mat - rot_est, dim=(1, 2))
            t_loss = torch.norm(trans_est - gt_t_vec,dim=1)  # Torch norm already does sqrt (p=1 for no sqrt)
        
        elif self.trans_loss_type == 2:
            x2_reconstruct_estimated = torch.matmul(rot_est, x_in[:, 0, :, 0:3].transpose(1, 2)) + trans_est
            x2_reconstruct_gt = torch.matmul(gt_rot_mat, x_in[:, 0, :, 0:3].transpose(1, 2)) + gt_t_vec

            r_loss = torch.mean(torch.norm(x2_reconstruct_estimated - x2_reconstruct_gt, dim=1), dim=1)
            t_loss = torch.zeros_like(r_loss)

        elif self.trans_loss_type == 3:
            x2_reconstruct_estimated = torch.matmul(rot_est, x_in[:, 0, :, 0:3].transpose(1, 2)) + trans_est
            x2_reconstruct_gt = torch.matmul(gt_rot_mat, x_in[:, 0, :, 0:3].transpose(1, 2)) + gt_t_vec

            r_loss = torch.mean(torch.sum(torch.abs(x2_reconstruct_estimated - x2_reconstruct_gt), dim=1), dim=1)
            t_loss = torch.zeros_like(r_loss)

        return r_loss, t_loss

    def evaluate(self, global_step, data, rot_est, trans_est):
        """
        Evaluates the pairwise loss function based on the current values

        Args: 
            global_step (int): current training iteration (used for controling which parts of the loss are used in the current iter) [1]
            data (dict): input data of the current batch 
            rot_est (torch tensor): rotation matrices estimated based on the current scores [b,3,3]
            trans_est  (torch tensor): translation vectors estimated based on the current scores [b,3,1]
        
        Return:
            loss (torch tensor): mean transformation loss of the current iteration over the batch
            loss_raw (torch tensor): mean transformation loss of the current iteration (return value for tenbsorboard before the trans loss is plugged in )
        """

        # Extract the current data  
        x_in, gt_R, gt_t = data['xs'].to(self.device), data['R'].to(self.device), data['t'].to(self.device)
        gt_inlier_ratio = data['inlier_ratio'].to(self.device)
        
        # Compute the transformation loss 
        r_loss, t_loss = self.trans_loss(x_in, rot_est, trans_est, gt_R, gt_t)

        # Extract indices of pairs with a minimum inlier ratio (do not propagate Transformation loss if point clouds do not overlap)
        idx_inlier_ratio = gt_inlier_ratio > self.inlier_threshold
        inlier_ratio_mask = torch.zeros_like(r_loss)
        inlier_ratio_mask[idx_inlier_ratio] = 1

        loss_raw = torch.tensor([0.]).to(self.device)


        if self.w_trans > 0:
            r_loss *= inlier_ratio_mask
            t_loss *= inlier_ratio_mask

            loss_raw += torch.mean(torch.min(self.w_trans * (r_loss + t_loss),  self.trans_loss_margin * torch.ones_like(t_loss)))

        # Check global_step and add essential loss
        loss = loss_raw if global_step >= self.trans_loss_iter else torch.tensor([0.]).to(self.device)

        return loss, loss_raw