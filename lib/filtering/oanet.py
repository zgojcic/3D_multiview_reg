"""
Extension of the filtering network proposed in Learning Two-View Correspondences and Geometry Using Order-Aware Network (ICCV 2019),
to 3D correspondence filtering. Source coude based on the OANet repository: https://github.com/zjhthu/OANet.

If you use in your project pelase consider also citing: https://arxiv.org/pdf/1908.04964.pdf

"""

import torch
import torch.nn as nn
from lib.utils import kabsch_transformation_estimation
import logging

# If the BN stat should be tracked and used in the inference mode
BN_TRACK_STATS = True


class PointCN(nn.Module):
    def __init__(self, channels, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
            out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(channels),
            nn.BatchNorm2d(channels, track_running_stats=BN_TRACK_STATS),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size=1),
            nn.InstanceNorm2d(out_channels),
            nn.BatchNorm2d(out_channels, track_running_stats=BN_TRACK_STATS),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        out = self.conv(x)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out


class trans(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class OAFilter(nn.Module):
    def __init__(self, channels, points, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
            out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels, track_running_stats=BN_TRACK_STATS),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size=1),  # b*c*n*1
            trans(1, 2))

        # Spatial Correlation Layer
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(points, track_running_stats=BN_TRACK_STATS),
            nn.ReLU(),
            nn.Conv2d(points, points, kernel_size=1)
        )
        self.conv3 = nn.Sequential(
            trans(1, 2),
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            nn.BatchNorm2d(out_channels, track_running_stats=BN_TRACK_STATS),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out


class diff_pool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel, track_running_stats=BN_TRACK_STATS),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x):
        embed = self.conv(x)  # b*k*n*1
        S = torch.softmax(embed, dim=2).squeeze(3)
        out = torch.matmul(x.squeeze(3), S.transpose(1, 2)).unsqueeze(3)
        return out


class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel, track_running_stats=BN_TRACK_STATS),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x_up, x_down):
        #x_up: b*c*n*1
        #x_down: b*c*k*1
        embed = self.conv(x_up)  # b*k*n*1
        S = torch.softmax(embed, dim=1).squeeze(3)  # b*k*n
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
        return out


class OANBlock(nn.Module):
    def __init__(self, net_channels, input_channel, depth, clusters, normalize_w):
        nn.Module.__init__(self)
        channels = net_channels
        self.layer_num = depth
        logging.info('OANET: channels:' + str(channels) + ', layer_num:' + str(self.layer_num))
        self.conv1 = nn.Conv2d(input_channel, channels, kernel_size=1)

        l2_nums = clusters

        self.l1_1 = []
        for _ in range(self.layer_num//2):
            self.l1_1.append(PointCN(channels))

        self.down1 = diff_pool(channels, l2_nums)

        self.l2 = []
        for _ in range(self.layer_num//2):
            self.l2.append(OAFilter(channels, l2_nums))

        self.up1 = diff_unpool(channels, l2_nums)

        self.l1_2 = []
        self.l1_2.append(PointCN(2*channels, channels))
        for _ in range(self.layer_num//2-1):
            self.l1_2.append(PointCN(channels))

        self.l1_1 = nn.Sequential(*self.l1_1)
        self.l1_2 = nn.Sequential(*self.l1_2)
        self.l2 = nn.Sequential(*self.l2)

        self.output = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, data, xs):
        #data: b*c*n*1
        x1_1 = self.conv1(data)
        x1_1 = self.l1_1(x1_1)
        x_down = self.down1(x1_1)
        x2 = self.l2(x_down)
        x_up = self.up1(x1_1, x2)
        out = self.l1_2(torch.cat([x1_1, x_up], dim=1))

        logits = torch.squeeze(torch.squeeze(self.output(out), 3), 1)
        weights = torch.relu(torch.tanh(logits))

        if torch.any(torch.sum(weights, dim=1) == 0.0):
            weights = weights + 1/weights.shape[1]

        x1, x2 = xs[:, 0, :, :3], xs[:, 0, :, 3:]

        rotation_est, translation_est, residuals, gradient_not_valid = kabsch_transformation_estimation(
            x1, x2, weights)

        return logits, weights, rotation_est, translation_est, residuals, out, gradient_not_valid


class OANet(nn.Module):
    """
    OANet filtering class. Build an OAnet object that represent the extension of the filtering network proposed in
    (https://arxiv.org/abs/1908.04964) to the problem of 3D correspondence filtering.

    The local context is aggregated using Context normalization and Order-Aware clustering blocks.

    Args:
        cfg (dict): configuration parameter

    """
    def __init__(self, cfg):
        nn.Module.__init__(self
        )
        
        self.iter_num = cfg['misc']['iter_num']
        depth_each_stage = cfg['misc']['net_depth']//(cfg['misc']['iter_num']+1)
        self.side_channel = (cfg['data']['use_mutuals'] == 2)

        self.reg_init = OANBlock(cfg['misc']['net_channel'], 6 + self.side_channel,
                                 depth_each_stage, cfg['misc']['clusters'], cfg['misc']['normalize_weights'])
        
        self.reg_iter = [OANBlock(cfg['misc']['net_channel'], 8 + self.side_channel, depth_each_stage, cfg['misc']['clusters'], cfg['misc']['normalize_weights'])
                             for _ in range(self.iter_num)]

        self.reg_iter = nn.Sequential(*self.reg_iter)

        self.device = torch.device('cuda' if (torch.cuda.is_available() and cfg['misc']['use_gpu']) else 'cpu')


    def forward(self, data):
        """
        For each of the putative correspondences infers a weight [0,1] denoting if the correspondence is an inlier (1) 
        or an outlier (0). Based on the weighted Kabsch algorithm it additionally estimates the pairwise rotation matrix
        and translation parameters. 

        Args:
        data (dict): dictionariy of torch tensors representing the input data

        Returns:
        output (duct): dictionary of output data
        
        """

        assert data['xs'].dim() == 4 and data['xs'].shape[1] == 1
        #data: b*1*n*c
        input_data = data['xs'].transpose(1, 3).to(self.device)

        res_logits, res_scores, res_rot_est, res_trans_est = [], [], [], []

        # First pass through the network
        logits, scores, rot_est, trans_est, residuals, latent_features, gradient_not_valid = self.reg_init(
            input_data, data['xs'].to(self.device))

        res_logits.append(logits), res_scores.append(scores), res_rot_est.append(rot_est), res_trans_est.append(trans_est)

        # If iterative approach then append residuals and scores and perform additional passes
        for i in range(self.iter_num):
            logits, scores, rot_est, trans_est, residuals, latent_features, temp_gradient_not_valid = self.reg_iter[i](
                                                            torch.cat([input_data, residuals.detach().unsqueeze(1).unsqueeze(3),
                                                            scores.unsqueeze(1).unsqueeze(3)], dim=1), data['xs'].to(self.device))

            gradient_not_valid = (temp_gradient_not_valid or gradient_not_valid)

            res_logits.append(logits), res_scores.append(
                scores), res_rot_est.append(rot_est), res_trans_est.append(trans_est)


        # Construct the output 
        output = {}
        output['logits'] = res_logits
        output['scores'] = res_scores
        output['rot_est'] = res_rot_est
        output['trans_est'] = res_trans_est
        output['latent features'] = latent_features
        output['gradient_flag'] = gradient_not_valid

        return output
