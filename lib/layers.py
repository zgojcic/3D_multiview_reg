import torch
import torch.nn.functional as F
import numpy as np
import time 
from sklearn.neighbors import NearestNeighbors
from lib.utils import extract_mutuals, pairwise_distance, knn_point
#from Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops import pointnet2_utils


class Soft_NN(torch.nn.Module):
    """ Nearest neighbor class. Constructs either a stochastic (differentiable) or hard nearest neighbors layer.

    Args:
        corr_type (string): type of the NN search 
        st (bool): if straight through gradient propagation should be used (biased) (https://arxiv.org/abs/1308.3432)
        inv_temp (float): initial value for the inverse temperature used in softmax and gumbel_softmax

    """

    def __init__(self, corr_type='soft', st=True, inv_temp=10):
        super().__init__()

        assert corr_type in ['soft', 'hard', 'soft_gumbel'], 'Wrong correspondence type selected. Must be one of [soft, soft_gumbel, hard]' 

        if corr_type == 'hard':
            print('Gradients cannot be backpropagated to the feature descriptor because hard NN search is selected.')

        self.temp_inv = torch.nn.Parameter(torch.tensor([inv_temp], requires_grad=True, dtype=torch.float))

        self.corr_type = corr_type
        self.st = st



    def forward(self, x_f, y_f, y_c):
        """ Computes the correspondences in the feature space based on the selected parameters.

        Args:
            x_f (torch.tensor): infered features of points x [b,n,c] 
            y_f (torch.tensor): infered features of points y [b,m,c] 
            y_c (torch.tensor): coordinates of point y [b,m,3]

        Returns:
            x_corr (torch.tensor): coordinates of the feature based correspondences of points x [b,n,3]
         
        """

        dist = pairwise_distance(x_f,y_f)
        #dist_min = torch.min(dist, dim=2,keepdim=True).values
        #dist = dist - dist_min

        if self.corr_type == 'soft':

            y_soft = torch.softmax(-dist*self.temp_inv, dim=2)

            if self.st:
                # Straight through.
                index = y_soft.max(dim=2, keepdim=True)[1]
                y_hard = torch.zeros_like(y_soft).scatter_(dim=2, index=index, value=1.0)
                ret = y_hard - y_soft.detach() + y_soft

            else:
                ret = y_soft      

        elif self.corr_type == 'soft_gumbel':    

            if self.st:
                # Straight through.
                ret = F.gumbel_softmax(-dist, tau=1.0/self.temp_inv, hard=True)
            else:
                ret = F.gumbel_softmax(-dist, tau=1.0/self.temp_inv, hard=False)

        else:
            index = dist.min(dim=2, keepdim=True)[1]
            ret = torch.zeros_like(dist).scatter_(dim=2, index=index, value=1.0)


        # Compute corresponding coordinates
        x_corr = torch.matmul(ret, y_c)

        return x_corr

class Sampler(torch.nn.Module):
    """ Sampler class. Constructs a layer used to sample the points either based on their metric distance (FPS) or by randomly selecting them.

    Args:
        samp_type (string): type of the sampling to be used 
        st (bool): if straight through gradient propagation should be used (biased) (https://arxiv.org/abs/1308.3432)
        inv_temp (float): initial value for the inverse temperature used in softmax and gumbel_softmax

    """
    def __init__(self, samp_type='fps', targeted_num_points=2000):
        super().__init__()
        assert samp_type in ['fps', 'rand'], 'Wrong sampling type selected. Must be one of [fps, rand]' 


        self.samp_type = samp_type
        self.targeted_num_points = targeted_num_points


    def forward(self, input_C, input_F, pts_list):
        """ Samples the predifined points from the input point cloud and the corresponding feature descriptors.

        Args:
            input_C (torch.tensor): coordinates of the points [~b*n,3] 
            input_F (torch.tensor): infered features [~b*n,c]
            pts_list (list): list with the number of points of each point cloud in the batch

        Returns:
           sampled_C (torch tensor): coordinates of the sampled points [b,m,3]
           sampled_F (torch tensor): features of the sampled points [b,m,c]
         
        """

        # Sample the data
        idx_temp = []
        sampled_F = []
        sampled_C = []
        
        # Final number of points to be sampled is the min of the desired number of points and smallest number of point in the batch
        num_points = min(self.targeted_num_points, min(pts_list))

        for i in range(len(pts_list)):

            pcd_range = np.arange(sum(pts_list[:i]), sum(pts_list[:(i + 1)]), 1)

            if self.samp_type == 'fps':
                temp_pcd = torch.index_select(input_C, dim=0, index=torch.from_numpy(pcd_range).to(input_C).long())
                
                # Perform farthest point sampling on the current point cloud
                idxs = pointnet2_utils.furthest_point_sample(temp_pcd, num_points)
            
                # Move the indeces to the start of this point cloud
                idxs += pcd_range[0]

            elif self.samp_type == 'rand':
                # Randomly select the indices to keep
                idxs = torch.from_numpy(np.random.choice(pcd_range,num_points, replace=False)).to(input_C)

            sampled_F.append(torch.index_select(input_F, dim=0, index=idxs.long()))
            sampled_C.append(torch.index_select(input_C, dim=0, index=idxs.long()))

        return torch.stack(sampled_C, dim=0), torch.stack(sampled_F, dim=0) 


        

if __name__ == "__main__":

    test = torch.rand((3,10,10))
    test_1 = torch.rand((3,10,10))
    test_2 = torch.rand((3,10,3))

    soft_nn_1 = Soft_NN(corr_type='soft')
    soft_nn_2 = Soft_NN(corr_type='soft_gumbel')
    soft_nn_3 = Soft_NN(corr_type='hard')

    # Iterrative 
    neigh = NearestNeighbors()
    ret_iter = []
    array_input = test_1[0,:,:]
    for i in range(test.shape[0]):
        neigh.fit(test_1[i,:,:].cpu().numpy())
        idx = neigh.kneighbors(test[i,:,:].cpu().numpy(), n_neighbors=1, return_distance=False)

        ret_iter.append(test_2[i,idx.reshape(-1,),:])

    ret_iter = torch.stack(ret_iter)
    ret_1 = soft_nn_1(test,test_1,test_2)
    ret_2 = soft_nn_2(test,test_1,test_2)
    ret_3 = soft_nn_3(test,test_1,test_2)
    
    diff = ret_1 - ret_2
    diff_2 = ret_2 - ret_3
    diff_3 = ret_1 - ret_3
    diff_4 = ret_1 - ret_iter



    
    # Test the mutuals 
    pc_1 = torch.rand((5,2000,3)).cuda()
    pc_2 = torch.rand((5,2000,3)).cuda()
    pc_1_soft_c = torch.rand((5,2000,3)).cuda()
    pc_2_soft_c = torch.rand((5,2000,3)).cuda()

    test_mutuals = extract_mutuals(pc_1, pc_2, pc_1_soft_c, pc_2_soft_c)


    # Test the sampler 
    test_C = torch.rand(3000,3).float()
    test_F = torch.rand(3000,32).float()

    pts_list = [300,700,1000,400,600]
   
    # Test random sampling
    sampler = Sampler(targeted_num_points=100,samp_type='rand')
    sampled_C, sampled_F = sampler(test_C,test_F,pts_list)
    
    # Test fps
    sampler_fps = Sampler(targeted_num_points=100, samp_type='fps')

