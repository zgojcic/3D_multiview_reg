import numpy as np
import glob
import logging
import os
import torch.utils.data as data
from lib.utils import augment_precomputed_data, add_jitter, get_file_list, get_folder_list, read_trajectory
import torch 

def collate_fn(batch):
    data = {}

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


    for key in batch[0]:
        data[key] = []

    for sample in batch:
        for key in sample:
            if isinstance(sample[key], list):
                data[key].append(sample[key])
            else:
                data[key].append(to_tensor(sample[key]))

    for key in data:
        if isinstance(data[key][0], torch.Tensor):
            data[key] = torch.stack(data[key])

    return data


class PrecomputedIndoorDataset(data.Dataset):
    """
    Dataset class for precomputed FCGF descrptors and established correspondences. Used to train the Registration blocks,
    confidence block, and transformation synchronization layer. If using this dataset the method descriptor argument and 
    train_desc flag in the arguments have to be set to null and False respectively. Eases the training of the second part 
    as the overlaping pairs that build a graph can be presampled

    """
    
    def __init__(self, phase, config):
        
        self.files = []
        self.random_shuffle = config['data']['shuffle_examples']
        self.root = config['data']['root']
        self.config = config
        self.data = None
        self.randng = np.random.RandomState()
        self.use_mutuals = config['data']['use_mutuals']
        self.dist_th = config['data']['dist_th']
        self.max_num_points = config['data']['max_num_points']
        self.augment_data = config['data']['augment_data']
        self.jitter = config['data']['jitter']

        self.device = torch.device('cuda' if (torch.cuda.is_available() and config['misc']['use_gpu']) else 'cpu') 

        logging.info("Loading the subset {} from {}!".format(phase,self.root))

        subset_names = open(self.DATA_FILES[phase]).read().split()

        for name in subset_names:
            self.files.append(name)

    def __getitem__(self, idx):

        file = os.path.join(self.root,self.files[idx])
        data = np.load(file)

        file_name = file.replace(os.sep,'/').split('/')[-1]

        xs = data['x']
        ys = data['y']
        Rs = data['R']
        ts = np.expand_dims(data['t'], -1)

        mutuals = data['mutuals']
        inlier_ratio = data['inlier_ratio'] # Thresholded at 5cm deviation
        inlier_ratio_mutuals = data['inlier_ratio_mutuals'] # Thresholded at 5cm deviation
        overlap = data['overlap'] # Max overlap


        # Shuffle the examples
        if self.random_shuffle:
            if xs.shape[0] >= self.max_num_points:
                sample_idx = np.random.choice(xs.shape[0], self.max_num_points, replace=False)
            else:
                sample_idx = np.concatenate((np.arange(xs.shape[0]),
                                np.random.choice(xs.shape[0], self.max_num_points-xs.shape[0], replace=True)),axis=-1)

            xs = xs[sample_idx,:]
            ys = ys[sample_idx]
            mutuals = mutuals[sample_idx]


        # Check if the the mutuals or the ratios should be used
        side = []
        if self.use_mutuals == 0:
            pass
        elif self.use_mutuals == 1:
            mask = mutuals.reshape(-1).astype(bool)
            xs = xs[mask,:]
            ys = ys[mask]
        elif self.use_mutuals == 2:
            side.append(mutuals.reshape(-1,1))
            side = np.concatenate(side,axis=-1)
        else:
            raise NotImplementedError

        # Augment the data augmentation 
        if self.augment_data:
            xs, Rs, ts = augment_precomputed_data(xs, Rs, ts)

        if self.jitter:
            xs, ys = add_jitter(xs, Rs, ts)


        # Threshold ys based on the distance threshol
        ys_binary = (ys < self.dist_th).astype(xs.dtype)

        if not side:
            side = np.array([0])

        # Prepare data
        xs = np.expand_dims(xs,0)
        ys = np.expand_dims(ys_binary,-1)


        return {'R': Rs, 
                't': ts,
                'xs': xs,
                'ys': ys,
                'side': side,
                'overlap': overlap,
                'inlier_ratio': inlier_ratio}

    def __len__(self):
        return len(self.files)

    def reset_seed(self,seed=41):
        logging.info('Resetting the data loader seed to {}'.format(seed))
        self.randng.seed(seed)






class PrecomputedPairwiseEvalDataset(data.Dataset):
    """
    Dataset class for evaluating the pairwise registration based on the precomputed feature correspondences

    """
    def __init__(self, args):
        
        self.files = []
        self.root = args.source_path
        self.use_mutuals = args.mutuals
        save_path = os.path.join(self.root, 'results', args.method)
        save_path += '/mutuals/' if args.mutuals else '/all/'
    

        logging.info("Loading the eval data from {}!".format(self.root))

        scene_names = get_folder_list(os.path.join(self.root,'correspondences'))

        for folder in scene_names:
            curr_scene_name = folder.split('/')[-1]
            if os.path.exists(os.path.join(save_path,curr_scene_name,'traj.txt')) and not args.overwrite:
                logging.info('Trajectory for scene {} already exists and will not be recomputed.'.format(curr_scene_name))
            else:
                if args.only_gt_overlaping:
                    gt_pairs, gt_traj = read_trajectory(os.path.join(self.root,'raw_data', curr_scene_name, "gt.log"))
                    for idx_1, idx_2, _ in gt_pairs:
                        self.files.append(os.path.join(folder,curr_scene_name + '_{}_{}.npz'.format(str(idx_1).zfill(3),str(idx_2).zfill(3))))

                else:
                    corr_files = get_file_list(folder)
                    for corr in corr_files:
                        self.files.append(corr)
    
    def __getitem__(self, idx):

        curr_file = os.path.join(self.files[idx])
        data = np.load(curr_file)

        idx_1 = str(curr_file.split('_')[-2])
        idx_2 = str(curr_file.split('_')[-1].split('.')[0])
        curr_scene_name = curr_file.split('/')[-2]
        metadata = [curr_scene_name, idx_1, idx_2]
        xs = data['x']

        xyz1 = np.load(os.path.join(self.root,'features',curr_scene_name, curr_scene_name + '_{}.npz'.format(idx_1)))['xyz']
        xyz2 = np.load(os.path.join(self.root,'features',curr_scene_name, curr_scene_name + '_{}.npz'.format(idx_2)))['xyz']

        if self.use_mutuals == 1:
            mutuals = data['mutuals']
            xs = xs[mutuals.astype(bool).reshape(-1), :]

        return {'xs': np.expand_dims(xs,0),
                'metadata':metadata,
                'idx':np.array(idx),
                'xyz1': [xyz1],
                'xyz2': [xyz2]}

    def __len__(self):
        test = len(self.files)
        return len(self.files)

    def reset_seed(self,seed=41):
        logging.info('Resetting the data loader seed to {}'.format(seed))
        self.randng.seed(seed)






### NEEDS TO BE IMPLEMENTED ###
class RawIndoorDataset(data.Dataset):
    def __init__(self, phase, config):
        
        self.files = []
        self.random_shuffle = config['data']['shuffle_examples']
        self.root = config['data']['root']
        self.config = config
        self.data = None
        self.randng = np.random.RandomState()
        self.use_ratio = config['misc']['use_ratio']
        self.use_ratio_tf = config['misc']['use_ratio_th']
        self.use_mutuals = config['misc']['use_mutuals']
        self.dist_th = config['misc']['dist_th']
        self.max_num_points = config['misc']['max_num_points']

        self.device = torch.device('cuda' if (torch.cuda.is_available() and config['misc']['use_gpu']) else 'cpu') 

        logging.info("Loading the subset {} from {}!".format(phase,self.root))


        subset_names = open(self.DATA_FILES[phase]).read().split()

        for name in subset_names:
            self.files.append(name)

    def __getitem__(self, idx):

        file = os.path.join(self.root,self.files[idx])
        data = np.load(file)

        file_name = file.replace(os.sep,'/').split('/')[-1]

        xs = data['x']
        ys = data['y']
        Rs = data['R']
        ts = data['t']
        ratios = data['ratios']
        mutuals = data['mutuals']
        inlier_ratio = data['inlier_ratio'] # Thresholded at 5cm deviation
        inlier_ratio_mutuals = data['inlier_ratio_mutuals'] # Thresholded at 5cm deviation
        overlap = data['overlap'] # Max overlap


        # Shuffle the examples
        if self.random_shuffle:
            if xs.shape[0] >= self.max_num_points:
                sample_idx = np.random.choice(xs.shape[0], self.max_num_points, replace=False)
            else:
                sample_idx = np.concatenate((np.arange(xs.shape[0]),
                                np.random.choice(xs.shape[0], self.max_num_points-xs.shape[0], replace=True)),axis=-1)

            xs = xs[sample_idx,:]
            ys = ys[sample_idx]
            ratios = ratios[sample_idx]
            mutuals = mutuals[sample_idx]


        # Check if the the mutuals or the ratios should be used
        side = []
        if self.use_ratio == 0 and self.use_mutuals == 0:
            pass
        elif self.use_ratio == 1 and self.use_mutuals == 0:
            mask = ratios.reshape(-1)  < self.use_ratio_tf
            xs = xs[mask,:]
            ys = ys[mask]
        elif self.use_ratio == 0 and self.use_mutuals == 1:
            mask = mutuals.reshape(-1).astype(bool)
            xs = xs[mask,:]
            ys = ys[mask]
        elif self.use_ratio == 2 and self.use_mutuals == 2:
            side.append(ratios.reshape(-1,1)) 
            side.append(mutuals.reshape(-1,1))
            side = np.concatenate(side,axis=-1)
        else:
            raise NotImplementedError

        # Threshold ys based on the distance threshol
        ys_binary = (ys < self.dist_th).astype(xs.dtype)

        if not side:
            side = np.array([0])


        return {'R': Rs, 
                't': np.expand_dims(ts, -1),
                'xs': np.expand_dims(xs, 0),
                'ys': np.expand_dims(ys_binary, -1),
                'side': side,
                'overlap': overlap,
                'inlier_ratio': inlier_ratio}


    def __len__(self):
        return len(self.files)

    def reset_seed(self,seed=41):
        logging.info('Resetting the data loader seed to {}'.format(seed))
        self.randng.seed(seed)

class Precomputed3DMatch(PrecomputedIndoorDataset):
    # 3D Match dataset all files
    DATA_FILES = {
        'train': './configs/3d_match/3d_match_all_train.txt',
        'val': './configs/3d_match/3d_match_all_valid.txt',
        'test': './configs/3d_match/test_all.txt'
    }

class Precomputed3DMatchFiltered(PrecomputedIndoorDataset):
    # 3D Match dataset with only overlaping point cloud and with examples 
    # that have more than 5% inliers (see dataset readme for more info)

    DATA_FILES = {
        'train': './configs/3d_match/3DMatch_filtered_train.txt',
        'val': './configs/3d_match/3DMatch_filtered_valid.txt',
        'test': './configs/3d_match/test_all.txt'
    }


# Map the datasets to string names
ALL_DATASETS = [Precomputed3DMatch, Precomputed3DMatchFiltered]
dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}


def make_data_loader(config, phase, shuffle_dataset=None):
    """
    Defines the data loader based on the parameters specified in the config file
    Args:
        config (dict): dictionary of the arguments
        phase (str): phase for which the data loader should be initialized in [train,val,test]
        shuffle_dataset (bool): shuffle the dataset or not

    Returns:
        loader (torch data loader): data loader that handles loading the data to the model
    """

    assert config['misc']['run_mode'] in ['train','val','test']

    if shuffle_dataset is None:
        shuffle_dataset = shuffle_dataset != 'test'

    # Select the defined dataset
    Dataset = dataset_str_mapping[config['data']['dataset']]

    dset = Dataset(phase, config=config)

    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=config[phase]['batch_size'],
        shuffle=shuffle_dataset,
        num_workers=config[phase]['num_workers'],
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True
    )

    return loader
