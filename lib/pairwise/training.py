import torch 
import os 
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from lib.loss import DescriptorLoss, TransformationLoss, ClassificationLoss


class Trainer():
    ''' 
    Trainer class of the pairwise registration network.

    Args:
        cfg (dict): configuration parameters
        model (nn.Module): PairwiseReg model 
        optimizer (optimizer): PyTorch optimizer
        tboard_logger (tensorboardx instance): TensorboardX logger used to track train and val stats
        device (pytorch device)
    '''

    def __init__(self, cfg, model, optimizer, tboard_logger, device):

        self.model = model
        self.optimizer = optimizer
        self.tboard_logger = tboard_logger
        self.device = device
        self.loss_desc = cfg['loss']['loss_desc']

        # Initialize the loss classes based on the input paramaters
        self.DescriptorLoss = DescriptorLoss(cfg)
        self.ClassificationLoss = ClassificationLoss(cfg)
        self.TransformationLoss = TransformationLoss(cfg)
        


    def train_step(self, data, global_step):
        ''' 
        Performs a single training step.
        
        Args:
            data (dict): data dictionary
            global_step (int): current training iteration
        
        '''
        
        self.model.train()
        self.optimizer.zero_grad()
        backprop_flag = False

        loss, gradient_flag = self.compute_loss(data, global_step)
        loss.backward()

        # Only update the parameters if there were no problems in the forward pass (mostly SVD)
        # Check if any of the gradients is NaN
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.any(torch.isnan(param.grad)):
                    print('Gradients include NaN values. Parameters will not be updated.')
                    backprop_flag = True
                    break
        
        if not (backprop_flag or gradient_flag):
            self.optimizer.step()

        return loss.item()


    def eval_step(self, data, global_step):
        ''' 
        Performs a single evaluation step.
        
        Args:
            data (dict): data dictionary
            global_step (int): current training iteration

        Return?
            eval_dict (dict): evaluation data of the current val batch
        
        '''

        self.model.eval()
        eval_dict = {}

        with torch.no_grad():
            # Extract the feature descriptors and correspondences
            filtering_input, F0, F1 = self.model.compute_descriptors(data)
            # Filter the correspondences and estimate the pairwise transformation parameters
            filtered_output = self.model.filter_correspondences(filtering_input)
        
            # Losses 
            # Descriptor loss
            desc_loss = torch.tensor([0.]).to(self.device)
            if self.loss_desc and F0 is not None:
                desc_loss = self.DescriptorLoss.evaluate(F0, F1, data['correspondences'])
                eval_dict['desc_loss'] = desc_loss

            # Classification and transformation loss
            class_loss = []
            trans_loss = []
            trans_loss_raw = []
            
            for i in range(len(filtered_output['rot_est'])):
                # Classification loss
                temp_class_loss, precision, recall = self.ClassificationLoss.evaluate(filtered_output['logits'][i], filtering_input['ys'], filtered_output['scores'][i])
                class_loss.append(temp_class_loss)

                # Transformation loss
                temp_trans_loss, temp_trans_loss_raw = self.TransformationLoss.evaluate(global_step, filtering_input, filtered_output['rot_est'][i], filtered_output['trans_est'][i])
                trans_loss.append(temp_trans_loss)
                trans_loss_raw.append(temp_trans_loss_raw)
            

        trans_loss_raw = torch.mean(torch.stack(trans_loss_raw))
        class_loss = torch.mean(torch.stack(class_loss))
        trans_loss = torch.mean(torch.stack(trans_loss))

        loss = desc_loss + class_loss + trans_loss

        
        eval_dict['class_loss'] = class_loss.item()
        eval_dict['trans_loss'] = trans_loss_raw.item()
        eval_dict['loss'] = loss.item()

        # If precision and recall stats are computed add them to the stats
        if precision:
            eval_dict['precision'] = precision
            eval_dict['recall'] = recall

        return eval_dict



    def compute_loss(self, data, global_step):
        ''' 
        Computes the combined loss (descriptor, classification, and transformation).
        
        Args:
            data (dict): data dictionary
            global_step (int): current training iteration

        Return:
            loss (torch tensor): combined loss values of the current batch
            gradient_flag (bool): flag denoting if the SVD estimation had any problem

        '''

        # Extract the feature descriptors and correspondences
        filtering_input, F0, F1 = self.model.compute_descriptors(data)

        # Filter the correspondences and estimate the pairwise transformation parameters
        filtered_output = self.model.filter_correspondences(filtering_input)
        
        # Losses 
        # Descriptor loss
        desc_loss = torch.tensor([0.]).to(self.device)
        if self.loss_desc and F0 is not None:
            desc_loss = self.DescriptorLoss.evaluate(F0, F1, data['correspondences'])
            self.tboard_logger.add_scalar('train/desc_loss', desc_loss, global_step)

        # Classification and transformation loss
        class_loss_iter = []
        trans_loss_iter = []
        trans_loss_raw_iter = []
        precision_iter = []
        recall_iter = []

        for i in range(len(filtered_output['rot_est'])):
            # Classification loss
            temp_class_loss, precision, recall = self.ClassificationLoss.evaluate(filtered_output['logits'][i], filtering_input['ys'], filtered_output['scores'][i])
            class_loss_iter.append(temp_class_loss)
            precision_iter.append(precision)
            recall_iter.append(recall)

            # Transformation loss
            temp_trans_loss, temp_trans_loss_raw = self.TransformationLoss.evaluate(global_step, filtering_input, filtered_output['rot_est'][i], filtered_output['trans_est'][i])
            trans_loss_iter.append(temp_trans_loss)
            trans_loss_raw_iter.append(temp_trans_loss_raw)

        trans_loss_raw = torch.mean(torch.stack(trans_loss_raw_iter))
        class_loss = torch.mean(torch.stack(class_loss_iter))
        trans_loss = torch.mean(torch.stack(trans_loss_iter))

        loss = desc_loss + class_loss + trans_loss

        # Print out the stats
        for i in range(len(class_loss_iter)):
            self.tboard_logger.add_scalar('train/class_loss_{}'.format(i), class_loss_iter[i].item(), global_step)
            self.tboard_logger.add_scalar('train/trans_loss_{}'.format(i), trans_loss_iter[i].item(), global_step)
            self.tboard_logger.add_scalar('train/trans_loss_raw_{}'.format(i), trans_loss_raw_iter[i].item(), global_step)

            # If precision and recall are computed, log them
            if precision:
                self.tboard_logger.add_scalar('train/precision_{}'.format(i), precision_iter[i], global_step)
                self.tboard_logger.add_scalar('train/recall_{}'.format(i), recall_iter[i], global_step)

        return loss, filtered_output['gradient_flag'] 




    def evaluate(self, val_loader, global_step):
        ''' 
        Performs the evaluation over the whole evaluation dataset.
        Args:
            val_loader (Pytorch dataloader): dataloader of the validation dataset
            global_step (int): current iteration

        Returns:
            eval_dict (defaultdict): evaluation values for the current validation epoch
        '''


        eval_list = defaultdict(list)

        for data in tqdm(val_loader):
            eval_step_dict = self.eval_step(data, global_step)

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

        
        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        
        return eval_dict
