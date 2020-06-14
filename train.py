import sys
import os 
import logging 
import torch
import time
import argparse
import numpy as np
import torch.optim as optim
from tensorboardX import SummaryWriter

import lib.config as config
from lib.utils import load_config
from lib.data import make_data_loader
from lib.checkpoints import CheckpointIO
from lib.logger import prepare_logger


# Set the random seeds for reproducibility
np.random.seed(41)
torch.manual_seed(41)
if torch.cuda.is_available():
    torch.cuda.manual_seed(41)

  
def main(cfg, logger):
    """
    Main function of this software. After preparing the data loaders, model, optimizer, and trainer,
    start with the training and evaluation process.

    Args:
        cfg (dict): current configuration paramaters
    """

    # Initialize parameters
    model_selection_metric = cfg['train']['model_selection_metric']

    if cfg['train']['model_selection_mode'] == 'maximize':
        model_selection_sign = 1
    elif cfg['train']['model_selection_mode'] == 'minimize':
        model_selection_sign = -1
    else:
        raise ValueError('model_selection_mode must be either maximize or minimize.')

    # Get data loader
    train_loader = make_data_loader(cfg, phase='train')
    val_loader = make_data_loader(cfg, phase='val')

    # Set up tensorboard logger
    tboard_logger = SummaryWriter(os.path.join(cfg['misc']['log_dir'], 'logs'))

    # Get model
    model = config.get_model(cfg)

    # Get optimizer and trainer
    optimizer = getattr(optim, cfg['optimizer']['alg'])(model.parameters(), lr=cfg['optimizer']['learning_rate'],
                                                            weight_decay=cfg['optimizer']['weight_decay'])

    trainer = config.get_trainer(cfg, model, optimizer, tboard_logger)

    # Load pre-trained model if existing
    kwargs = {
    'model': model,
    'optimizer': optimizer,
    }

    checkpoint_io = CheckpointIO(cfg['misc']['log_dir'], initialize_from=cfg['model']['init_from'],
                                 initialization_file_name=cfg['model']['init_file_name'], **kwargs)
    
    try:
        load_dict = checkpoint_io.load('model.pt')
    except FileExistsError:
        load_dict = dict()
    
    epoch_it = load_dict.get('epoch_it', -1)
    it = load_dict.get('it', -1)

    metric_val_best = load_dict.get('loss_val_best', -model_selection_sign * np.inf)

    if metric_val_best == np.inf or metric_val_best == -np.inf:
        metric_val_best = -model_selection_sign * np.inf

    logger.info('Current best validation metric ({}): {:.5f}'.format(model_selection_metric, metric_val_best))

    # Training parameters
    stat_interval = cfg['train']['stat_interval']
    stat_interval = stat_interval if stat_interval > 0 else abs(stat_interval* len(train_loader))

    chkpt_interval = cfg['train']['chkpt_interval']
    chkpt_interval = chkpt_interval if chkpt_interval > 0 else abs(chkpt_interval* len(train_loader))

    val_interval = cfg['train']['val_interval']
    val_interval = val_interval if val_interval > 0 else abs(val_interval* len(train_loader))

    # Print model parameters and model graph
    nparameters = sum(p.numel() for p in model.parameters())
    #print(model)
    logger.info('Total number of parameters: {}'.format(nparameters))

    # Training loop
    while epoch_it < cfg['train']['max_epoch']:
        epoch_it += 1

        for batch in train_loader:
            it += 1
            loss = trainer.train_step(batch, it)
            tboard_logger.add_scalar('train/loss', loss, it)

            # Print output
            if stat_interval != 0 and (it % stat_interval) == 0  and it != 0:
                logger.info('[Epoch {}] it={}, loss={:.4f}'.format(epoch_it, it, loss))

            # Save checkpoint
            if (chkpt_interval != 0 and (it % chkpt_interval) == 0) and it != 0:
                logger.info('Saving checkpoint')
                checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                                loss_val_best=metric_val_best)

            # Run validation
            if val_interval != 0 and (it % val_interval) == 0 and it != 0:
                eval_dict = trainer.evaluate(val_loader,it)

                metric_val = eval_dict[model_selection_metric]
                logger.info('Validation metric ({}): {:.4f}'.format(model_selection_metric, metric_val))

                for k, v in eval_dict.items():
                    tboard_logger.add_scalar('val/{}'.format(k), v, it)

                if model_selection_sign * (metric_val - metric_val_best) > 0:
                    metric_val_best = metric_val
                    logger.info('New best model (loss {:.4f})'.format(metric_val_best))
                    checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                    loss_val_best=metric_val_best)


    # Quit after the maximum number of epochs is reached
    logger.info('Training completed after {} Epochs ({} it) with best val metric ({})={}'.format(epoch_it, it, model_selection_metric, metric_val_best))

if __name__ == "__main__":
    logger = logging.getLogger


    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help= 'Path to the config file.')
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Create the output dir if it does not exist 
    if not os.path.exists(cfg['misc']['log_dir']):
        os.makedirs(cfg['misc']['log_dir'])

    logger, checkpoint_dir = prepare_logger(cfg,cfg['misc']['log_path'])
    
    cfg['misc']['log_dir'] = checkpoint_dir
    # Argument: path to the config file
    logger.info('Torch version: {}'.format(torch.__version__))

    main(cfg, logger)
