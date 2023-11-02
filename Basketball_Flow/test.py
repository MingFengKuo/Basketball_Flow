import os
import math
import random
import warnings
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.backends import cudnn

from args import get_args
from utils import set_random_seed
from visualizer import visualize

from dataset import get_dataset, denormalize_pos
from models.BasketballFlow import BBGAME_CFVAEGAN

def main_worker(gpu, load_dir, ngpus_per_node, args):
    
    # basic setup
    cudnn.benchmark = True
    
    # resume model
    model = BBGAME_CFVAEGAN(args)
    model = model.cuda()
        
    model.load_state_dict(torch.load(os.path.join(load_dir, 'Basketball-Flow-999.pt')), strict=False)
    
    # initialize datasets and loaders
    train_data, test_data, norm_dict = get_dataset(args)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=1, shuffle=False, pin_memory=True)
    
    # test
    model.eval()
    with torch.no_grad():
        samples = 0
        max_samples = 50
        iterator = iter(test_loader)
        all_sample_data = []
        
        for data in tqdm(iterator):
            idx_batch, ref_data, ref_cond, mask = data['idx'], data['data'], data['cond'], data['mask']
            ref_data = ref_data.cuda() if args.gpu is None else ref_data.cuda(args.gpu)
            ref_cond = ref_cond.cuda() if args.gpu is None else ref_cond.cuda(args.gpu)
            mask     = mask.cuda() if args.gpu is None else mask.cuda(args.gpu)
            # sample
            sample_x = model.test_sample(ref_data, ref_cond, mask, norm_dict, samples=20)
            
            # denormalize
            sample_x = denormalize_pos(sample_x, norm_dict, dtype='smp') * mask
            
            # save
            all_sample_data.append(sample_x)
            
            # visualize
            data_length = int(mask.count_nonzero() / 22)
            path_length = math.ceil(data_length / 4)
            visualize(sample_x.cpu().numpy(), data_length, path_length,
                      save_path=os.path.join(load_dir, 'visualize_recon_' + str(samples)))
            
            samples += 1
            if samples >= max_samples:
                break
        
if __name__ == '__main__':
    # command line args
    args = get_args()
    load_dir = args.model_path
        
    if args.seed is None:
        args.seed = random.randint(0, 1000000)
    set_random_seed(args.seed)
    
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
        
    print("----------------------------------------")
    print("Arguments:")
    print(args)
    
    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, load_dir, ngpus_per_node, args)