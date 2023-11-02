import os
import sys
import time
import warnings
import faulthandler

import torch
import random

from torch.backends import cudnn
from tensorboardX import SummaryWriter

from args import get_args
from dataset import get_dataset

from utils import AverageValueMeter, set_random_seed, validate
from models.BasketballFlow import Basketball_Flow

faulthandler.enable()

def main_worker(gpu, save_dir, ngpus_per_node, args):
    print("----------------------------------------")
    # basic setup
    cudnn.benchmark = True
    
    if args.log_name is not None:
        log_dir = args.log_name
    else:
        log_dir = "runs/time-%d" % time.time()
        
    # writer
    writer = SummaryWriter(log_dir=log_dir)

    # model
    model = Basketball_Flow(args)
    model = model.cuda()
        
    # initialize datasets and loaders
    train_data, test_data, norm_dict = get_dataset(args)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        
    # main training loop
    start_time = time.time()
    
    # Generate loss
    recon_avg_meter = AverageValueMeter()
    kl_div_avg_meter = AverageValueMeter()
    critic_avg_meter = AverageValueMeter()
    sample_avg_meter = AverageValueMeter()
    # Feature loss
    dribbler_avg_meter = AverageValueMeter()
    blocked_avg_meter = AverageValueMeter()
    ball_pass_avg_meter = AverageValueMeter()
    vel_avg_meter = AverageValueMeter()
    acc_avg_meter = AverageValueMeter()
    # Sketch Reconstruction loss
    sket_recon_avg_meter = AverageValueMeter()
    stat_recon_avg_meter = AverageValueMeter()
    # Prior loss
    prior_avg_meter = AverageValueMeter()

    start_epoch = 0
    print("Start epoch: %d" % start_epoch)
    print("End   epoch: %d" % args.epochs)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        for bidx, data in enumerate(train_loader):
            idx_batch, tr_data, tr_cond, tr_mask = data['idx'], data['data'], data['cond'], data['mask']
            step = bidx + len(train_loader) * epoch
            data = tr_data.cuda(args.gpu, non_blocking=True)
            cond = tr_cond.cuda(args.gpu, non_blocking=True)
            mask = tr_mask.cuda(args.gpu, non_blocking=True)
            
            # Train D
            if epoch < args.critic_pretrain_epochs:
                for _ in range(args.critic_pretrain_iteration):
                    model.train_D(data, cond, mask)
            else:
                for _ in range(args.critic_iteration):
                    model.train_D(data, cond, mask)
            
            # Train G
            out = model.train_G(data, cond, mask, norm_dict, step, writer=writer)
            # Generate loss
            recon_avg_meter.update(out['recon'])
            kl_div_avg_meter.update(out['kl-div'])
            critic_avg_meter.update(out['critic'])
            sample_avg_meter.update(out['sample'])
            # Feature loss
            dribbler_avg_meter.update(out['dribbler'])
            blocked_avg_meter.update(out['blocked'])
            ball_pass_avg_meter.update(out['ball_pass'])
            vel_avg_meter.update(out['velocity'])
            acc_avg_meter.update(out['acceleration'])
            # Sketch Reconstruct loss
            sket_recon_avg_meter.update(out['sketch_recon'])
            stat_recon_avg_meter.update(out['status_recon'])
            
            # Train F
            out = model.train_F(data, cond, mask, norm_dict, step, writer=writer)
            # Prior loss
            prior_avg_meter.update(out['prior'])
            
            if step % args.log_freq == 0:
                duration = time.time() - start_time
                start_time = time.time()
                print("****************************************")
                print("Train Epoch %d Batch [%2d/%2d] Time [%3.2fs]" % (epoch, bidx, len(train_loader), duration))
                print("****************************************")
                print("Sample")
                print("----------------------------------------")
                print("Recon  %2.5f" % recon_avg_meter.avg)
                print("KL-div %2.5f" % kl_div_avg_meter.avg)
                print("Critic %2.5f" % critic_avg_meter.avg)
                print("Sample %2.5f" % sample_avg_meter.avg)
                print("----------------------------------------")
                print("Feature")
                print("----------------------------------------")
                print("Dribbler %2.5f"     % dribbler_avg_meter.avg)
                print("Blocked %2.5f"      % blocked_avg_meter.avg)
                print("Ball pass %2.5f"    % ball_pass_avg_meter.avg)
                print("Velocity %2.5f"     % vel_avg_meter.avg)
                print("Acceleration %2.5f" % acc_avg_meter.avg)     
                print("----------------------------------------")
                print("Condition Reconstruct")
                print("----------------------------------------")
                print("Sketch Recon %2.5f" % sket_recon_avg_meter.avg)
                print("Status Recon %2.5f" % stat_recon_avg_meter.avg)
                print("----------------------------------------")
                print("Prior")
                print("----------------------------------------")
                print("Prior %2.5f" % prior_avg_meter.avg)
                
        # validate
        validate(test_loader, model, epoch, norm_dict, writer, save_dir, args)

        # save checkpoints
        if (epoch + 1) % args.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, 'Basketball-Flow-%d.pt' % epoch))



if __name__ == '__main__':
    args = get_args()
    save_dir = os.path.join(args.log_name, "checkpoints")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with open(os.path.join(save_dir, 'command.sh'), 'w') as f:
        f.write('python -X faulthandler ' + ' '.join(sys.argv))
        f.write('\n')
        
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
    main_worker(args.gpu, save_dir, ngpus_per_node, args)
