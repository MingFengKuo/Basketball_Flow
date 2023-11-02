import os
import torch
import random
import numpy as np
import torch.nn.functional as F
from dataset import denormalize_pos

class AverageValueMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_random_seed(seed):
    """set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
            
def validate(loader, model, epoch, norm_dict, writer, save_dir, args):
    # Make epoch wise save directory
    if args.save_val_results and (epoch+1) % args.val_freq == 0:
        save_dir = os.path.join(save_dir, 'epoch-%d' % epoch)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = None
    
    # Validate
    model.eval()
    with torch.no_grad():
        all_fake_data = []
        all_sample_data = []
        iterator = iter(loader)
        
        # Generate loss
        recon_avg_meter = AverageValueMeter()
        kl_div_avg_meter = AverageValueMeter()
        em_dist_avg_meter = AverageValueMeter()
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
        
        for data in iterator:
            idx_batch, ref_data, ref_cond, mask = data['idx'], data['data'], data['cond'], data['mask']
            ref_data = ref_data.cuda() if args.gpu is None else ref_data.cuda(args.gpu)
            ref_cond = ref_cond.cuda() if args.gpu is None else ref_cond.cuda(args.gpu)
            mask     = mask.cuda() if args.gpu is None else mask.cuda(args.gpu)
            # Test
            out = model.validate(ref_data.size(0), ref_data, ref_cond, mask, norm_dict)
            # Generate loss
            recon_avg_meter.update(out['recon'])
            kl_div_avg_meter.update(out['kl-div'])
            em_dist_avg_meter.update(out['em-dist'])
            # Feature loss
            dribbler_avg_meter.update(out['dribbler'])
            blocked_avg_meter.update(out['blocked'])
            ball_pass_avg_meter.update(out['ball_pass'])
            vel_avg_meter.update(out['velocity'])
            acc_avg_meter.update(out['acceleration'])
            # Sketch Reconstruct loss
            sket_recon_avg_meter.update(out['sketch_recon'])
            stat_recon_avg_meter.update(out['status_recon'])
            # Prior loss
            prior_avg_meter.update(out['prior'])
            
            # Denormalize
            fake_data = denormalize_pos(out['fake_x'], norm_dict, dtype='smp') * mask
            all_fake_data.append(fake_data)
            sample_data = denormalize_pos(out['sample_x'], norm_dict, dtype='smp') * mask
            all_sample_data.append(sample_data)
            
        print("****************************************")
        print("Test Epoch %d " % epoch)
        print("****************************************")
        print("Sample")
        print("----------------------------------------")
        print("Recon   %2.5f" % recon_avg_meter.avg)
        print("KL-div  %2.5f" % kl_div_avg_meter.avg)
        print("EM-Dist %2.5f" % em_dist_avg_meter.avg)
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
        
        if writer is not None:
            writer.add_scalar('test-gan/recon',              recon_avg_meter.avg, epoch)
            writer.add_scalar('test-gan/kl-div',             kl_div_avg_meter.avg, epoch)
            writer.add_scalar('test-gan/em-dist',            em_dist_avg_meter.avg, epoch)
            writer.add_scalar('test-feature/dribbler',       dribbler_avg_meter.avg, epoch)
            writer.add_scalar('test-feature/blocked',        blocked_avg_meter.avg, epoch)
            writer.add_scalar('test-feature/ball_pass',      ball_pass_avg_meter.avg, epoch)
            writer.add_scalar('test-feature/velocity',       vel_avg_meter.avg, epoch)
            writer.add_scalar('test-feature/acceleration',   acc_avg_meter.avg, epoch)
            writer.add_scalar('test-cond-recon/sketch',      sket_recon_avg_meter.avg, epoch)
            writer.add_scalar('test-cond-recon/status',      stat_recon_avg_meter.avg, epoch)
            writer.add_scalar('test-flow/prior',             prior_avg_meter.avg, epoch)
            
        fake_data = torch.cat(all_fake_data, dim=0)
        sample_data = torch.cat(all_sample_data, dim=0)
        
        if save_dir is not None and args.save_val_results and (epoch+1) % args.val_freq == 0:
            # fake data
            fake_data_save_name = os.path.join(save_dir, "fake_data.npy")
            np.save(fake_data_save_name, fake_data.cpu().detach().numpy())
            # sample data
            sample_data_save_name = os.path.join(save_dir, "sample_data.npy")
            np.save(sample_data_save_name, sample_data.cpu().detach().numpy())
            print("----------------------------------------")
            print("Saving file: %s" % save_dir)