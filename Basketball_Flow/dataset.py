import os
import torch
import numpy as np

from torch.utils.data import Dataset

class BasketballDataset(Dataset):
    def __init__(self, real_data, real_cond, seq_data, seq_cond, mask_data):
        self.mask_data = mask_data
        self.real_data = real_data
        self.real_cond = real_cond
        self.seq_data  = seq_data
        self.seq_cond  = seq_cond
        
        self.BASKET_LEFT = [4, 25]
        self.BASKET_RIGHT = [88, 25]
        
        # make training data ready
        self.data, self.cond = self.get_ready()
    
    def get_ready(self):
         # data
         data = np.concatenate([self.real_data, self.real_cond], axis=-1)
         # cond
         cond = np.concatenate([self.seq_data, self.seq_cond], axis=-1)
         return data, cond
        
    def __len__(self):
        return len(self.real_data)
    
    def __getitem__(self, idx):
        data   = torch.from_numpy(self.data[idx]).float()
        cond   = torch.from_numpy(self.cond[idx]).float()
        mask   = torch.from_numpy(self.mask_data[idx]).float()
        
        return {
            'idx': idx,
            'data': data,
            'cond': cond,
            'mask': mask,
        }
    
def get_dataset(args):
    # read data
    real_data = np.load(os.path.join(args.data_path, 'Real.npy'))
    real_cond = np.load(os.path.join(args.data_path, 'RealCond.npy'))
    seq_data  = np.load(os.path.join(args.data_path, 'Seq.npy'))
    seq_cond  = np.load(os.path.join(args.data_path, 'SeqCond.npy'))
    print("Real Data: " + str(real_data.shape))
    print("Real Cond: " + str(real_cond.shape))
    print("Seq  Data: " + str(seq_data.shape))
    print("Seq  Cond: " + str(seq_cond.shape))
    
    # normalize
    mask_data = np.load(os.path.join(args.data_path, 'Mask.npy'))
    real_data = np.where(mask_data!=0, real_data, np.nan)
    real_data, norm_dict = normalize_pos(real_data)
    real_data = np.nan_to_num(real_data)
    
    # split data into training data and testing data
    tr_rd, te_rd = np.split(real_data, [real_data.shape[0] // 10 * 9])
    tr_rc, te_rc = np.split(real_cond, [real_cond.shape[0] // 10 * 9])
    tr_sd, te_sd = np.split(seq_data,  [seq_data.shape[0] // 10 * 9])
    tr_sc, te_sc = np.split(seq_cond,  [seq_cond.shape[0] // 10 * 9])
    tr_mk, te_mk = np.split(mask_data, [mask_data.shape[0] // 10 * 9])
    
    train_data = BasketballDataset(tr_rd, tr_rc, tr_sd, tr_sc, tr_mk)
    test_data  = BasketballDataset(te_rd, te_rc, te_sd, te_sc, te_mk)
    
    return train_data, test_data, norm_dict

def normalize_pos(real_data):
    """ directly normalize player x, y on real_data """
    norm_dict = {}
    axis_list = ['x', 'y']
    BASKET_RIGHT = [88, 25]
    
    # X position
    mean_x = np.nanmean(real_data[:,:,[0,2,4,6,8,10,12,14,16,18,20]])
    stddev_x = np.nanstd(real_data[:,:,[0,2,4,6,8,10,12,14,16,18,20]])
    real_data[:,:,[0,2,4,6,8,10,12,14,16,18,20]] = \
    (real_data[:,:,[0,2,4,6,8,10,12,14,16,18,20]] - mean_x) / stddev_x
    
    norm_dict['x'] = {}
    norm_dict['x']['mean'] = mean_x
    norm_dict['x']['stddev'] = stddev_x
    
    # Y position
    mean_y = np.nanmean(real_data[:,:,[1,3,5,7,9,11,13,15,17,19,21]])
    stddev_y = np.nanstd(real_data[:,:,[1,3,5,7,9,11,13,15,17,19,21]])
    real_data[:,:,[1,3,5,7,9,11,13,15,17,19,21]] = \
    (real_data[:,:,[1,3,5,7,9,11,13,15,17,19,21]] - mean_y) / stddev_y
    
    norm_dict['y'] = {}
    norm_dict['y']['mean'] = mean_y
    norm_dict['y']['stddev'] = stddev_y
    
    # Basket position
    norm_dict['x']['basket'] = (BASKET_RIGHT[0] - mean_x) / stddev_x
    norm_dict['y']['basket'] = (BASKET_RIGHT[1] - mean_y) / stddev_y
    
    return real_data, norm_dict
    
def denormalize_pos(data, norm_dict, dtype='ref'):
    """denormalize player x,y on data """
    if dtype=='ref' or dtype=='rec' or dtype=='smp':
        # X position
        data[:,:,[0,2,4,6,8,10,12,14,16,18,20]] = \
            data[:,:,[0,2,4,6,8,10,12,14,16,18,20]] * norm_dict['x']['stddev'] + norm_dict['x']['mean']
        # Y position
        data[:,:,[1,3,5,7,9,11,13,15,17,19,21]] = \
            data[:,:,[1,3,5,7,9,11,13,15,17,19,21]] * norm_dict['y']['stddev'] + norm_dict['y']['mean']
                
    elif dtype=='cond':
        # X position
        data[:,:,[0,2,4,6,8,10]] = \
            data[:,:,[0,2,4,6,8,10]] * norm_dict['x']['stddev'] + norm_dict['x']['mean']
        # Y position
        data[:,:,[1,3,5,7,9,11]] = \
            data[:,:,[1,3,5,7,9,11]] * norm_dict['y']['stddev'] + norm_dict['y']['mean']
    
    else:
        print("[Warning!] dtype needs to be ref, rec, smp or cond!")
            
    return data
