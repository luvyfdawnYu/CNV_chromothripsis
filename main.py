import torch
from run import run
import argparse
import os
import pandas as pd
from utils import split_data
parser = argparse.ArgumentParser()
parser.add_argument('--split', required=True)
parser.add_argument('--test_seed',type=int,required=True)
parser.add_argument('--batch_size',type=int,required=True)
parser.add_argument('--data_dir', default='./dataset')
parser.add_argument('--GE_dim', default=[16,32],nargs="+", type=int)
parser.add_argument('--LF_stride', default=32, type=int)
parser.add_argument('--LF_input_shape', default=[1024,24*32,16*32,8*32], nargs="+",type=int)
parser.add_argument('--LF_output_shape',default = [24,16,8,4], nargs="+",type=int)
parser.add_argument('--NL_input_shape',default = 32*4, type=int)
parser.add_argument('--NL_output_shape',default = 32*4, type=int)
parser.add_argument('--learning_rate', default=1e-2, type=float)
parser.add_argument('--num_epochs', default= 120, type=int)
parser.add_argument('--gamma', default=1.0, type=float)
parser.add_argument('--epsilon', default=0.02, type=float)
parser.add_argument('--osbfb_seed',default=3600,type=int)

opt = parser.parse_args()
if opt.split not in ['1','2','3','4','5']:
    raise AssertionError('split should be in range of 1 to 5')

class Config():
    def __init__(self):
        self.device = torch.device('cpu')
        self.learning_rate = opt.learning_rate
        self.num_epochs = opt.num_epochs
        self.gamma = opt.gamma
        self.num_classes = 2
        self.e = opt.epsilon
        self.weight_decay = 5e-5
        self.osbfb_seed = opt.osbfb_seed
        self.test_n_folds = 10
        self.test_seed = opt.test_seed
        self.train_ros = True
        self.batch_fix = True
        self.batch_fix_v2 = True
        self.batch_size = opt.batch_size
        self.batch_fix_balance = True
        self.use_class_weight = True
        self.data_dir = opt.data_dir
        self.split = opt.split

        self.GE_dim = opt.GE_dim
        self.LF_stride = opt.LF_stride
        self.LF_input_shape = opt.LF_input_shape
        self.LF_output_shape = opt.LF_output_shape
        self.NL_input_shape = opt.NL_input_shape
        self.NL_output_shape = opt.NL_output_shape
        self.dropout_rate = 0.5

if __name__ == '__main__':       
    config = Config()
    if not os.path.isdir(config.data_dir):
        os.makedirs(config.data_dir)
    if not os.path.isfile(config.data_dir + '/CV%s'%opt.split):
        split_data('./CNV.csv',config)
    run(config)