from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import sys
from datetime import timedelta
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import pandas as pd

def make_cv_splits(data_cv,label_cv,n_cv_splits,cv_seed):
    splits_cv = []
    kf_cv = StratifiedKFold(n_splits=n_cv_splits,
                            shuffle=True,
                            random_state=cv_seed)
    for tridx_cv,tsidx_cv in kf_cv.split(data_cv,label_cv):
        splits_cv.append([tridx_cv,tsidx_cv])
    return splits_cv

def make_train_val_split(data_train_val,label_train_val):
    splits_train_val = []
    sss_train_val = StratifiedShuffleSplit(n_splits=5,test_size=0.2)
    for tridx_train_val, vaidx_train_val in sss_train_val.split(data_train_val,label_train_val):
        splits_train_val.append([tridx_train_val,vaidx_train_val])
    return splits_train_val

def split_data(org_path,config):
    df_all = pd.read_csv(org_path)
    data_cv_bl = np.array(df_all.iloc[:,-28:].values,dtype=np.float32)
    label_cv_bl = np.array(df_all['chromothripsis'].values,dtype=np.float32)

    cv_bl_list = make_train_val_split(data_cv_bl,label_cv_bl)

    for n_split,(cv_idx,bl_idx) in enumerate(cv_bl_list):
        df_cv = df_all.loc[cv_idx].reset_index(drop=True)
        df_bl = df_all.loc[bl_idx].reset_index(drop=True)
        df_cv.to_csv(config.data_dir + '/CV%s.csv'%str(int(n_split)+1),index=False)
        df_bl.to_csv(config.data_dir + '/BL%s.csv'%str(int(n_split)+1),index=False)


def to_categorical(labels_flat,num_classes=None):
    hotenc=OneHotEncoder(categories=[[c for c in range(num_classes)]])
    labels_hot=hotenc.fit_transform(np.reshape(labels_flat,(-1,1))).toarray().astype(np.float32)
    if num_classes:
        assert(labels_hot.shape[1]==num_classes)
    return labels_hot


def fix_batch(x_train,y_train_flat,
              seed,batch_size,
              train_ros=True,
              batch_fix=True,batch_fix_v2=True,batch_fix_balance=True,
              ):
    if train_ros:
        x_train_flat = x_train
        ros=RandomOverSampler(random_state=seed)
        x_train_ros,y_train_ros=ros.fit_resample(x_train_flat,y_train_flat)
        print('Over sampling:',x_train.shape[0],"->",x_train_ros.shape[0])
        x_train_new = x_train_ros
        y_train_new = y_train_ros
    else:
        x_train_new = x_train
        y_train_new = y_train_flat

    if batch_fix:
        new_size=x_train_new.shape[0]
        n_add=0
        if new_size%batch_size!=0:
            n_add+=batch_size-new_size%batch_size
            new_size+=n_add
        if (new_size//batch_size)%2!=0:
            n_add+=batch_size
            new_size+=n_add
        if n_add>0:
            rnd=np.random.RandomState(seed)
            if batch_fix_v2:
                x_train_lb=x_train
                y_train_lb=y_train_flat
            else:
                x_train_lb=x_train_new
                y_train_lb=y_train_new
            if not batch_fix_balance:
                idxs_add=rnd.choice(x_train_lb.shape[0],n_add,replace=False)
                x_tr_add=x_train_lb[idxs_add]
                y_tr_add=y_train_lb[idxs_add]
                x_train_new = np.concatenate([x_train_new,x_tr_add])
                y_train_new = np.concatenate([y_train_new,y_tr_add])
            else:
                # assert(num_classes==2)
                for i in range(2):
                    lb_add=i
                    if lb_add==0:
                        n_add_lbc=n_add//2
                    else:
                        n_add_lbc=n_add-n_add_lbc
                    ids_lbc=np.where(y_train_lb==lb_add)[0]
                    x_train_lbc=x_train_lb[ids_lbc]
                    y_train_lbc=y_train_lb[ids_lbc]
                    idxs_add=rnd.choice(x_train_lbc.shape[0],n_add_lbc,replace=False)
                    x_tr_add=x_train_lbc[idxs_add]
                    y_tr_add=y_train_lbc[idxs_add]
                    x_train_new = np.concatenate([x_train_new,x_tr_add])
                    y_train_new = np.concatenate([y_train_new,y_tr_add])
            print('Add train samples:',n_add)
    return x_train_new, y_train_new
def prepare_data(df_org):
    df_mb = df_org.iloc[:,:3]
    df_count = df_org.iloc[:,3:8]
    df_jump = df_org.iloc[:,8:11]
    df_band = df_org.iloc[:,11:14]
    df_osci = df_org.iloc[:,14:18]
    df_size = df_org.iloc[:,18:]

    ######## padding 0
    data_mb = np.concatenate((df_mb.values,np.zeros((df_mb.shape[0],7))),axis=1)
    data_count = np.concatenate((df_count.values,np.zeros((df_mb.shape[0],5))),axis=1)
    data_jump = np.concatenate((df_jump.values,np.zeros((df_mb.shape[0],7))),axis=1)
    data_band = np.concatenate((df_band.values,np.zeros((df_mb.shape[0],7))),axis=1)
    data_osci = np.concatenate((df_osci.values,np.zeros((df_mb.shape[0],6))),axis=1)
    data_size = np.array(df_size.values)
    data_v_end = np.zeros_like(data_size)

    data_all = np.concatenate((data_mb,data_count,data_jump,data_band,data_osci,data_size,data_v_end),axis=1)

    return data_all

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return int(round(time_dif))

import random
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class Logger(object):
    def __init__(self, logfile=None):
        self.terminal = sys.stdout
        self.logfile = logfile
    def write(self, message):
        self.terminal.write(message)
        if self.logfile:
            with open(self.logfile,"a",encoding='utf-8') as f: #a/追加模式
                f.write(message)
    def flush(self):
        self.terminal.flush()

class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

class downLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (self.total_iters-self.last_epoch)/ (self.total_iters + 1e-8) for base_lr in self.base_lrs]

class focal_loss_mixed(nn.Module):
    def __init__(self,config):
        super(focal_loss_mixed,self).__init__()
        self.num_classes=config.num_classes
        self.gamma=config.gamma
        self.device=config.device
        self.eps=1e-7

    def forward(self,y_pred,y_true):
        y_pred_prob=F.softmax(y_pred,dim=-1)
        y_pred_prob=torch.clamp(y_pred_prob,min=self.eps,max=1-self.eps)
        pos=-1.0*torch.pow(1.0-y_pred_prob,self.gamma)*torch.log(y_pred_prob)*y_true
        neg=-1.0*torch.pow(y_pred_prob,self.gamma)*torch.log(1.0-y_pred_prob)*(1.0-y_true)
        loss=pos+neg
        loss=torch.sum(loss,axis=-1)
        loss=torch.mean(loss,axis=-1)
        return loss

def custom_loss(y_true,y_pred,config):
    loss1=focal_loss_mixed(config)(y_pred,y_true)
    loss2=focal_loss_mixed(config)(y_pred,torch.ones_like(y_pred)/config.num_classes)
    e=config.e
    loss=(1-e)*loss1+e*loss2
    return loss
