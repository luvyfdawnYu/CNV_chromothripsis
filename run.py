import pandas as pd
import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as GraphDataLoader
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from utils import get_time_dif, fix_batch, to_categorical, set_random_seed, prepare_data,make_cv_splits,Logger
from sklearn.metrics import classification_report,confusion_matrix,\
     precision_score, f1_score, accuracy_score, recall_score,\
     roc_auc_score, auc, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from train import train_func, evaluate
from GECNVNet import GECNVNet
import shutil
from create_graph import TrainDataset,ValDataset,TestDataset

def run(config):
    num_classes = config.num_classes
    # log_dir = config.log_dir
    osbfb_seed = config.osbfb_seed
    test_n_folds = config.test_n_folds
    test_seed = config.test_seed
    train_ros = config.train_ros
    batch_fix = config.batch_fix
    batch_fix_v2 = config.batch_fix_v2
    batch_size = config.batch_size
    batch_fix_balance = config.batch_fix_balance
    use_class_weight = config.use_class_weight
    ###############################################
    df = pd.read_csv(config.data_dir +'/CV%s.csv'%config.split,index_col=0).fillna(0)
    df_org = df.iloc[:,-28:]
    df_pred = df['chromothripsis']
    data_org = np.array(df_org.values,dtype=np.float32)

    df_bl = pd.read_csv(config.data_dir +'/BL%s.csv'%config.split,index_col=0)
    df_bl_org = df_bl.iloc[:,-28:]
    df_bl_pred = df_bl['chromothripsis']
    data_bl_org = np.array(df_bl_org.values,dtype=np.float32)
    ###############################################
    tag = 'GECNVNet_Split%s'%config.split
    save_dir="output"
    save_dir_abs = os.path.join(os.path.dirname(os.path.abspath(__file__)),save_dir)
    log_dir = os.path.join(save_dir_abs, tag)
    if not os.path.isdir(save_dir_abs):
        os.makedirs(save_dir_abs)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
        os.makedirs(os.path.join(log_dir,'model_weights'))
    sys.stdout = Logger(log_dir+"/stdout.txt")

    ss_x_mm = MinMaxScaler((0,1))
    data_org = ss_x_mm.fit_transform(data_org)
    data_bl_org = ss_x_mm.transform(data_bl_org)

    ss_x = StandardScaler(with_mean=True,with_std=True)
    data_org = ss_x.fit_transform(data_org)
    data_bl_org = ss_x.transform(data_bl_org)

    df_org = pd.DataFrame(data=data_org,columns=df_org.columns)
    df_bl_org = pd.DataFrame(data=data_bl_org, columns = df_bl_org.columns)

    data_all = prepare_data(df_org)
    data_bl_all = prepare_data(df_bl_org)

    label_all = np.array(df_pred.values,dtype=np.float32)
    label_bl_all = np.array(df_bl_pred.values,dtype=np.float32)

    print("CV DATA:")
    print("data_all:",data_all.shape,label_all.shape,data_all.dtype)
    print("min-max:",data_all.min(),data_all.max())
    print(confusion_matrix(label_all, label_all))
    print("")
    print("BL DATA:")
    print("data_all:",data_bl_all.shape,label_bl_all.shape,data_bl_all.dtype)
    print("min-max:",data_bl_all.min(),data_bl_all.max())
    print(confusion_matrix(label_bl_all, label_bl_all))
    print("")
    
    class_weight_array=compute_class_weight('balanced',
                                        classes=np.arange(num_classes),y=label_all.ravel())

    if not use_class_weight:
        class_weight=None 
    else:
        class_weight=dict()
        for c in np.arange(num_classes):
            class_weight[c]=class_weight_array[c]


    class_weight_array_neg=np.ones_like(class_weight_array)
    class_weight_nmlist=np.array([class_weight_array[i]/
                                (class_weight_array[i]+class_weight_array_neg[i])
                                for i in range(class_weight_array.shape[0])],
                                dtype=np.float32)
    class_weight_nmlist_def=np.array([1.0
                                    for i in range(class_weight_array.shape[0])],
                                    dtype=np.float32)
    print("using class_weight:",use_class_weight)
    print("class_weight_array:",class_weight_array.tolist())
    print("class_weight_nmlist:",class_weight_nmlist.tolist())
    print("class_weight_nmlist_def:",class_weight_nmlist_def.tolist())
    print("\n")
    
    test_folds_list=make_cv_splits(data_all,label_all,test_n_folds,test_seed)

    df_hist = pd.DataFrame()
    hist_acc = []
    hist_prec = []
    hist_recall = []
    hist_f1 = []
    hist_rocp = []
    hist_prcp = []
    y_allcv = np.array([],dtype=int)
    rocp_allcv = np.array([],dtype=float)
    prcp_allcv = np.array([],dtype=float)

    # start=time.time()
    for n_cv, (train_index, val_index) in enumerate(test_folds_list):

        t=time.time()
        print("\n==================Loading Data==================\n")

        print("\n=====================CV[%s]========================\n"%n_cv)

        x_train_org = data_all[train_index].copy()
        
        y_train_org= label_all[train_index].copy()

        x_val_org = data_all[val_index].copy()
        y_val_org = label_all[val_index].copy()


        print('train samples:',x_train_org.shape)
        print(confusion_matrix(y_train_org, y_train_org))
        print("min-max:",x_train_org.min(),x_train_org.max())


        x_train_org,y_train_org=fix_batch(x_train_org,y_train_org,
                                        osbfb_seed,batch_size,
                                        train_ros,batch_fix,batch_fix_v2,batch_fix_balance)
        print('new train samples:',x_train_org.shape)
        print(confusion_matrix(y_train_org, y_train_org))
        


        y_train_cat = to_categorical(y_train_org, num_classes)
        # if num_classes == 2:
        #     watch_cls = 1
        #     y_train_watch=y_train_cat[:,watch_cls]

        y_val_cat = to_categorical(y_val_org, num_classes)
        if num_classes == 2:
            watch_cls = 1
            y_val_watch=y_val_cat[:,watch_cls]

        sample_weight = np.ones(y_train_org.shape[0], dtype=np.float32)
        if use_class_weight:
            for i, val in enumerate(y_train_org):
                sample_weight[i] = class_weight[val]
        print("")
        print("")

        x_test_org=data_bl_all.copy()
        y_test_org=label_bl_all.copy()
        print('test samples:',x_test_org.shape)
        print(confusion_matrix(y_test_org, y_test_org))

        #
        print("min-max:",x_test_org.min(),x_test_org.max())


        print("")
        y_test_cat = to_categorical(y_test_org, num_classes)
        if num_classes == 2:
            y_test_watch=y_test_cat[:,watch_cls]

        x_train_dl = x_train_org
        y_train_dl = y_train_cat

        x_val_dl = x_val_org
        y_val_dl = y_val_cat

        x_test_org=data_bl_all.copy()
        y_test_org=label_bl_all.copy()
        print('test samples:',x_test_org.shape)
        print(confusion_matrix(y_test_org, y_test_org))

            #
        print("min-max:",x_test_org.min(),x_test_org.max())


        print("")
        y_test_cat = to_categorical(y_test_org, num_classes)

        if num_classes == 2:
            y_test_watch=y_test_cat[:,watch_cls]
        x_test_dl = x_test_org
        y_test_dl = y_test_cat


        # batch_iters=x_train_dl.shape[0]//batch_size

        train_dataset = TrainDataset(root=log_dir + '/graph_data/train',
        feat=x_train_dl,label=y_train_dl)
        val_dataset = ValDataset(root=log_dir + '/graph_data/val',
        feat=x_val_dl,label=y_val_dl)
        test_dataset = TestDataset(root= log_dir + '/graph_data/test',
        feat=x_test_dl,label=y_test_dl)
        

        train_iter = GraphDataLoader(train_dataset, batch_size = batch_size)
        
            #print(_.num_graphs)
        val_iter = GraphDataLoader(val_dataset, batch_size = batch_size)
        
        test_iter = GraphDataLoader(test_dataset,batch_size = batch_size)
        time_data_usage = get_time_dif(t)
        print('Tims Usage:', time_data_usage)

        set_random_seed(test_seed)
        model = GECNVNet(
            config.GE_dim,
            config.LF_stride,
            config.LF_input_shape,
            config.LF_output_shape,
            config.NL_input_shape,
            config.NL_output_shape,
            config.dropout_rate
        )
        savepath=log_dir + '/model_weights/model'+str(n_cv)+'.pkl'
        
        train_func(config, model, train_iter, val_iter, y_val_watch, test_iter , y_test_watch, savepath)
        model.load_state_dict(torch.load(savepath))
        cv_acc, cv_loss,cv_rocp, cv_f1, cv_confusion_m,cv_precision,cv_recall,\
            cv_label_out_pred_prob,cv_prcp=evaluate(config,model, val_iter,y_val_watch, test=True)
        hist_acc.append(cv_acc)
        hist_prec.append(cv_precision)
        hist_recall.append(cv_recall)
        hist_rocp.append(cv_rocp)
        hist_f1.append(cv_f1)
        hist_prcp.append(cv_prcp)

        y_allcv = np.concatenate([y_allcv,y_val_watch])
        rocp_allcv = np.concatenate([rocp_allcv,cv_label_out_pred_prob])
        prcp_allcv = np.concatenate([prcp_allcv,cv_label_out_pred_prob])

        precision_oc, recall_oc, _ = precision_recall_curve(y_val_watch, cv_label_out_pred_prob,pos_label=1)
        
        plt.figure(6661,figsize=(10,8))
        plt.plot(recall_oc, precision_oc, ':',
                label='CV%s(AUC:%0.4f|PRC:%0.4f|F1:%0.4f|PR:%0.4f|RC:%0.4f|ACC:%0.4f)'% (str(n_cv),hist_rocp[-1],hist_prcp[-1],hist_f1[-1],
                    hist_prec[-1],hist_recall[-1],hist_acc[-1]))
        plt.xlabel('False positive rate(1-Specificity)')
        plt.ylabel('True positive rate(Sensitivity)')
        plt.title('\n\nROC curve(CVs)')
        plt.legend(loc='best')
        
        shutil.rmtree(log_dir + '/graph_data/train/')
        shutil.rmtree(log_dir + '/graph_data/val/')
        shutil.rmtree(log_dir + '/graph_data/test/')


    ##############CVs ROC##########################
    df_hist["acc"] = np.array(hist_acc,dtype=float)
    df_hist["prec"] = np.array(hist_prec,dtype=float)
    df_hist["recall"] = np.array(hist_recall,dtype=float)
    df_hist["f1"] = np.array(hist_f1,dtype=float)
    df_hist["rocp"] = np.array(hist_rocp,dtype=float)
    df_hist["prcp"] = np.array(hist_prcp,dtype=float)
    print(df_hist)
    print(df_hist.describe())

    pr_all,rc_all,_ = precision_recall_curve(y_allcv,prcp_allcv)

    no_skill = y_allcv[y_allcv==1].shape[0] / y_allcv.shape[0]
    plt.figure(6661,figsize=(10,8))
    plt.plot([0, 1], [no_skill,no_skill], 'k--')
    plt.plot(rc_all, pr_all, 'r-',
            label='AVG(AUC:%0.4f|PRC:%0.4f|F1:%0.4f|PR:%0.4f|RC:%0.4f|ACC:%0.4f)'
            % (
    
                df_hist.rocp.mean(),
                df_hist.prcp.mean(),
                df_hist.f1.mean(),
                df_hist.prec.mean(),
                df_hist.recall.mean(),
                df_hist.acc.mean()
                ))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('\n\nPRC curve(CVs)')
    plt.legend(loc='best')
    plt.savefig(log_dir + "/model_weights/prc-cvs%s.svg"%("GECNVNet"))
    plt.show()
    df_cvs=pd.DataFrame()
    df_cvs["pred_prob"]=rocp_allcv
    df_cvs["real_label"]=y_allcv
    df_cvs.to_csv(log_dir + "/model_weights/pred-cvs%s.csv"%("GECNVNet"),index=False)


    #############Blind ROC
    test_dataset = TestDataset(root=log_dir+'/graph_data/test',
        feat=x_test_dl,label=y_test_dl)
    test_iter = GraphDataLoader(test_dataset, batch_size = batch_size)
    print("\n====================================")
    print("")
    print("BLIND TEST")
    prob_list=[]
    for i in range(10):
        outputs_list=[]
        savepath=log_dir+'/model_weights/model'+str(i)+'.pkl'
        model = GECNVNet(
            config.GE_dim,
            config.LF_stride,
            config.LF_input_shape,
            config.LF_output_shape,
            config.NL_input_shape,
            config.NL_output_shape,
            config.dropout_rate
        ).to(config.device)
        model.load_state_dict(torch.load(savepath))
        model.eval()
        with torch.no_grad():
            for data in test_iter:
                labels = data.y
                outputs=model(data)
                outputs_prob=F.softmax(outputs,dim=-1)
                outputs_list.append(outputs_prob)
        outputs_prob=torch.cat(outputs_list,0)
        prob_list.append(outputs_prob.detach().cpu().numpy())
    x=np.zeros((y_test_org.shape[0],2)) #165 108
    for i in prob_list:
        x+=i
    x_1=x/10
    label_out_pred_d = np.argmax(x_1,axis=1)
    label_out_pred_m = label_out_pred_d
    label_out_pred=np.asarray(label_out_pred_m==1,dtype=np.int32)
    label_out_pred_prob = x_1[:,1]

    print(classification_report(y_test_org, label_out_pred_m))
    print(confusion_matrix(y_test_org, label_out_pred_m))
    acc = accuracy_score(y_test_org, label_out_pred_m)
    f1 = f1_score(y_test_watch, label_out_pred)
    rocp_avg = roc_auc_score(y_test_watch, label_out_pred_prob)
    print('acc_avg=',acc)
    print('f1_avg=',f1)
    print('rocp_avg=',rocp_avg)
    precision_oc,recall_oc,_ = precision_recall_curve(y_test_watch,label_out_pred_prob)
    prcp = auc(recall_oc, precision_oc)

    no_skill = y_test_watch[y_test_watch==1].shape[0]/y_test_watch.shape[0]
    plt.figure(8881,figsize=(10,8))
    plt.plot([0, 1], [no_skill, no_skill], 'k--')
    plt.plot(recall_oc, precision_oc, 'r-',
            label='BlindTest(AUC:%0.4f|PRC:%0.4f|F1:%0.4f|PR:%0.4f|RC:%0.4f|ACC:%0.4f)'
            % (roc_auc_score(y_test_watch, label_out_pred_prob),
                prcp,
            f1_score(y_test_watch, label_out_pred),
            precision_score(y_test_watch, label_out_pred),
            recall_score(y_test_watch, label_out_pred),
            accuracy_score(y_test_org, label_out_pred_m)))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('\n\nPRC curve(Blind)')
    plt.legend(loc='best')
    plt.savefig(log_dir+"/model_weights/prcp-blind.svg")
    plt.show()

    df_blind=pd.DataFrame()
    df_blind["pred_prob"]=label_out_pred_prob
    df_blind["real_label"]=y_test_watch
    #df_blind.plot()
    df_blind_file=log_dir+"/model_weights/pred-blind.csv"
    df_blind.to_csv(df_blind_file,index=False)


    print("\n====================================")

    shutil.rmtree(log_dir + '/graph_data/test/')