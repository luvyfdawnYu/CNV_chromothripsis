import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
from utils import WarmUpLR, downLR, custom_loss, get_time_dif
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix,\
     precision_score, f1_score, accuracy_score, recall_score,\
     roc_auc_score, auc, precision_recall_curve



def train_func(config,model,train_iter, val_iter,val_watch,test_iter,test_watch, savepath):
    start_time=time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate,
    weight_decay=config.weight_decay,
    )
    warmup_epoch = config.num_epochs/2
    iter_per_epoch = 1
    warmup_scheduler = WarmUpLR(optimizer, warmup_epoch*iter_per_epoch)
    scheduler = downLR(optimizer,(config.num_epochs-warmup_epoch)*iter_per_epoch)
    total_batch = 0  
    iter_per_epoch = 1
    cv_best_rfa = 0.0
    cv_best_acc = 0.0
    cv_best_rocp = 0.0
    for epoch in tqdm(range(config.num_epochs)):
        loss_total = 0
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        if (epoch>=warmup_epoch):
            learn_rate = scheduler.get_last_lr()

        else:
            learn_rate = warmup_scheduler.get_last_lr()

        print("Learn_rate:%s" % learn_rate)
        for i, (data) in enumerate(train_iter):
            
            trains = data
            labels = data.y
            trains = trains.to(config.device)
            labels = labels.to(config.device)
                
            outputs = model(trains)
            model.zero_grad()
            loss = custom_loss(labels, outputs, config)
        
            loss.backward()
            optimizer.step()
            total_batch += 1
            loss_total+=loss

        if (epoch<warmup_epoch):
            warmup_scheduler.step()
        else:
            scheduler.step()
        
        loss_avg_batch=loss_total/len(train_iter)
        cv_acc, cv_loss, cv_rocp, cv_f1 ,cv_prcp = evaluate(config, model, val_iter, val_watch)
        cv_rfa =  1/3. * cv_rocp + 1/3. * cv_f1 + 1/3. * cv_prcp
        if cv_rfa > cv_best_rfa:
            cv_best_rfa = cv_rfa
            cv_best_acc = cv_acc
            cv_best_f1 = cv_f1
            cv_best_epoch = epoch + 1
            cv_best_rocp = cv_rocp
            cv_best_prcp = cv_prcp
            torch.save(model.state_dict(), savepath)
            improve = '*'
        else:
            improve = ' '

        time_dif = get_time_dif(start_time)
        msg = 'Epoch: {0:>6},  Train Loss: {1:>6.4},'+'\n'+'Eval Val Loss: {2:>5.4}, Eval Val Acc: {3:>6.4%}, Eval Val ROCP: {4:>5.4}, Eval Val F1: {5:>5.4}, Eval Val PRCP: {6:>5.4}'+'\n'+'Time: {7} {8}'
        print(msg.format(epoch+1, loss_avg_batch.item(),  cv_loss, cv_acc, cv_rocp, cv_f1, cv_prcp, time_dif, improve))
        print("CV BEST SOFAR ===> epoch: %d - rocp: %5f - f1: %5f - acc: %5f - prcp: %5f"%(
                    cv_best_epoch,cv_best_rocp,cv_best_f1,cv_best_acc,cv_best_prcp))
        print('\n\n')
        model.train()
    
    print('LOAD BEST ON CV TO PREDICT')
    test(config, model,test_iter,savepath,test_watch)

def evaluate(config,model, data_iter, watch, test=False):
    model.eval()
    loss_total = 0
    predict_all = []
    labels_all = []
    with torch.no_grad():
        for data in data_iter:
            labels = data.y
            data=data.to(config.device)
            labels=labels.to(config.device)
            outputs = model(data)
            labels_index_t=torch.max(labels.data,1)[1]
            loss = F.cross_entropy(outputs,labels_index_t)
            loss_total += loss
            labels_all.append(labels)
            predict_all.append(outputs)
    labels = torch.cat(labels_all,0)
    labels = labels.data.cpu().numpy()
    labels_index=np.argmax(labels,axis=-1)
    predict = torch.cat(predict_all,0)
    predic_prob = F.softmax(predict,dim=-1).cpu().numpy()
    label_out_pred_d = np.argmax(predic_prob,axis=1)
    label_out_pred_m = label_out_pred_d
    label_out_pred=np.asarray(label_out_pred_m==1,dtype=np.int32)
    label_out_pred_prob = predic_prob[:,1]
    acc = accuracy_score(labels_index, label_out_pred_d)
    f1 = f1_score(watch, label_out_pred)
    precision=precision_score(watch, label_out_pred)
    recall=recall_score(watch, label_out_pred)
    try:
        pr,rc,_  = precision_recall_curve(watch,label_out_pred_prob)
        prc_score = auc(rc,pr)
    except:
        prc_score = 0.0

    try:
        rocp = roc_auc_score(watch, label_out_pred_prob)  
    except:
        rocp=0.0

    if test:
        confusion = confusion_matrix(labels_index, label_out_pred_d)
        return acc, loss_total/len(data_iter), rocp, f1, confusion,precision,recall,label_out_pred_prob, prc_score
    return acc, loss_total/len(data_iter), rocp, f1, prc_score

def test( config,model, test_iter,savepath,watch):
    # test
    model.load_state_dict(torch.load(savepath))
    start_time = time.time()
    test_acc, test_loss,test_rocp, test_f1, test_confusion,test_precision,test_recall,out_pred ,prc_score= evaluate(config, model, test_iter,watch, test=True)
    msg = 'Test Loss: {0:>6.4},  Test Acc: {1:>6.4%}, Test ROCP: {2:>6.4}, Test F1: {3:>6.4},Test PRCP: {4:>6.4}'
    print(msg.format(test_loss, test_acc,test_rocp,test_f1,prc_score))
    print("Confusion Matrix")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)



