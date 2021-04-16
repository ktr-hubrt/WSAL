from __future__ import print_function
from __future__ import division
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from torch.nn.functional import softmax
from torch.nn.functional import sigmoid
from torch.nn import MSELoss
from torch.nn import L1Loss
from torch.nn import SmoothL1Loss
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from models import HOE_model
from dataset import dataset_h5
from utils import *
import pdb
import os
import time
from torch import nn

def train_wsal():
    torch.cuda.empty_cache()

    ver ='WSAL'

    videos_pkl_train = "/test/UCF-Crime/UCF/Anomaly_Detection_splits/Anomaly_Train.txt"

    hdf5_path = "/test/UCF-Crime/UCF/gcn_feas.hdf5" 
    mask_path = "/test/UCF-Crime/UCF/gcn_mask.hdf5"


    modality = "rgb"
    gpu_id = 0
    batch_size = 30
    iter_size = 30//batch_size
    random_crop = False
    
    train_loader = torch.utils.data.DataLoader(dataset_h5(videos_pkl_train, hdf5_path, mask_path),
                                                    batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    model = HOE_model(nfeat=1024, nclass=1)

    criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
    Rcriterion = torch.nn.MarginRankingLoss(margin=1.0, reduction = 'mean')

    if gpu_id != -1:

        model = model.cuda(gpu_id)
        criterion = criterion.cuda(gpu_id)
        Rcriterion = Rcriterion.cuda(gpu_id)

    optimizer = optim.Adagrad(model.parameters(), lr=0.001)
    opt_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300,400], gamma=0.5)
    start_epoch = 0
        
    resume = './weights/WSAL_1.1/rgb_300.pth'
    if os.path.isfile(resume):
      print("=> loading checkpoint '{}'".format(resume))
      checkpoint = torch.load(resume)
      start_epoch = checkpoint['epoch']
      opt_scheduler.load_state_dict(checkpoint['scheduler'])
      model.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])

    iter_count = 0
    alpha = 0.5
    vid2mean_pred = {}
    losses = AverageMeter()
    data_time = AverageMeter()
    model.train()

    for epoch in range(start_epoch, 500):
       
        end = time.time()
        pbar = tqdm(total=len(train_loader))
        for step, data in enumerate(train_loader):
            data_time.update(time.time() - end)
            an_feats, no_feats, preds = data
            an_feat, no_feat, pred = Variable(an_feats), Variable(no_feats), Variable(preds)

            
            if gpu_id != -1:
                an_feat = an_feat.cuda(gpu_id)
                no_feat = no_feat.cuda(gpu_id)
                pred = pred.cuda(gpu_id).float()

            if iter_count % iter_size == 0:
                optimizer.zero_grad()

            ano_ss, ano_fea = model(an_feat)
            nor_ss, nor_fea = model(no_feat)

            ano_cos = torch.cosine_similarity(ano_fea[:,1:], ano_fea[:,:-1], dim=2)
            dynamic_score_ano = 1-ano_cos
            nor_cos = torch.cosine_similarity(nor_fea[:,1:], nor_fea[:,:-1], dim=2)
            dynamic_score_nor = 1-nor_cos
            
            ano_max = torch.max(dynamic_score_ano,1)[0]
            nor_max = torch.max(dynamic_score_nor,1)[0]

            loss_dy = Rcriterion(ano_max, nor_max, pred[:,0])
            
            semantic_margin_ano = torch.max(ano_ss,1)[0]-torch.min(ano_ss,1)[0]
            semantic_margin_nor = torch.max(nor_ss,1)[0]-torch.min(nor_ss,1)[0]

            loss_se = Rcriterion(semantic_margin_ano, semantic_margin_nor, pred[:,0])

            loss_3 = torch.mean(torch.sum(dynamic_score_ano,1))+torch.mean(torch.sum(dynamic_score_nor,1))+torch.mean(torch.sum(ano_ss,1))+torch.mean(torch.sum(nor_ss,1))
            loss_5 = torch.mean(torch.sum((dynamic_score_ano[:,:-1]-dynamic_score_ano[:,1:])**2,1))+torch.mean(torch.sum((ano_ss[:,:-1]-ano_ss[:,1:])**2,1))

            loss_train = loss_se + loss_dy+ loss_3*0.00008+ loss_5*0.00008 

            
            iter_count += 1
            loss_train.backward()
            losses.update(loss_train.item(), 1)

            if (iter_count + 1) % iter_size == 0:
                optimizer.step()

            pbar.set_postfix({
                    'Data': '{data_time.val:.3f}({data_time.avg:.4f})\t'.format(data_time=data_time),
                    ver: '{0}'.format(epoch),
                    'lr': '{lr:.5f}\t'.format(lr=optimizer.param_groups[-1]['lr']),
                    'Loss': '{loss.val:.4f}({loss.avg:.4f})\t '.format(loss=losses)
                    })
            
            pbar.update(1)

        pbar.close()
        model_path = 'weights/'+ver+'/'
        if not os.path.isdir(model_path):
            os.mkdir(model_path)

        if epoch%50==0:
            
            state = {
              'epoch': epoch,
              'state_dict': model.state_dict(),
              'optimizer' : optimizer.state_dict(),
              'scheduler': opt_scheduler.state_dict(),
            }
            torch.save(state, model_path+"rgb_%d.pth" % epoch)
            # model = model.cuda(gpu_id)
        # if epoch%25==0:
        losses.reset()
        opt_scheduler.step()


if __name__ == '__main__':
    train_wsal()

