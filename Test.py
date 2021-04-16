from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import numpy as np
from models import HOE_model
from dataset import dataset_h5_test
from utils import *
import os
from os import path
from matplotlib import pyplot as plt
import glob
from sklearn.metrics import roc_auc_score, roc_curve
import time
from tqdm import tqdm


ver = 'wsal/'
modality = "rgb"
gpu_id = 0

model_paths = "./weights/"+ver
output_folder = "./results/"+ver
videos_pkl_train = "/test/UCF-Crime/UCF/Anomaly_Detection_splits/variables.txt"
hdf5_path = "/test/UCF-Crime/UCF/gcn_test.hdf5" 
data_root = '/pcalab/tmp/UCF-Crime/UCF_Crimes/Anomaly_train_test_imgs/test/'
UCFdata_LABEL_PATH = '/test/UCF-Crime/Temporal_Anomaly_Annotation_For_Testing_Videos/Txt_formate/Temporal_Anomaly_Annotation.txt'

def get_gts(LABEL_PATH):
    
    video_path_list = []
    videos = {}
    for line in open(LABEL_PATH):
        video_path_list.append(line)
    for video in video_path_list:
        data_dir = data_root + video.split(' ')[0][:-4]
        img_list = glob.glob(os.path.join(data_dir, '*.jpg'))
        start_1 = int(video.split(' ')[4])
        end_1 = int(video.split(' ')[6])
        start_2 = int(video.split(' ')[8])
        end_2 = int(video.split(' ')[10])
        sub_video_gt = np.zeros((len(img_list),), dtype=np.int8)
        if start_1>=0 and end_1>=0:
            sub_video_gt[start_1-1:end_1]=1
        if start_2>=0 and end_2>=0:
            sub_video_gt[start_2-1:end_2]=1
        # pdb.set_trace()
        if len(img_list) ==0:
            pdb.set_trace()
        # pdb.set_trace()
        videos[video.split(' ')[0][:-4]] = sub_video_gt
    return videos

def evaluate_result_ucf(vid2abnormality, fpath, videos):
        gt = []
        ans = []
        GT = []
        ANS = []
        
        video_labels = []
        # video_scores = []

        for vid in videos:
            if vid not in vid2abnormality.keys():
                print("The video %s is excluded on the result!" % vid)
                continue

            video_labels.append(videos[vid].max())
            # video_scores.append(glo_scores[vid][0])
            cur_ab = np.array(vid2abnormality[vid])
            cur_gt = np.array(videos[vid])
            ratio = float(len(cur_gt)) / float(len(cur_ab))
            cur_ans = np.zeros_like(cur_gt, dtype='float32')
            for i in range(len(cur_ab)):
                b = int(i * ratio + 0.5)
                e = int((i + 1) * ratio + 0.5)
                # if 'Normal' in vid:
                #     cur_ans[b: e] = 0.0
                # else:
                #     cur_ans[b: e] = 1.0
                cur_ans[b: e] = cur_ab[i]
            if 'Normal' not in vid:
                gt.extend(cur_gt.tolist())
                ans.extend(cur_ans.tolist())
            
            GT.extend(cur_gt.tolist())
            ANS.extend(cur_ans.tolist())
            continue
            fig = plt.figure()
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            #print vid, tf_idf
            ax1.plot(cur_gt, color='r')
            ax2.plot(cur_ans, color='g')
            plt.title(vid)
            plt.show()
            root = 'png'
            plt.savefig(root+'/'+vid+'.png')
            # print('Save: ',root +'/'+vid+'.png')
            plt.close()
            # pdb.set_trace()
        # pdb.set_trace()
        
        if not os.path.isdir(fpath):
            os.mkdir(fpath)
        # output_file = fpath+"/%s_rgb.npz" % vid[0]

        output_file = fpath+"/gt-ans.npz"
        
        ret = roc_auc_score(gt, ans)
        Ret = roc_auc_score(GT, ANS)
        # video_auc = roc_auc_score(video_labels, video_scores)
        np.savez(output_file, gt=gt, ans=ans, GT=GT, ANS=ANS, v_gt=video_labels,)
        # pdb.set_trace()
        return Ret,ret

def eval_model(gcn_model_path, model, train_loader, version):
    
    checkpoint = torch.load(gcn_model_path,map_location='cuda:0')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    vid2ans = {}
    pbar = tqdm(total=len(train_loader))
    for step, data in enumerate(train_loader):
        feat, pred, vid = data
        feat, pred = Variable(feat), Variable(pred)

        if gpu_id != -1:
            feat = feat.cuda(gpu_id)
            pred = pred.cuda(gpu_id)
        score, fea = model(feat)
        
        se_score = score.squeeze()

        ano_score = torch.zeros_like(fea[0,:,0])
        ano_cos = torch.cosine_similarity(fea[0,:-1], fea[0,1:], dim=1)

        ano_score[:-1] += 1-ano_cos
        ano_score[1:] += 1-ano_cos
        ano_score[1:-1] /= 2
        ano_score = ano_score.data.cpu().numpy().flatten()

        se_score = se_score.data.cpu().numpy().flatten()

        new_pred = ano_score.copy()

        fpath = output_folder+version
        if not os.path.isdir(fpath):
            os.mkdir(fpath)
        output_file = fpath+"/%s_rgb.npz" % vid[0]

        np.savez(output_file, output_dy=ano_score.copy(), output_se=se_score.copy())

        pbar.set_postfix({'Epoch': '{0}'.format(version),
                            'Score': '{0}'.format(new_pred.max()),
                            'Vid': '{0}'.format(vid[0]),    
                            })
        pbar.update(1)
    pbar.close()
    return

if __name__ == '__main__':
    
    video_gts = get_gts(UCFdata_LABEL_PATH)
    train_loader = torch.utils.data.DataLoader(dataset_h5_test(videos_pkl_train,hdf5_path), num_workers=0, pin_memory=True)
    model = HOE_model(nfeat=1024, nclass=1)
    if gpu_id != -1:
        model = model.cuda(gpu_id)

    snapshot_dir = model_paths
    psnr_dir = output_folder
    if not os.path.isdir(psnr_dir):
        os.mkdir(psnr_dir)
    f2 = open(videos_pkl_train,"r")
    test_videos = f2.readlines()
    video_dict={}
    for q in range(len(test_videos)):
        path = test_videos[q].strip().split('/')[-1].split(' ')[0]
        num =  test_videos[q].strip().split('/')[-1].split(' ')[1]
        video_dict[path] = int(num)

    if os.path.isdir(snapshot_dir):
            def check_ckpt_valid(ckpt_name):
                is_valid = False
                if ckpt_name.startswith('rgb'):
                    ckpt_path = os.path.join(snapshot_dir, ckpt_name)
                    if os.path.exists(ckpt_path):
                        is_valid = True

                return is_valid, ckpt_name.split('.')[0]

            def scan_psnr_folder():
                tested_ckpt_in_psnr_sets = set()
                for test_psnr in os.listdir(psnr_dir):
                    tested_ckpt_in_psnr_sets.add(test_psnr)
                return tested_ckpt_in_psnr_sets

            def scan_model_folder():
                saved_models = set()
                for ckpt_name in os.listdir(snapshot_dir):
                    is_valid, ckpt = check_ckpt_valid(ckpt_name)
                    if is_valid:
                        saved_models.add(ckpt)
                return saved_models

            tested_ckpt_sets = scan_psnr_folder()
            best_auc = 0.0
            best_id = 0
            ckpt_dict = {}
            while True:
                all_model_ckpts = scan_model_folder()
                new_model_ckpts = all_model_ckpts - tested_ckpt_sets
                for ckpt_name in new_model_ckpts:
                    ckpt = os.path.join(snapshot_dir, ckpt_name+'.pth')
                    eval_model(ckpt, model, train_loader, ckpt_name)
                    tested_ckpt_sets.add(ckpt_name)

                print('waiting for models...')
                if not os.path.isdir(psnr_dir):
                    loss_file_list = [psnr_dir]
                else:
                    loss_file_list = os.listdir(psnr_dir)
                    loss_file_list = [os.path.join(psnr_dir, sub_loss_file) for sub_loss_file in loss_file_list]

                def getkey(s):
                    return int(s.split('/')[-1].split('_')[1])

                new_list = sorted(loss_file_list, key=getkey)
                for sub_loss_file in new_list:

                    version = 'Epoch %.2d'% int(sub_loss_file.split('/')[-1].split('_')[1])
                    if version in ckpt_dict.keys():
                        ret = ckpt_dict[version] 
                        print(version+" AUC@ROC: %.4f" % ret+".     Best: "+best_id+" AUC@ROC: %.4f(%.4f,%.4f) " % (best_auc,best_dy,best_se) + ver[:-1])
                        continue

                    vid2abscore = {}
                    vid2abscore_d = {}
                    vid2abscore_s = {}
                    for vid in test_videos:
                        new_id = vid.strip().split('/')[-1].split(' ')[0]

                        with np.load(sub_loss_file+"/%s_rgb.npz" % new_id, 'r') as f:
                            
                            tmpdy = f["output_dy"]
                            tmpdy = np.reshape(tmpdy,(-1,1))
                            tmpdy = tmpdy.mean(axis=1)
                            
                            tmpse = f["output_se"]
                            tmpse = np.reshape(tmpse,(-1,1))
                            tmpse = tmpse.mean(axis=1)

                            tmp = (tmpdy+tmpse)/2
                            vid2abscore[new_id] = tmp
                            vid2abscore_d[new_id] = tmpdy
                            vid2abscore_s[new_id] = tmpse
                    ret,_ = evaluate_result_ucf(vid2abscore, sub_loss_file, video_gts)
                    Retdy,_ = evaluate_result_ucf(vid2abscore_d, sub_loss_file, video_gts)
                    Retse,_ = evaluate_result_ucf(vid2abscore_s, sub_loss_file, video_gts)
                    # import pdb;pdb.set_trace()
                    ckpt_dict[version] = ret
                    if ret>best_auc:
                        best_id = version
                        best_auc = ret
                        best_se = Retse
                        best_dy = Retdy
                    print(version+" AUC@ROC: %.4f(%.4f,%.4f)" % (ret,Retdy,Retse)+".     Best: "+best_id+" AUC@ROC: %.4f(%.4f,%.4f)  " % (best_auc,best_dy,best_se) + ver[:-1])
                time.sleep(60)
    else:
            # validate(snapshot_dir, dataset_name, evaluate_name)
            ckpt = snapshot_dir
            version = 'test'
            eval_model(ckpt, model, train_loader, version)
            
            fpath = path.join(output_folder,version)
            # output_file = path.join(fpath, "%s_rgb.npz" % vid[0])
            vid2abscore = {}
            vid2abscore_d = {}
            vid2abscore_s = {}
            for vid in test_videos:
                new_id = vid.strip().split('/')[-1].split(' ')[0]
                # print(new_id)
                # if 'Normal' in new_id:
                #     continue
                with np.load(fpath+"/%s_rgb.npz" % new_id, 'r') as f:
                    tmpdy = f["output_dy"]
                    tmpdy = np.reshape(tmpdy,(-1,1))
                    tmpdy = tmpdy.mean(axis=1)
                    
                    tmpse = f["output_se"]
                    tmpse = np.reshape(tmpse,(-1,1))
                    tmpse = tmpse.mean(axis=1)
                    # tmpse = tmps-tmpse.min()

                    tmp = (tmp+tmpse)/2
                    vid2abscore[new_id] = tmp
                    vid2abscore_d[new_id] = tmpdy
                    vid2abscore_s[new_id] = tmpse
            ret,_ = evaluate_result_ucf(vid2abscore, sub_loss_file, video_gts)
            Retdy,_ = evaluate_result_ucf(vid2abscore_d, sub_loss_file, video_gts)
            Retse,_ = evaluate_result_ucf(vid2abscore_s, sub_loss_file, video_gts)
            
            print(version+" AUC@ROC: %.4f" % ret)
            


    