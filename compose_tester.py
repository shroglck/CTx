import torch
import torch.nn as nn
import numpy as np
import pytorchvideo
from pytorchvideo.layers import MultiScaleBlock, SpatioTemporalClsPositionalEncoding
from pytorchvideo.layers.utils import round_width, set_attributes
from pytorchvideo.models.head import create_vit_basic_head
from pytorchvideo.models.weight_init import init_net_weights
from PIL import Image
import cv2 as cv
from Code.helpers import getImg, imgLoader, Imgset, myresize
from torch.utils.data import DataLoader
import numpy as np
import math
import torch
from Data_256.UCF101 import get_ucf101
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from Code.vMFMM import *
from Initialization_Code.config_initialization import vc_num, dataset, categories, data_path, cat_test, device_ids, Astride, Apad, Arf, vMF_kappa, layer,init_path, nn_type, dict_dir, offset, extractor
from Code.helpers import getImg, imgLoader, Imgset, myresize
import torch
from torch.utils.data import DataLoader
import cv2
import glob
import pickle
import os
from Code.model import Net
from Code.helpers import getImg, Imgset, imgLoader, save_checkpoint,getCompositionModel,getVmfKernels, update_clutter_model
from Code.config import device_ids, mix_model_path, categories, categories_train, dict_dir, dataset, data_path, layer, vc_num, model_save_dir, compnet_type,backbone_type, vMF_kappa,num_mixtures
from Code.config import config as cfg
from torch.utils.data import DataLoader
from Code.losses import ClusterLoss
from utils import AverageMeter, accuracy
import torch.nn.functional as F
from new_model import Classification_model
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random
class feature_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_model = torch.hub.load("facebookresearch/pytorchvideo","mvit_base_16x4",pretrained = True)
        self.feature_model.head.proj = torch.nn.Linear(768,101,bias =True)
        state_dict = torch.load("results/AUG_MVIT_UCF101_SUPERVISED_TRAINING/model_best.pth.tar")
        self.feature_model.load_state_dict(state_dict['state_dict'])
    def forward(self,x):

        x = self.feature_model.patch_embed(x)
        x = self.feature_model.cls_positional_encoding(x)
        thw = self.feature_model.cls_positional_encoding.patch_embed_shape()
        for blck in self.feature_model.blocks:
            x,thw= blck(x,thw)
        out = x[:,1:,:]
        out = out.reshape(-1,thw[0],768,thw[1],thw[2])
        return x,thw
class aClassification_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = feature_model()
        self.drop = nn.Dropout(.3)
        self.pool = nn.AvgPool2d(7)
        self.ln  = nn.Linear(768,400)
        #self.occ = nn.Linear(768,1)
    def forward(self,x):
        x,thw = self.feature_extractor(x)
        out = x[:,1:,:]
        out = out.reshape(-1,thw[0],768,thw[1],thw[2])
        cls_token = x[:,0,:]
        cls_token = cls_token.reshape(-1,1,768,1,1)
       # out = out+cls_token
        out = torch.mean(out,dim=1)
        #out = self.pool(out)
        #out = out.reshape(-1,768)
        #x,_ = torch.max(torch.max(x,dim =-1)[0],dim = -1)
        #x = torch.mean(x,dim=1)
        #out = self.drop(out)
        #cls_score = self.ln(out)
        #occ_lik = self.occ(out)
        return out#,occ_lik

#class_specific_performance={}
#total_class_performance = {}
#for i in range(101):
#    class_specific_performance[i] = 0
#    total_class_performance[i]=0
class feature_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_model = torch.hub.load("facebookresearch/pytorchvideo","mvit_base_16x4",pretrained = True)
        self.feature_model.head.proj = torch.nn.Linear(768,101,bias =True)
        #state_dict = torch.load("results/AUG_MVIT_UCF101_SUPERVISED_TRAINING/model_best.pth.tar")
        state_dict = torch.load("results/less_MVIT_UCF101_SUPERVISED_TRAINING/model_best.pth.tar")
        self.feature_model.load_state_dict(state_dict['state_dict'])
    def forward(self,x):

        x = self.feature_model.patch_embed(x)
        x = self.feature_model.cls_positional_encoding(x)
        thw = self.feature_model.cls_positional_encoding.patch_embed_shape
        for blck in self.feature_model.blocks:
            x,thw= blck(x,thw)
        #out = x[:,1:,:]
        #out = out.reshape(-1,thw[0],768,thw[1],thw[2])
        return x,thw

def visualize_response_map(rmap,tit,cbarmax=10.0):

    fig, ax = plt.subplots(nrows=1, ncols=1)
    im = ax.imshow(rmap)
    plt.title(tit,fontsize=18)
    if cbarmax!=0:
        im.set_clim(0.0, cbarmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax,ticks=np.arange(0, cbarmax*2, cbarmax))
    else:
        im.set_clim(0.0, 7.0)
    plt.axis('off')
    occ_img_name = '/tmp/tmp'+str(random.randint(1,1000000))+'.png'
    plt.savefig(occ_img_name, bbox_inches='tight')
    img = cv2.imread(occ_img_name)
    os.remove(occ_img_name)

    # remove white border top and bottom
    loop = True
    while loop:
        img = img[1:img.shape[0],:,:]
        loop = np.sum(img[0,:,:]==255)==(img.shape[1]*3)
    loop = True
    while loop:
        img = img[0:img.shape[0]-2,:,:]
        loop = np.sum(img[img.shape[0]-1,:,:]==255)==(img.shape[1]*3)

    return img


occ_indx = {0:"Desktop",
            1:"Aeroplane",
            2:"Car",
            3:"Human",
            4:"Motorcycle",
            5:"Cat",
            6:"Plant",
            7:"Human",
            8:"Human",
            9: " ",
            10: " ",
            11: " ",
            12: " ",
            13: " "
           }

class feature_mvit(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_model = Classification_model(101)
        #self.feature_model.head.proj = torch.nn.Linear(768,101,bias =True)
        self.feature_model = self.feature_model.cuda()
        #state_dict = torch.load("results/Finern_Augmented_MVIT_UCF101_SUPERVISED_TRAINING/model_best.pth.tar")
        state_dict = torch.load("results/NEWER_AUG_MVIT_UCF101_SUPERVISED_TRAINING/model_best.pth.tar")
        
        self.feature_model.load_state_dict(state_dict['state_dict'])
        
        #state_dict = torch.load("results/AUG_MVIT_UCF101_SUPERVISED_TRAINING/model_best.pth.tar")
        #self.feature_model.load_state_dict(state_dict['state_dict'])
    def forward(self,x,train = False):
        x,thw = self.feature_model.feature_extractor(x)
        
        #x = self.feature_model.cls_positional_encoding(x)
        #thw = self.feature_model.cls_positional_encoding.patch_embed_shape
        #for blck in self.feature_model.blocks:
         #   x,thw= blck(x,thw)
        out = x[:,1:,:]
        cls_token = x[:,0,:]
        cls_token = cls_token.reshape(-1,1,768,1,1)
        out = out.reshape(-1,thw[0],768,thw[1],thw[2])
        out = out+cls_token
        out = torch.mean(out,dim=1)
        ##out_avg,_ = torch.max(out,dim =1)
        return out


class a_feature_mvit(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_model = torch.hub.load("facebookresearch/pytorchvideo","mvit_base_16x4",pretrained = True)
        #state_dict = torch.load("results/AUG_MVIT_UCF101_SUPERVISED_TRAINING/model_best.pth.tar")
        #self.feature_model.load_state_dict(state_dict['state_dict'])
    def forward(self,x):
        x = self.feature_model.patch_embed(x)
        #print(x.shape,"patch embedding")
        x = self.feature_model.cls_positional_encoding(x)
        #print(x.shape,"cls_embedding")
        thw = self.feature_model.cls_positional_encoding.patch_embed_shape
        for blck in self.feature_model.blocks:
            x,thw= blck(x,thw)
        out = x[:,1:,:]
        out = out.reshape(-1,thw[0],768,thw[1],thw[2])
        out_avg,_ = torch.max(out,dim =1)
        return out_avg
def test(test_loader,model): 
    classification_loss = nn.CrossEntropyLoss()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    predicted_target = {}
    ground_truth_target = {}
    predicted_target_not_softmax = {}
 #   video_class_idx={}
    with torch.no_grad():
            for batch_idx, (inputs, targets, video_name) in enumerate(test_loader):
                data_time.update(time.time() - end)
                model.eval()

                inputs = inputs.cuda()
                targets = targets.cuda()
                outputs,_,_ = model(inputs)
                label = targets.detach().cpu().numpy()
                    #outputs = torch.max(outputs,0,keepdims = True)[0]
                out = outputs.argmax(1)
                    
                    
                loss = classification_loss(outputs, targets)/ outputs.shape[0]
                    
                out_prob = F.softmax(outputs, dim=1)
                    
                out_prob = out_prob.cpu().numpy().tolist()
                targets = targets.cpu().numpy().tolist()
                outputs = outputs.cpu().numpy().tolist()
            
                for iterator in range(len(video_name)):
  #                  if video_name[iterator] not in video_class_idx:
  #                      total_class_performance[int(targets[iterator])]+=1
                
                    if video_name[iterator] not in predicted_target:
                        predicted_target[video_name[iterator]] = []
                
                    if video_name[iterator] not in predicted_target_not_softmax:
                        predicted_target_not_softmax[video_name[iterator]] = []

                    if video_name[iterator] not in ground_truth_target:
                        ground_truth_target[video_name[iterator]] = []

                    predicted_target[video_name[iterator]].append(out_prob[iterator])
                    predicted_target_not_softmax[video_name[iterator]].append(outputs[iterator])
                    ground_truth_target[video_name[iterator]].append(targets[iterator])
                
                
                losses.update(loss.item(), inputs.shape[0])
                batch_time.update(time.time() - end)
                end = time.time()
            
    for key in predicted_target:
        clip_values = np.array(predicted_target[key]).mean(axis=0)
        video_pred = np.argmax(clip_values)
        predicted_target[key] = video_pred
    
    for key in predicted_target_not_softmax:
        clip_values = np.array(predicted_target_not_softmax[key]).mean(axis=0)
        video_pred = np.argmax(clip_values)
        predicted_target_not_softmax[key] = video_pred
    
    for key in ground_truth_target:
        clip_values = np.array(ground_truth_target[key]).mean(axis=0)
        ground_truth_target[key] = int(clip_values)

    pred_values = []
    pred_values_not_softmax = []
    target_values = []
    #for key in predicted_target:
    #    if predicted_target[key] == ground_truth_target[key]:
    #        class_specific_performance[video_class_idx[key]] +=1
    

    for key in predicted_target:
        pred_values.append(predicted_target[key])
        pred_values_not_softmax.append(predicted_target_not_softmax[key])
        target_values.append(ground_truth_target[key])
    
    pred_values = np.array(pred_values)
    pred_values_not_softmax = np.array(pred_values_not_softmax)
    target_values = np.array(target_values)

    secondary_accuracy = (pred_values == target_values)*1
    secondary_accuracy = (sum(secondary_accuracy)/len(secondary_accuracy))*100
    print(f'test accuracy after softmax: {secondary_accuracy}')

    secondary_accuracy_not_softmax = (pred_values_not_softmax == target_values)*1
    secondary_accuracy_not_softmax = (sum(secondary_accuracy_not_softmax)/len(secondary_accuracy_not_softmax))*100
    print(f'test accuracy before softmax: {secondary_accuracy_not_softmax}')
    #for i in class_specific_performance:
    #    total_class_performance[i] = class_specific_performance[i]/total_class_performance[i] 
    #print(class_specific_performance)
    #print(total_class_performance)


def main():
    parser = argparse.ArgumentParser(description='PyTorch Classification Testing')
    parser.add_argument('--occ_index',default=1,type = int,help = "Occluder to be used")
    parser.add_argument('--occ_size',default=60,type = int,help = "Area of the image to be occluded")
    parser.add_argument('--motion' ,default ="random_placement", type = str , help = "motion followed by the occluder")
    args = parser.parse_args()
    occ_dict= {"occlusion_index":args.occ_index,"occlusion_size":args.occ_size,"motion":args.motion}
    
    print("config occluder {}, occluder size{}, occluder motion {}".format(occ_indx[args.occ_index],args.occ_size,args.motion))
    alpha = 3  # vc-loss
    beta = 3 # mix loss
    likely = 0.5 # occlusion likelihood
    lr = 1e-4 # learning rate
    batch_size = 1 # these are pseudo batches as the aspect ratio of images for CompNets is not square
    # Training setup
    vc_flag = True # train the vMF kernels
    mix_flag = True # train mixture components
    ncoord_it = 100 	#number of epochs to train

    bool_mixture_model_bg = False #True: use a mixture of background models per pixel, False: use one bg model for whole image
    bool_load_pretrained_model = False
    bool_train_with_occluders = False

    extractor = feature_mvit()#Classification_model()
    extractor.cuda()
    dict_dir = "models/init_vgg/dictionary_vgg/finer_dictionary_mvit_prertrained_768.pickle"
    weights = getVmfKernels(dict_dir, device_ids)
    
    occ_likely =[]
    for i in range(101): #changed from len()
        # setting the same occlusion likelihood for all classes
        occ_likely.append(.6)

    mix_models = getCompositionModel(device_ids,mix_model_path,layer,categories_train,compnet_type=compnet_type)
    model = Net(extractor, weights,vMF_kappa, occ_likely, mix_models,
              bool_mixture_bg=bool_mixture_model_bg,compnet_type=compnet_type,num_mixtures=num_mixtures, 
          vc_thresholds=cfg.MODEL.VC_THRESHOLD)
    t = torch.load("./models/test_mvit_train_only_class_with_model_wiht concat_pool5_a1_b5_vcTrue_mixTrue_occlikely0.6_vc768_lr_1e-06_pascal3d+_pretrainedFalse_epochs_100_occFalse_backbonevgg_0/vc81.pth")
    #t = torch.load("./models/trial_single_test_mvit_train_only_class_with_model_wiht concat_pool5_a3_b3_vcTrue_mixTrue_occlikely0.7_vc768_lr_1e-05_pascal3d+_pretrainedFalse_epochs_100_occFalse_backbonevgg_0/vc10.pth") 
    #torch.load("./models/spatial_trial_single_test_mvit_train_only_class_with_model_wiht concat_pool5_a3_b3_vcTrue_mixTrue_occlikely0.7_vc768_lr_1e-05_pascal3d+_pretrainedFalse_epochs_15_occFalse_backbonevgg_0/vc10.pth")#models/test_mvit_train_only_class_with_model_wiht concat_pool5_a3_b3_vcTrue_mixTrue_occlikely0.6_vc768_lr_1e-05_pascal3d+_pretrainedFalse_epochs_100_occFalse_backbonevgg_0/vc3.pth")#torch.load("models/test_mvit_train_only_class_with_model_wiht concat_pool5_a1_b5_vcTrue_mixTrue_occlikely0.6_vc768_lr_0.0001_pascal3d+_pretrainedFalse_epochs_100_occFalse_backbonevgg_0/vc9.pth")
    model.load_state_dict(t['state_dict'])
    for i in range(101):
        print(i)
        
        train_sampler = RandomSampler
        train_dataset, test_dataset =  get_ucf101(cl =i,root='Data_256',frames_path ='/home/c3-0/datasets/UCF101_frames/frames_256',occ_dict = occ_dict)
    
        train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler(train_dataset),
        batch_size=2,
        num_workers=4,
        drop_last=True,
        pin_memory=True)

        test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=1,
        num_workers=4,
        pin_memory=True)
        
        test(test_loader,model)
    
    
    
if __name__ == "__main__":
    main()

        
