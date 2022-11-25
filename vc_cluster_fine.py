import torch
import torch.nn as nn
import numpy as np
import pytorchvideo
from pytorchvideo.layers import MultiScaleBlock, SpatioTemporalClsPositionalEncoding
from pytorchvideo.layers.utils import round_width, set_attributes
from pytorchvideo.models.head import create_vit_basic_head
from pytorchvideo.models.weight_init import init_net_weights
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from Initialization_Code.vcdist_funcs import vc_dis_paral, vc_dis_paral_full
import time
import pickle
import os
from Initialization_Code.config_initialization import vc_num, dataset, categories, data_path, cat_test, device_ids, Astride, Apad, Arf,vMF_kappa, layer,init_path, dict_dir, sim_dir, extractor
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
import torchvision
from PIL import Image
import cv2 as cv
from new_model import Classification_model
import pytorchvideo
import pytorchvideo.data
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip
)


class feature_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_model = torch.hub.load("facebookresearch/pytorchvideo","mvit_base_16x4",pretrained = True)
        self.feature_model.head.proj = torch.nn.Linear(768,101,bias =True)
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
        #out = out+cls_token
        out = torch.mean(out,dim=1)
        #out = self.pool(out)
        #out = out.reshape(-1,768)
        #x,_ = torch.max(torch.max(x,dim =-1)[0],dim = -1)
        #x = torch.mean(x,dim=1)
        #out = self.drop(out)
        #cls_score = self.ln(out)
        #occ_lik = self.occ(out)
        return out#,occ_lik


class feature_mvit(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_model = Classification_model()
        #self.feature_model.head.proj = torch.nn.Linear(768,101,bias =True)
        self.feature_model = self.feature_model.cuda()
        state_dict = torch.load("results/NEWER_AUG_MVIT_UCF101_SUPERVISED_TRAINING/model_best.pth.tar")
        #state_dict = torch.load("results/Finern_Augmented_MVIT_UCF101_SUPERVISED_TRAINING/model_best.pth.tar")
        self.feature_model.load_state_dict(state_dict['state_dict'])
        
        #state_dict = torch.load("results/AUG_MVIT_UCF101_SUPERVISED_TRAINING/model_best.pth.tar")
        #self.feature_model.load_state_dict(state_dict['state_dict'])
    def forward(self,x):
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
    

def load_vid(video_path):
    no_frames = len(os.listdir(video_path))
    skip_rate = 1
    total_frames = 16*skip_rate

    if total_frames > no_frames:
        skip_rate = skip_rate -1
        if skip_rate == 0:
            skip_rate = 1
        total_frames = 16*skip_rate

    try:
        start_frame = random.randint(0, no_frames - total_frames) ## 32, 16 frames
    except:
        start_frame = 0
    video_container = []
    for item in range(start_frame, start_frame + total_frames, skip_rate):
        image_name = "frame"+str(item+1).zfill(6) + '.jpg'
        image_path = os.path.join(video_path, image_name)
        current_image = np.asarray(Image.open(image_path).convert('RGB').resize((224,224),0))
        video_container.append(current_image)
    video_container = np.array(video_container)
    
    print(video_container.shape)
    return video_container
def save_vid(one_vid,name = "config_name"):
    index = np.random.randint(low = 0,high = len(test_dataset)-1,size=6)
    #subset = torch.utils.data.Subset(test_dataset, index)
    _,h,w,_ = one_vid.shape
    if not os.path.isdir(name):
        os.mkdir(name)
    out = cv.VideoWriter(os.path.join(name+'.avi'),cv.VideoWriter_fourcc('M','J','P','G'), 12, (h,w))
    for i in range(16):
        frame = one_vid[i,:,:,:]
        #print(frame.shape)
        frame = frame.astype(np.uint8)
        out.write(frame)
    out.release()
    

def train(model,train_loader,vMF_kappa):
    
    #model  = another_model()
    img_per_cat = 72
    samp_size_per_img = 20
    imgs_par_cat =np.zeros(101)
    bool_load_existing_cluster = False
    bins = 4
    #nimgs = len(train_dataset)
    loc_set = []
    feat_set = []
    nfeats = 0
    vc_num = 768
    fname = []
    for ii,i in enumerate(train_loader):
        #x,y,z,_ = i
        x = i["video"]
        y= i["label"]
        y = int(y.detach().numpy())
        #fname.append(z)
        #print(y)
        if y in list(range(101)) and imgs_par_cat[y]<img_per_cat:
            x = x.cuda()
            fname = []

            with torch.no_grad():
                tmp = model(x).detach().cpu().numpy()
                tmp = tmp.squeeze(0)
                
                #print(tmp.shape)
                height, width = tmp.shape[-2:]
                #print(height,width)
                #offset = 1
                tmp = tmp[:,offset:height - offset, offset:width - offset]
                #print(tmp.shape)
                gtmp = tmp.reshape(tmp.shape[0], -1)
                #s = np.argsort(np.sum(gtmp*gtmp,0))
                #print(gtmp.shape)
                if gtmp.shape[1] >= samp_size_per_img:
                    rand_idx = np.random.permutation(gtmp.shape[1])[:samp_size_per_img]
                else:
                    rand_idx = np.random.permutation(gtmp.shape[1])[:samp_size_per_img - gtmp.shape[1]]
            #rand_idx = np.append(range(gtmp.shape[1]), rand_idx)
                tmp_feats = gtmp[:, rand_idx].T
                cnt = 0
                for rr in rand_idx:
                    ihi, iwi = np.unravel_index(rr, (height - 2 * offset, width - 2 * offset))
         #   print(ihi,iwi)
                    hi = (ihi+offset)*(x.shape[2]/height)-Apad
                    wi = (iwi + offset)*(x.shape[3]/width)-Apad
          #  print(hi,wi)
                    loc_set.append([y, ii,hi,wi,hi+Arf,wi+Arf])
           # print([y, ii, hi,wi,hi+Arf,wi+Arf])
                    feat_set.append(tmp_feats[cnt,:])
            
                    cnt+=1
        
            imgs_par_cat[y]+=1
            
    #print(cnt)

    feat_set = np.asarray(feat_set)
    loc_set = np.asarray(loc_set).T

#print(feat_set.shape)
    new_model = vMFMM(768, 'k++')
    new_model.fit(feat_set, vMF_kappa, max_it=200)
    with open(dict_dir+'dictionary_{}_{}.pickle'.format("finer_mvit_prertrained","768"), 'wb') as fh:#"finer_mvit_kinetics_prertrained"
        pickle.dump(new_model.mu, fh)

    num = 50
    SORTED_IDX = []
    SORTED_LOC = []
    for vc_i in range(768):
        sort_idx = np.argsort(-new_model.p[:, vc_i])[0:num]
        SORTED_IDX.append(sort_idx)
        tmp=[]
        for idx in range(num):
            iloc = loc_set[:, sort_idx[idx]]
            tmp.append(iloc)
        SORTED_LOC.append(tmp)


    with open(dict_dir + 'dictionary_{}_{}_p.pickle'.format("finer_lvit_pretrained",768), 'wb') as fh:
        pickle.dump(new_model.p, fh)
    example = [None for vc_i in range(768)]
    out_dir = dict_dir + '/cluster_images_{}_{}/'.format("mvit",vc_num)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    for vc_i in range(vc_num):
        patch_set = np.zeros(((Arf**2)*3, num)).astype('uint8')
        sort_idx = SORTED_IDX[vc_i]#np.argsort(-p[:,vc_i])[0:num]
        opath = out_dir + str(vc_i) + '/'
        if not os.path.exists(opath):
            os.makedirs(opath)
        locs=[]
        for idx in range(num):
            iloc = loc_set[:,sort_idx[idx]]
            category = iloc[0]
            loc = iloc[1:6].astype(int)
            if not loc[0] in locs:
                locs.append(loc[0])
            #img = cv2.imread(imgs[int(loc[0])])
                img = load_vid(fname[int(loc[0])][0])
            #img = myresize(img, 224, 'short')
                patch = img[:,loc[1]:loc[3], loc[2]:loc[4], :]
            #patch_set[:,idx] = patch.flatten()
                if patch.size:
                #cv2.imwrite(opath+str(idx)+'.JPEG',patch)
                    name= fname[int(loc[0])][0].split("/")[-1]
                    save_vid(patch,opath+name+str(idx))
                  #example[vc_i] = np.copy(patch_set)
    
    
    
    
    print("finished")
def main_vc(dataset,kappa):
    #train_dataset, test_dataset =  get_ucf101('Data',frames_path ='/home/yogesh/Naman/ucf-101/frames-128x128')
    train_dataset, test_dataset =  get_ucf101(root = 'Data_256',frames_path ='/home/c3-0/datasets/UCF101_frames/frames_256')#/home/c3-0/datasets/UCF101_frames/frames_256',/home/yogesh/Naman/ucf-101/frames-128x128
    train_transform = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    UniformTemporalSubsample(16),
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    RandomCrop(224),
                 ]
                ),
              ),
            ]
        )
    train_dataset = pytorchvideo.data.Kinetics(
            data_path=os.path.join("/home/ak119590/datasets/K400/videos_256", "train"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("random",2),
           transform=train_transform
      )
    
    if dataset == "UCF101":
        train_dataset, test_dataset =  get_ucf101(root = 'Data_256',frames_path ='/home/c3-0/datasets/UCF101_frames/frames_256')
        model = feature_mvit().cuda()
    elif dataset == "Kinetics":
        train_dataset = pytorchvideo.data.Kinetics(
            data_path=os.path.join("/home/ak119590/datasets/K400/videos_256", "train"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("random",2),
           transform=train_transform
      )
        model = aClassification_model()
    
        
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=4,
        drop_last=True,
        pin_memory=True)
    #state = torch.load("./Kineticsrun/checkpoints/checkpoint_epoch_00010.pyth")


    model = feature_mvit().cuda()
    #model.load_state_dict(state["model_state"])

    main(model,train_loader)
    
if __name__ == "__main__":
    #train_dataset, test_dataset =  get_ucf101('Data',frames_path ='/home/yogesh/Naman/ucf-101/frames-128x128')
    train_dataset, test_dataset =  get_ucf101(root = 'Data_256',frames_path ='/home/c3-0/datasets/UCF101_frames/frames_256')#/home/c3-0/datasets/UCF101_frames/frames_256',/home/yogesh/Naman/ucf-101/frames-128x128
    train_transform = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    UniformTemporalSubsample(16),
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    RandomCrop(224),
                 ]
                ),
              ),
            ]
        )
    #train_dataset = pytorchvideo.data.Kinetics(
    #        data_path=os.path.join("/home/ak119590/datasets/K400/videos_256", "train"),
    #        clip_sampler=pytorchvideo.data.make_clip_sampler("random",2),
    #       transform=train_transform
    #  )
    
    train_sampler = RandomSampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=4,
        drop_last=True,
        pin_memory=True)
    #state = torch.load("./Kineticsrun/checkpoints/checkpoint_epoch_00010.pyth")


    model = feature_mvit().cuda()
    #model.load_state_dict(state["model_state"])

    main(model,train_loader)
    
