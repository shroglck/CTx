from Code.model import Net
from Code.helpers import getImg, Imgset, imgLoader, save_checkpoint,getCompositionModel,getVmfKernels, update_clutter_model
from Code.config import device_ids, mix_model_path, categories, categories_train, dict_dir, dataset, data_path, layer, vc_num, model_save_dir, compnet_type,backbone_type, vMF_kappa,num_mixtures
from Code.config import config as cfg
from torch.utils.data import DataLoader
from Code.losses import ClusterLoss
from Code.model import resnet_feature_extractor
import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np
import pytorchvideo
from pytorchvideo.layers import MultiScaleBlock, SpatioTemporalClsPositionalEncoding
from pytorchvideo.layers.utils import round_width, set_attributes
from pytorchvideo.models.head import create_vit_basic_head
from pytorchvideo.models.weight_init import init_net_weights
from pytorchvideo.layers import convolutions
import time
import os
import torch
import torch.nn as nn
import numpy as np
import random
from utils import AverageMeter, accuracy
from Data_256.UCF101 import get_ucf101
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
import torchvision
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
    RandomHorizontalFlip,
    CenterCrop
)
from Kinetics.dataset import labeled_video_dataset


from new_model import Classification_model
vMF_kappa=76

class another_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = torchvision.models.video.r3d_18(pretrained = True)
    def forward(self,x):
        x = self.feature.stem(x)
        x = self.feature.layer1(x)
        x = self.feature.layer2(x)
        x = self.feature.layer3(x)
        x = self.feature.layer4(x)
        x = torch.mean(x,dim = 2)
        return x


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
    def forward(self,x,train=False):
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
        #else:
        #    out = 2*99/100*out+2/100*cls_token
        
        out = torch.mean(out,dim=1)
        ##out_avg,_ = torch.max(out,dim =1)
        return out

class a_feature_mvit(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_model = torch.hub.load("facebookresearch/pytorchvideo","mvit_base_16x4",pretrained = True)
        self.feature_model.head.proj = torch.nn.Linear(768,101,bias =True)
        self.feature_model = self.feature_model.cuda()
        state_dict = torch.load("results/AUG_MVIT_UCF101_SUPERVISED_TRAINING/model_best.pth.tar")
        self.feature_model.load_state_dict(state_dict['state_dict'])
        #self.t_reduction = torch.nn.ModuleList()
        #self.t_reduction.append(convolutions.create_conv_2plus1d(in_channels = 8,out_channels =4,kernel_size = (3,3,3),stride = (1,1,1),padding=(1,1,1)))
        #self.t_reduction.append(convolutions.create_conv_2plus1d(in_channels = 4,out_channels =2,kernel_size = (3,3,3),stride = (1,1,1),padding=(1,1,1)))
        #self.t_reduction.append(convolutions.create_conv_2plus1d(in_channels = 2,out_channels =1,kernel_size = (3,3,3),stride = (1,1,1),padding=(1,1,1)))
  
    def forward(self,x):
        x = self.feature_model.patch_embed(x)
        x = self.feature_model.cls_positional_encoding(x)
        thw = self.feature_model.cls_positional_encoding.patch_embed_shape
        for blck in self.feature_model.blocks:
            x,thw= blck(x,thw)
        out = x[:,1:,:]
        out = out.reshape(-1,thw[0],768,thw[1],thw[2])
        #for lay in self.t_reduction:
         #   out = lay(out)
        if torch.sum(torch.isnan(out)) > 0:
            print("'problem")
            print(self.t_reduction[0].parameters.weights)
        out,_= torch.max(out,dim =1)
        #print(out.shape)
        return out.squeeze(1)
class feature_model_cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = torch.hub.load("facebookresearch/pytorchvideo",model = "slow_r50",pretrained = True)
    def forward(self,x):
        for block in self.feature_extractor.blocks[:-1]:
            x = block(x)
        if torch.sum(torch.isnan(x)) > 0:
            print("'problem")
            #print(x[0].parameters.weights)
        
        x = torch.mean(x,dim =2)
        #x = x.squeeze(0)
        #x = x.permute(1,0,2,3)
        #x = torch.mean
        #x = x.reshape(-1,16)
        #x = torch.nn.functional.max_pool1d(x,4,4)
        #x = x.reshape(4,512,28,28)
        return x 
#class feature_model(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.feature_model = torch.hub.load("facebookresearch/pytorchvideo","mvit_base_16x4",pretrained = True)
#        #self.feature_model.head.proj = torch.nn.Linear(768,101,bias =True)
#        #state_dict = torch.load("results/AUG_MVIT_UCF101_SUPERVISED_TRAINING/model_best.pth.tar")
        ##self.feature_model.load_state_dict(state_dict['state_dict'])
#    def forward(self,x):
#
 #       x = self.feature_model.patch_embed(x)
  #      x = self.feature_model.cls_positional_encoding(x)
   #     thw = self.feature_model.cls_positional_encoding.patch_embed_shape()
  #      for blck in self.feature_model.blocks:
    #        x,thw= blck(x,thw)
        #out = x[:,1:,:]
        #out = out.reshape(-1,thw[0],768,thw[1],thw[2])
     #   return x,thw
#class Classification_model(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.feature_extractor = feature_model()
#        self.drop = nn.Dropout(.3)
#        self.pool = nn.AvgPool2d(7)
#        self.ln  = nn.Linear(768,400)
        #self.occ = nn.Linear(768,1)
 #   def forward(self,x):
  #      x,thw = self.feature_extractor(x)
   #     out = x[:,1:,:]
    #    out = out.reshape(-1,thw[0],768,thw[1],thw[2])
     #   cls_token = x[:,0,:]
      #  cls_token = cls_token.reshape(-1,1,768,1,1)
       # out = out+cls_token
        ##out = torch.mean(out,dim=1)
        #out = self.pool(out)
        #out = out.reshape(-1,768)
        #x,_ = torch.max(torch.max(x,dim =-1)[0],dim = -1)
        #x = torch.mean(x,dim=1)
        #out = self.drop(out)
        #cls_score = self.ln(out)
        #occ_lik = self.occ(out)
        #return out#,occ_lik



#---------------------
# Training Parameters
#---------------------
alpha = 3  # vc-loss
beta = 3 # mix loss
likely = 0.7 # occlusion likelihood
lr = 1e-5 # learning rate
batch_size = 1 # these are pseudo batches as the aspect ratio of images for CompNets is not square
# Training setup
vc_flag = True # train the vMF kernels
mix_flag = True # train mixture components
ncoord_it = 15 	#number of epochs to train

bool_mixture_model_bg = False #True: use a mixture of background models per pixel, False: use one bg model for whole image
bool_load_pretrained_model = False
bool_train_with_occluders = False
#print("SGD")

if bool_train_with_occluders:
    occ_levels_train = ['ZERO', 'ONE', 'FIVE', 'NINE']
else:
    occ_levels_train = ['ZERO']

out_dir = model_save_dir + 'Kinerics_spatial_trial_single_test_mvit_train_only_class_with_model_wiht concat_{}_a{}_b{}_vc{}_mix{}_occlikely{}_vc{}_lr_{}_{}_pretrained{}_epochs_{}_occ{}_backbone{}_{}/'.format(
    layer, alpha,beta, vc_flag, mix_flag, likely, vc_num, lr, dataset, bool_load_pretrained_model,ncoord_it,bool_train_with_occluders,backbone_type,device_ids[0])


def train(model, train_data, val_data, epochs, batch_size, learning_rate, savedir, alpha=3,beta=3, vc_flag=True, mix_flag=False):
    
    
    train_transform = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    UniformTemporalSubsample(16),
                    #Lambda(lambda x: x / 255.0),
                    #Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    RandomCrop(224),
                 ]
                ),
              ),
            ]
    )
    test_transform = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    UniformTemporalSubsample(16),
                    #Lambda(lambda x: x / 255.0),
                    #Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    CenterCrop(224),
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
    
    test_datasets = [labeled_video_dataset(cl = i,
            data_path=os.path.join("/home/ak119590/datasets/K400/videos_256", "val"),#os.path.join("/home/ak119590/datasets/K400/videos_256/test", "test"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform",.53),
           transform=test_transform,
        occ = True
      ) for i in range(101)]
    
    train_sampler = RandomSampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=4,
        pin_memory = True,        
        drop_last=True,
        )

  
    
    
    best_check = {
        'epoch': 0,
        'best': 0,
        'val_acc': 0
    }
    out_file_name = savedir + 'result.txt'
    total_train = len(train_data)
    #train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    #val_loaders=[]
    
    #for i in range(len(val_data)):
    #    val_loader = DataLoader(dataset=val_data[i], batch_size=1, shuffle=True)
    #    val_loaders.append(val_loader)
    #train_dataset, test_dataset =  get_ucf101('Data_256',frames_path ="/home/c3-0/datasets/UCF101_frames/frames_256")
    #"/home/yogesh/Naman/ucf-101/frames-128x128",'/home/c3-0/datasets/UCF101_frames/frames_256'
    #train_sampler = RandomSampler
    #train_loader = DataLoader(
    #    train_dataset,
    #    sampler=train_sampler(train_dataset),
    #    batch_size=1,
    #    num_workers=4,
    #    drop_last=True,
    #    pin_memory=True)
    #state_dict = torch.load("models/train_only_class_with_model_wiht concat_pool5_a3_b3_vcTrue_mixTrue_occlikely0.7_vc2048_lr_1e-07_pascal3d+_pretrainedFalse_epochs_100_occFalse_backbonevgg_0/vc1.pth")
    #model.load_state_dict(state_dict['state_dict'])
    


    # we observed that training the backbone does not make a very big difference but not training saves a lot of memory
    # if the backbone should be trained, then only with very small learning rate e.g. 1e-7
    for param in model.backbone.parameters():
        param.requires_grad = False
    #model.backbone.t_reduction.requires_grad = True
    
    if not vc_flag:
        model.conv1o1.weight.requires_grad = False
    else:
        model.conv1o1.weight.requires_grad = True

    if not mix_flag:
        model.mix_model.requires_grad = False
    else:
        model.mix_model.requires_grad = True

    classification_loss = nn.CrossEntropyLoss()
    cluster_loss = ClusterLoss()

    optimizer = torch.optim.Adagrad(params=filter(lambda param: param.requires_grad, model.parameters()), lr=learning_rate,weight_decay = 0.05)#torch.optim.Adagrad(params=filter(lambda param: param.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=.99)#0.98)

    print('Training')

    for epoch in range(epochs):
        out_file = open(out_file_name, 'a')
        train_loss = 0.0
        correct = 0
        start = time.time()
        model.train()
        model.backbone.eval()
        cnt =0
        for index, data in enumerate(train_loader):
            if index % 500 == 0 and index != 0:
                end = time.time()
        #        print('Epoch{}: {}/{}, Acc: {}, Loss: {} Time:{}'.format(epoch + 1, index, total_train, correct.cpu().item() / index, train_loss.cpu().item() / index, (end-start)))
                start = time.time()

            #input,label,z,_ = data
            input = data["video"]
            label= data["label"]
        
            y = int(label.detach().numpy())
            if y>=101:
                continue
            if cnt > 1:
                break
            input = input.cuda(device_ids[0])
            label = label.cuda(device_ids[0])
            cnt+=1
            
            #input = input.reshape(16,3,224,224)
            output, vgg_feat, like = model(input)
            
            #output = torch.max(output,0,keepdims  = True)[0]
            out = output.argmax(1)
            correct += torch.sum(out == label)
            class_loss = classification_loss(output, label) / output.shape[0]
            
            loss = class_loss
            if alpha != 0:
                clust_loss = cluster_loss(vgg_feat, model.conv1o1.weight) / output.shape[0]
                loss += alpha * clust_loss

            if beta!=0:
                mix_loss = like[0,label[0]]
                loss += -beta *mix_loss
            
            #with torch.autograd.set_detect_anomaly(True):
            loss.backward()
            
            # pseudo batches
            if np.mod(index,batch_size)==0:# and index!=0:
                optimizer.step()
                optimizer.zero_grad()
            #check = {"state_dict":model.backbone.state_dict(),
            #"val_acc":0,
             #        "epoch":0}
            #save_checkpoint(check,"backbone_weights.pth",True)
            train_loss += class_loss.detach() * input.shape[0]

        updated_clutter = update_clutter_model(model,device_ids)
        #updated_clutter = model.get_clutter_model("vmf",60)
        #model.clutter_model = updated_clutter
        model.clutter_model = updated_clutter
        scheduler.step()
    
        train_acc = correct.cpu().item() / cnt #total_train
        train_loss = train_loss.cpu().item() / cnt #total_train
        out_str = 'Epochs: [{}/{}], Train Acc:{}, Train Loss:{}'.format(epoch + 1, epochs, train_acc, train_loss)
        print(out_str)
        out_file.write(out_str)

        # Evaluate Validation images
        import models.video_resnet as models
        model = models.r2plus1d_18(num_classes=400,pretrained=True)
        model = model.cuda()
       #    
        with torch.no_grad():
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            end = time.time()
            predicted_target = {}
            ground_truth_target = {}
            predicted_target_not_softmax = {}
                
            for test_dataset in test_datasets:
                test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                num_workers=1,
                pin_memory=True)

                cnt = 0
                print(cnt)
                with torch.no_grad():
                    for batch_idx, data in enumerate(test_loader):
                        data_time.update(time.time() - end)
                        model.eval()
                                
                        inputs = data["video"]
                        targets= data["label"]
                        inputs = inputs.cuda()
                        targets = targets.cuda()
                        video_name = data["video_name"]
                        #inputs = inputs.reshape(16,3,224,224)
                        
                        
                        outputs = model(inputs)
                        #outputs = torch.max(outputs,0,keepdims = True)[0]
                        out = output.argmax(1)
                        cnt+=1
                        
                        loss = classification_loss(outputs, targets)/ outputs.shape[0]
                        
                        out_prob = F.softmax(outputs, dim=1)
                        out_prob = out_prob.cpu().numpy().tolist()
                        targets = targets.cpu().numpy().tolist()
                        outputs = outputs.cpu().numpy().tolist()
            
                        for iterator in range(len(video_name)):
                            if video_name[iterator] not in predicted_target:
                                predicted_target[video_name[iterator]] = []
                
                            if video_name[iterator] not in predicted_target_not_softmax:
                                predicted_target_not_softmax[video_name[iterator]] = []

                            if video_name[iterator] not in ground_truth_target:
                                ground_truth_target[video_name[iterator]] = []

                            predicted_target[video_name[iterator]].append(out_prob[iterator])
                            predicted_target_not_softmax[video_name[iterator]].append(outputs[iterator])
                            ground_truth_target[video_name[iterator]].append(targets[iterator])
                        #print(len(predicted_target))
                
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
            val_acc = secondary_accuracy
            secondary_accuracy_not_softmax = (pred_values_not_softmax == target_values)*1
            secondary_accuracy_not_softmax = (sum(secondary_accuracy_not_softmax)/len(secondary_accuracy_not_softmax))*100
            print(f'test accuracy before softmax: {secondary_accuracy_not_softmax}')

            if val_acc>best_check['val_acc']:
                print('BEST: {}'.format(val_acc))
                out_file.write('BEST: {}\n'.format(val_acc))
                best_check = {
                       'state_dict': model.state_dict(),
                        'val_acc': val_acc,
                        'epoch': epoch
                        }
            save_checkpoint(best_check, savedir + 'vc' + str(epoch + 1) + '.pth', True)

            print('\n')
            out_file.close()
    #return best_check

def main():
    
    if backbone_type=='vgg':
        if layer=='pool4':
            extractor = models.vgg16(pretrained=True).features[0:24]
        else:
            extractor = models.vgg16(pretrained=True).features
    elif backbone_type=='resnet50' or backbone_type=='resnext':
        extractor = resnet_feature_extractor(backbone_type, layer)
    extractor = Classification_model()
    extractor.cuda(device_ids[0])
    #state_dict = torch.load("backbone_weights.pth")
    #extractor.load_state_dict(state_dict['state_dict'])
    dict_dir = "models/init_vgg/dictionary_vgg/dictionary_finer_mvit_kinetics_prertrained_768.pickle"
    #dict_dir = "CompositionalNets/models/init_vgg/dictionary_vgg/dictionary_pool4.pickle"
    weights = getVmfKernels(dict_dir, device_ids)
    #print(weights)
    bool_load_pretrained_model = False
    if bool_load_pretrained_model:
        pretrained_file = 'PATH TO .PTH FILE HERE'
    else:
        pretrained_file = ''
    occ_likely = []
    for i in range(101): #changed from len()
        # setting the same occlusion likelihood for all classes
        occ_likely.append(likely)

    # load the CompNet initialized with ML and spectral clustering#mix_model_path="CompositionalNets/models/init_vgg/mix_model_vmf_pascal3d+_EM_all"
    mix_models = getCompositionModel(device_ids,mix_model_path,layer,categories_train,compnet_type=compnet_type)
    net = Net(extractor, weights, vMF_kappa, occ_likely, mix_models,
              bool_mixture_bg=bool_mixture_model_bg,compnet_type=compnet_type,num_mixtures=num_mixtures, 
          vc_thresholds=cfg.MODEL.VC_THRESHOLD)
    if bool_load_pretrained_model:
        net.load_state_dict(torch.load(pretrained_file, map_location='cuda:{}'.format(device_ids[0]))['state_dict'])

    net = net.cuda(device_ids[0])

    train_imgs=[]
    train_masks = []
    train_labels = []
    val_imgs = []
    val_labels = []
    val_masks=[]
    
	# get training and validation images
    for occ_level in occ_levels_train:
        if occ_level == 'ZERO':
            occ_types = ['']
            train_fac=0.9
        else:
            occ_types = ['_white', '_noise', '_texture', '']
            train_fac=0.1

    #for occ_type in occ_types:
     #   imgs, labels, masks = getImg('train', categories_train, dataset, data_path, categories, occ_level, occ_type, bool_load_occ_mask=False)
      #  nimgs=len(imgs)
       #     for i in range(nimgs):
        #        if (random.randint(0, nimgs - 1) / nimgs) <= train_fac:
         #           train_imgs.append(imgs[i])
          #          train_labels.append(labels[i])
           #         train_masks.append(masks[i])
            #    elif not bool_train_with_occluders:
             #       val_imgs.append(imgs[i])
              #      val_labels.append(labels[i])
               #     val_masks.append(masks[i])

    #print('Total imgs for train ' + str(len(train_imgs)))
    #print('Total imgs for val ' + str(len(val_imgs)))
    #train_imgset = Imgset(train_imgs,train_masks, train_labels, imgLoader,bool_square_images=False)

#val_imgsets = []
#if val_imgs:
#        val_imgset = Imgset(val_imgs,val_masks, val_labels, imgLoader,bool_square_images=False)
#        val_imgsets.append(val_imgset)

    # write parameter settings into output folder
    train_dataset, test_dataset =  get_ucf101('Data',frames_path ='/home/yogesh/Naman/ucf-101/frames-128x128')
   # get_ucf101('Data',frames_path ='/home/yogesh/Naman/ucf-101/frames-128x128')
    train_sampler = RandomSampler
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler(train_dataset),
        batch_size=1,
        num_workers=4,
        drop_last=True,
        pin_memory=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=1,
        num_workers=4,
        pin_memory=True)

    load_flag = False
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    info = out_dir + 'config.txt'
    config_file = open(info, 'a')
    config_file.write(dataset)
    out_str = 'layer{}_a{}_b{}_vc{}_mix{}_occlikely{}_vc{}_lr{}/'.format(layer,alpha,beta,vc_flag,mix_flag,likely,vc_num,lr)
    config_file.write(out_str)
    out_str = 'Train\nDir: {}, vMF_kappa: {}, alpha: {},beta: {}, likely:{}\n'.format(out_dir, vMF_kappa, alpha,beta,likely)
    config_file.write(out_str)
    #print(out_str)
    out_str = 'pretrain{}_file{}'.format(bool_load_pretrained_model,pretrained_file)
    #print(out_str)
    config_file.write(out_str)
    config_file.close()
    train(model=net, train_data=train_loader, val_data=test_loader, epochs=ncoord_it, batch_size=batch_size,
          learning_rate=lr, savedir=out_dir, alpha=alpha,beta=beta, vc_flag=vc_flag, mix_flag=mix_flag)

if __name__ == "__main__":
    main()
