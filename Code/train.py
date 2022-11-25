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



class feature_mvit(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_model = torch.hub.load("facebookresearch/pytorchvideo","mvit_base_16x4",pretrained = True)
        
    def forward(self,x):
        x = self.feature_model.patch_embed(x)
        x = self.feature_model.cls_positional_encoding(x)
        thw = self.feature_model.cls_positional_encoding.patch_embed_shape
        for blck in self.feature_model.blocks:
            x,thw= blck(x,thw)
        out = x[:,:-1,:]
        out = out.reshape(-1,thw[0],768,thw[1],thw[2])
        out_avg = torch.mean(out,dim =1)
        return out_avg


#---------------------
# Training Parameters
#---------------------
alpha = 3  # vc-loss
beta = 3 # mix loss
likely = 0.6 # occlusion likelihood
lr = 1e-2 # learning rate
batch_size = 1 # these are pseudo batches as the aspect ratio of images for CompNets is not square
# Training setup
vc_flag = True # train the vMF kernels
mix_flag = True # train mixture components
ncoord_it = 50 	#number of epochs to train

bool_mixture_model_bg = False #True: use a mixture of background models per pixel, False: use one bg model for whole image
bool_load_pretrained_model = False
bool_train_with_occluders = False


if bool_train_with_occluders:
	occ_levels_train = ['ZERO', 'ONE', 'FIVE', 'NINE']
else:
	occ_levels_train = ['ZERO']

out_dir = model_save_dir + 'train_{}_a{}_b{}_vc{}_mix{}_occlikely{}_vc{}_lr_{}_{}_pretrained{}_epochs_{}_occ{}_backbone{}_{}/'.format(
	layer, alpha,beta, vc_flag, mix_flag, likely, vc_num, lr, dataset, bool_load_pretrained_model,ncoord_it,bool_train_with_occluders,backbone_type,device_ids[0])


def train(model, train_data, val_data, epochs, batch_size, learning_rate, savedir, alpha=3,beta=3, vc_flag=True, mix_flag=False):
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
    train_dataset, test_dataset =  get_ucf101('Data_256',frames_path ='/home/c3-0/datasets/UCF101_frames/frames_256')
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
        batch_size=8,
        num_workers=4,
        pin_memory=True)


    # we observed that training the backbone does not make a very big difference but not training saves a lot of memory
    # if the backbone should be trained, then only with very small learning rate e.g. 1e-7
    for param in model.backbone.parameters():
        param.requires_grad = False

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

    optimizer = torch.optim.Adagrad(params=filter(lambda param: param.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=0.98)

    print('Training')

    for epoch in range(epochs):
        out_file = open(out_file_name, 'a')
        train_loss = 0.0
        correct = 0
        start = time.time()
        model.train()
        model.backbone.eval()
        for index, data in enumerate(train_loader):
            if index % 500 == 0 and index != 0:
                end = time.time()
                print('Epoch{}: {}/{}, Acc: {}, Loss: {} Time:{}'.format(epoch + 1, index, total_train, correct.cpu().item() / index, train_loss.cpu().item() / index, (end-start)))
                start = time.time()

            input,label = data

            input = input.cuda(device_ids[0])
            label = label.cuda(device_ids[0])

            output, vgg_feat, like = model(input)
            
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

            train_loss += loss.detach() * input.shape[0]

        updated_clutter = update_clutter_model(model,device_ids)
        model.clutter_model = updated_clutter
        scheduler.step()
        train_acc = correct.cpu().item() / total_train
        train_loss = train_loss.cpu().item() / total_train
        out_str = 'Epochs: [{}/{}], Train Acc:{}, Train Loss:{}'.format(epoch + 1, epochs, train_acc, train_loss)
        print(out_str)
        out_file.write(out_str)

        # Evaluate Validation images
        model.eval()
        with torch.no_grad():
 #           correct = 0
#			mval_accs=[]
#			for i in range(len(val_loaders)):
#				val_loader = val_loaders[i]
#				correct_local=0
#				total_local = 0
#				val_loss = 0
#				out_pred = torch.zeros(len(val_data[i].images))
#				for index, data in enumerate(val_loader):
#					input,_, label = data
#					input = input.cuda(device_ids[0])
#					label = label.cuda(device_ids[0])
#					output,_,_ = model(input)
#					out = output.argmax(1)
#					out_pred[index] = out
#					correct_local += torch.sum(out == label)
#					total_local += label.shape[0]

#					class_loss = classification_loss(output, label) / output.shape[0]
#					loss = class_loss
#					val_loss += loss.detach() * input.shape[0]
#				correct += correct_local
#				val_acc = correct_local.cpu().item() / total_local
#				val_loss = val_loss.cpu().item() / total_local
#				val_accs.append(val_acc)
#				out_str = 'Epochs: [{}/{}], Val-Set {}, Val Acc:{} Val Loss:{}\n'.format(epoch + 1, epochs,i , val_acc,val_loss)
#				print(out_str)
#				out_file.write(out_str)
#			val_acc = np.mean(val_accs)
#			out_file.write('Epochs: [{}/{}], Val Acc:{}\n'.format(epoch + 1, epochs, val_acc))
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            end = time.time()
            predicted_target = {}
            ground_truth_target = {}
            predicted_target_not_softmax = {}
            with torch.no_grad():
                for batch_idx, (inputs, targets, video_name) in enumerate(test_loader):
                    data_time.update(time.time() - end)
                    model.eval()

                    inputs = inputs.cuda()
                    targets = targets.cuda()
                    outputs,_,_ = model(inputs)
                    out = output.argmax(1)
                    
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
    return best_check

if __name__ == '__main__':

    if backbone_type=='vgg':
        if layer=='pool4':
            extractor = models.vgg16(pretrained=True).features[0:24]
        else:
            extractor = models.vgg16(pretrained=True).features
    elif backbone_type=='resnet50' or backbone_type=='resnext':
        extractor = resnet_feature_extractor(backbone_type, layer)
    extractor = feature_mvit()
    extractor.cuda(device_ids[0]).eval()
    weights = getVmfKernels(dict_dir, device_ids)

    if bool_load_pretrained_model:
        pretrained_file = 'PATH TO .PTH FILE HERE'
    else:
        pretrained_file = ''


    occ_likely = []
    for i in range(101): #changed from len()
        # setting the same occlusion likelihood for all classes
        occ_likely.append(likely)

    # load the CompNet initialized with ML and spectral clustering
    mix_models = getCompositionModel(device_ids,mix_model_path,layer,categories_train,compnet_type=compnet_type)
    net = Net(extractor, weights, vMF_kappa, occ_likely, mix_models, bool_mixture_bg=bool_mixture_model_bg,compnet_type=compnet_type,num_mixtures=num_mixtures, vc_thresholds=cfg.MODEL.VC_THRESHOLD)
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

        for occ_type in occ_types:
            imgs, labels, masks = getImg('train', categories_train, dataset, data_path, categories, occ_level, occ_type, bool_load_occ_mask=False)
            nimgs=len(imgs)
            for i in range(nimgs):
                if (random.randint(0, nimgs - 1) / nimgs) <= train_fac:
                    train_imgs.append(imgs[i])
                    train_labels.append(labels[i])
                    train_masks.append(masks[i])
                elif not bool_train_with_occluders:
                    val_imgs.append(imgs[i])
                    val_labels.append(labels[i])
                    val_masks.append(masks[i])

    print('Total imgs for train ' + str(len(train_imgs)))
    print('Total imgs for val ' + str(len(val_imgs)))
    train_imgset = Imgset(train_imgs,train_masks, train_labels, imgLoader,bool_square_images=False)

    val_imgsets = []
    if val_imgs:
        val_imgset = Imgset(val_imgs,val_masks, val_labels, imgLoader,bool_square_images=False)
        val_imgsets.append(val_imgset)

    # write parameter settings into output folder
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
    print(out_str)
    out_str = 'pretrain{}_file{}'.format(bool_load_pretrained_model,pretrained_file)
    print(out_str)
    config_file.write(out_str)
    config_file.close()

    train(model=net, train_data=train_imgset, val_data=val_imgsets, epochs=ncoord_it, batch_size=batch_size,
          learning_rate=lr, savedir=out_dir, alpha=alpha,beta=beta, vc_flag=vc_flag, mix_flag=mix_flag)


