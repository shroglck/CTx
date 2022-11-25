import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2 as cv
from Code.helpers import getImg, imgLoader, Imgset, myresize
from torch.utils.data import DataLoader
import numpy as np
import math
import torch
from Data_256.UCF101 import get_ucf101
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import argparse
from vc_cluster_fine import main_vc
from simmat_finer import main_sim
from mxi_model_learn_finer import main_mix

def main():
    parser = argparse.ArgumentParser(description='PyTorch Classification Testing')
    parser.add_argument('--Dataset',default="UCF101",type = str,help = "Dataset")
    parser.add_argument('--train_vMF',default=True,type = bool,help = "train vmf kernel")
    parser.add_argument('--train_simmat',default=True,type = bool,help = "train similarity matrix")
    parser.add_argument('--train_mixture_model',default=True,type = bool,help = "train mixture_moodel")
    parser.add_argument('--train',default = True,type = bool,help = "train model")
    args = parser.parse_args()
    if args.train_vMF:
        main_vc(args.Dataset)
    if args.train_simmat:
        main_sim(args.Dataset)
    if args.train_mixture_model:
        main_mix(args.Dataset)
    if args.train:
        if args.Dataset == "UCF101":
            from compose_train_2 import main as train
        elif args.Dataset == "Kinetics":
            from compose_train import main as train
        train()

if __name__ == "__main__":
    main()
        
    
    
