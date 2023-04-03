from data.dataset import Dataset
from utils.config import conf
from frcn.faster_rcnn_vgg16 import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.models import resnet18
from torchvision import transforms
from torch import nn
import copy
import os
import matplotlib
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(data_dir, image_path, annot_path, choosen_class, pre_choosen_class, load_path = 'none'):
  conf.image_path = image_path
  conf.annot_path = annot_path
  dataset = Dataset(data_dir, choosen_class, split='beta')

  print('load data for base1')
  dataloader = data_.DataLoader(dataset,batch_size=1,shuffle=True,num_workers=conf.num_workers)

  faster_rcnn = FasterRCNNVGG16(n_fg_class = len(choosen_class), cos = True)
  print('model construct completed')

  trainer = FasterRCNNTrainer(faster_rcnn).to(device)

  if load_path != 'none' :
    pre_faster_rcnn = FasterRCNNVGG16(n_fg_class = len(pre_choosen_class), cos = True)
    pre_trainer = FasterRCNNTrainer(pre_faster_rcnn).to(device)

    pre_trainer.load(load_path)

    pretrained_dict = pre_trainer.faster_rcnn.state_dict()
    model_dict = trainer.faster_rcnn.state_dict()
    new_dict = trainer.faster_rcnn.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['head.cls_loc.weight','head.cls_loc.bias','head.score.weight','head.score.bias']}
    new_dict.update(pretrained_dict) 

    model_dict = {k: v for k, v in model_dict.items() if k in ['head.cls_loc.weight','head.cls_loc.bias','head.score.weight','head.score.bias']}
    new_dict.update(pretrained_dict) 
    
    trainer.faster_rcnn.load_state_dict(new_dict)

    sd = [k for k,v in trainer.faster_rcnn.state_dict().items()]

    print(sd)


  best_map = 0
  lr_ = conf.lr

  

  for epoch in range(conf.epoch):

    ls = 0.0
    tn = 0

    for ii, (img, bbox_, label_, scale, fname) in tqdm(enumerate(dataloader)):

      scale = at.scalar(scale)
      img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()

      losses = trainer.train_step(img, bbox, label, scale)

      ls += losses.total_loss.item()
      tn += 1 

    print("total loss for ecpoch ",epoch," is ", ls/tn )
    

    if epoch == 9:
      trainer.faster_rcnn.scale_lr(conf.lr_decay)
      lr_ = lr_ * conf.lr_decay


  trainer.save(save_path=conf.weight_path,fn='base2_base1_50.pth')
      

if __name__ == '__main__':
  train(data_dir = conf.base2, image_path = conf.base2_image, annot_path = conf.base2_annot,
  choosen_class = conf.base2_choosen_class, pre_choosen_class = conf.base1_choosen_class,
   load_path = '/content/drive/MyDrive/FSBD/weights/base1_50.pth' )