from data.dataset import FSBDDataset4x, TestDataset
from frcn.faster_rcnn_vgg16 import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from utils.config import conf
from data.util import read_image
from utils import array_tool as at
from utils.eval_tool import eval_detection_voc
import torch
from torch.utils import data as data_
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'



def train(split, cos, load_path, shots, model_name):


  faster_rcnn = FasterRCNNVGG16(n_fg_class = len(conf.choosen_class), cos = cos)
  fsbd_trainer = FasterRCNNTrainer(faster_rcnn).to(device)

  if load_path != 'none' :
    fsbd_trainer.load(load_path = load_path)


  lr_ = conf.lr

  dataset = FSBDDataset4x(split=split)
  dataloader = data_.DataLoader(dataset,batch_size=1,shuffle=True,num_workers=conf.num_workers)


  for epoch in range(conf.epoch):

    ls = 0.0
    tn = 0
    flag = 0

    for ii, (img, bbox_, label_, scale, fname) in tqdm(enumerate(dataloader)):

      scale = at.scalar(scale)
      img, bbox, label = img.to(device).float(), bbox_.to(device), label_.to(device)

      losses = fsbd_trainer.train_step(img, bbox, label, scale)

      if flag == 0 and np.isnan(losses.total_loss.item()) :
        flag = 1
        print(img)

      ls += losses.total_loss.item()
      tn += 1 

    print("total loss for ecpoch ",epoch," is ", ls/tn )
    

    if epoch == 9:
      fsbd_trainer.faster_rcnn.scale_lr(conf.lr_decay)
      lr_ = lr_ * conf.lr_decay

    if epoch % 10 == 0 : fsbd_trainer.save(save_path=conf.weight_path,fn=model_name+'_'+shots+'.pth')
    
  fsbd_trainer.save(save_path=conf.weight_path,fn=model_name+'_'+shots+'.pth')


def test(split, cos, load_path, shots, model_name):

  faster_rcnn = FasterRCNNVGG16(n_fg_class = len(conf.choosen_class), cos = cos)
  inf_frcnft_trainer = FasterRCNNTrainer(faster_rcnn).to(device)
  inf_frcnft_trainer.load(load_path = load_path)

  testdataset = TestDataset(split = split)
  testdataloader = data_.DataLoader(testdataset,batch_size=1,shuffle=True,num_workers=conf.num_workers)

  pred_bboxes, pred_labels, pred_scores = list(), list(), list()
  gt_bboxes, gt_labels = list(), list()

  label_acc = 0

  for ii, (imgs, sizes, gt_bboxes_, gt_labels_, fname) in tqdm(enumerate(testdataloader)):

    sizes = [sizes[0][0].item(), sizes[1][0].item()]
    pred_bboxes_, pred_labels_, pred_scores_ = inf_frcnft_trainer.faster_rcnn.predict(imgs, [sizes])

    gtboxes = gt_bboxes_[0]
    boxes = pred_bboxes_[0]
    scores = pred_scores_[0]
    labels = pred_labels_[0]

    if gt_labels_[0][0].item() in pred_labels_[0] : label_acc += 1

    img = read_image(os.path.join(conf.image_path,fname[0] + conf.image_extension), color=True)
    img /= 255
    img = img.transpose((1,2,0))

    for box in boxes:
      # print(box)
      img = cv2.rectangle(img,(int(box[1]),int(box[0])),(int(box[3]),int(box[2])),(0, 0, 255),6)
    
    # my_file = str(ii) + conf.image_extension
    # my_path = os.path.join(conf.test_predictions_path,model_name+'_'+shots)
    # if os.path.exists(my_path) == False :os.makedirs(my_path) 

    # plt.imshow(img)
    # plt.savefig(os.path.join(my_path, my_file))

    gt_bboxes += list(gt_bboxes_.numpy())
    gt_labels += list(gt_labels_.numpy())
  
    pred_bboxes += pred_bboxes_
    pred_labels += pred_labels_
    pred_scores += pred_scores_
  
  result = eval_detection_voc(pred_bboxes, pred_labels, pred_scores,gt_bboxes, gt_labels)

  print('mAP for '+model_name+' '+shots+' : ',result['map'])
  print("label acc : ",label_acc)




if __name__ == '__main__':
  # train()
  print('Inside FSBD\'s main')