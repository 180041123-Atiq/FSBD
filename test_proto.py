from frcn.faster_rcnn_vgg16 import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.dataset import TestDataset, Dataset
from data.util import read_image
from utils.config import conf
from utils.eval_tool import eval_detection_voc
from utils import array_tool as at
from data.dataset import CDDataset
from prototypical import ProtoNet

import torch
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import os
from torchvision.transforms import ToPILImage
import numpy as np
from torch.utils import data as data_
from torchvision.models import resnet18,resnet101
from torchvision import transforms
from torch import nn
import copy
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tdevice = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

image_size = 28


def test(split, cos, load_path, shots, model_name):
  
  faster_rcnn = FasterRCNNVGG16(n_fg_class = len(conf.choosen_class), cos = cos)
  inf_trainer = FasterRCNNTrainer(faster_rcnn).to(device)
  inf_trainer.load(load_path = load_path)

  dataset = Dataset(split=split)
  dataloader = data_.DataLoader(dataset,batch_size=1,shuffle=True,num_workers=conf.num_workers)

  testdataset = TestDataset(split=split)
  testloader = data_.DataLoader(testdataset,batch_size=1,shuffle=True,num_workers=conf.num_workers)

  support_images = []
  support_labels = []
  query_images = []

  resize_obj = transforms.Resize([int(image_size * 1.15), int(image_size * 1.15)])
  totensor_obj = transforms.ToTensor()

  
  for ii, (img, bbox_, label_, scale, fname) in tqdm(enumerate(dataloader)):
    img, bbox, label = img.to(device).float(), bbox_.to(device), label_.to(device)
    nimg = at.tonumpy(img[0]).astype(float)
    bbox = at.tonumpy(bbox[0]).astype(int)
    label = at.tonumpy(label).astype(int)
    nimg = nimg.transpose((1,2,0))
    for bb in bbox:
      tmp = totensor_obj(nimg[bb[0]-1:bb[2],bb[1]-1:bb[3],0:3])
      tmp = resize_obj(tmp)
      support_images.append(tmp)
    for ll in label:
      support_labels.append(ll[0])
    # plt.imshow(nimg)
    # plt.savefig('proto.png')

  pred_bboxes, pred_labels, pred_scores = list(), list(), list()
  gt_bboxes, gt_labels = list(), list()

  zeroDimensionImg = []
  label_acc = 0

  for ii, (imgs, sizes, gt_bboxes_, gt_labels_, fname) in tqdm(enumerate(testloader)):

    sizes = [sizes[0][0].item(), sizes[1][0].item()]
    pred_bboxes_, pred_labels_, pred_scores_ = inf_trainer.faster_rcnn.predict(imgs, [sizes])

    nimg = at.tonumpy(imgs[0]).astype(float)
    nimg = nimg.transpose((1,2,0))
    bbox = at.tonumpy(pred_bboxes_[0]).astype(int)
    # pred_score = at.tonumpy(pred_scores_[0])
    # print('qimg shape : ',nimg.shape)
    # print('bbox shape : ',bbox.shape)
    # print('pred_score shape : ',pred_score.shape)

    if gt_labels_[0][0].item() in pred_labels_[0] : label_acc += 1

    zdi = []

    for idx,(bb) in enumerate(bbox):
      tmp = totensor_obj(nimg[bb[0]-1:bb[2],bb[1]-1:bb[3],0:3])

      H,W,C = tmp.shape
      if H <= 0 or W <= 0 or C <= 0 :
        zdi.append(idx)
      else :  
        tmp = resize_obj(tmp)
        query_images.append(tmp)

    gt_bboxes += list(gt_bboxes_.numpy())
    gt_labels += list(gt_labels_.numpy())
  
    pred_bboxes += pred_bboxes_
    pred_labels += pred_labels_
    pred_scores += pred_scores_

    zeroDimensionImg.append(zdi)

    if ii == 10000: break

  ##Attaching Prototypical Network
  support_images = [t.numpy() for t in support_images]
  support_images = np.array(support_images,dtype='float64')
  support_labels = np.array(support_labels,dtype='int')

  query_images = [t.numpy() for t in query_images]
  query_images = np.array(query_images,dtype='float64')

  query_images = at.totensor(query_images)
  support_images = at.totensor(support_images)
  support_labels = at.totensor(support_labels).int()

  convolutional_network = resnet101(pretrained=True)
  num_features = convolutional_network.fc.in_features
  num_classes = 43
  convolutional_network.fc = nn.Linear(num_features,num_classes)
  convolutional_network.load_state_dict(torch.load('/content/drive/MyDrive/simplest-faster-rcnn/proto_weight/resnet101_gtrsb.pth',map_location=tdevice))
  convolutional_network.fc = nn.Flatten()


  proto_model = ProtoNet(convolutional_network).to(device)
  
  proto_model.eval()
  cscores = proto_model(
      support_images.to(device).float(),
      support_labels.to(device),
      query_images.to(device).float(),
  )

  ori_pred_scores = copy.deepcopy(pred_scores)
  al = 0.001
  yy = []
  xx = []

  while al <= 1.00 :
    
    pred_scores = copy.deepcopy(ori_pred_scores)

    another_idx_cs = 0

    for ix in range(len(pred_scores)):
      for jx in range(len(pred_scores[ix])):
        if jx not in zeroDimensionImg[ix]:
          pl = pred_labels[ix][jx]
          try :
            if pl < cscores.shape[1]:
              cs = cscores[another_idx_cs][pl]
          except :
            print("Error Operation Aborted : ",cscores.shape,another_idx_cs,pl)
          
          another_idx_cs += 1

          ns = ( (1-al) * pred_scores[ix][jx]) + ( al * cs )
          pred_scores[ix][jx] = ns

    yy.append(eval_detection_voc(pred_bboxes, pred_labels, pred_scores,gt_bboxes, gt_labels)['map'])
    xx.append(al)
    al += 0.001
  
  plt.plot(xx, yy)
  
  # naming the x axis
  plt.xlabel('value for alpha')
  # naming the y axis
  plt.ylabel('map with proto')
    
  # giving a title to my graph
  plt.title('Plot to find best value of alpha')
    
  # function to show the plot
  plt.savefig('myplot2.png')
  plt.show()

  result = eval_detection_voc(pred_bboxes, pred_labels, pred_scores,gt_bboxes, gt_labels)
  ori_result = eval_detection_voc(pred_bboxes, pred_labels, ori_pred_scores,gt_bboxes, gt_labels)

  xx,yy,ores = xx,yy,ori_result

  print(model_name+'_'+shots)
  print("Best map ", math.floor(max(yy)*100)/100," at alpha value ",xx[yy.index(max(yy))])
  print("Without proto map is ", math.floor(ores['map']*100)/100)
  print("Best map ", max(yy)," at alpha value ",xx[yy.index(max(yy))])
  print("Without proto map is ", ores['map'])
  print("Label Accuracy : ",label_acc)

if __name__ == '__main__':
  # test()
  # test_proto()
  print("Inside test_proto's main")