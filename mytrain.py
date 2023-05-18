import os
# import ipdb
import matplotlib
from tqdm import tqdm

from mydata.mydataset import Dataset, TestDataset, inverse_normalize
from myutils.myconfig import opt
from model import FasterRCNNVGG16
from torch.utils import data as data_
from mytrainer import FasterRCNNTrainer
from myutils import array_tool as at
# from myutils.vis_tool import visdom_bbox
from myutils.eval_tool import eval_detection_voc

import matplotlib.pyplot as plt
from prototypical import ProtoNet
import numpy as np
import torch
from torchvision.models import resnet18
from torchvision import transforms
from torch import nn
import copy

image_size = 28
ALPHA = 0.13

def test(dataloader,support_proto, faster_rcnn,test_num=10000):

  isProto = True

  support_images = []
  support_labels = []
  query_images = []

  resize_obj = transforms.Resize([int(image_size * 1.15), int(image_size * 1.15)])
  totensor_obj = transforms.ToTensor()

  if isProto :
    for ii, (img, bbox_, label_, scale) in tqdm(enumerate(support_proto)):
      img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
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

  for ii, (imgs, sizes, gt_bboxes_, gt_labels_) in tqdm(enumerate(dataloader)):

    sizes = [sizes[0][0].item(), sizes[1][0].item()]
    pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])

    nimg = at.tonumpy(imgs[0]).astype(float)
    nimg = nimg.transpose((1,2,0))
    bbox = at.tonumpy(pred_bboxes_[0]).astype(int)
    # pred_score = at.tonumpy(pred_scores_[0])
    # print('qimg shape : ',nimg.shape)
    # print('bbox shape : ',bbox.shape)
    # print('pred_score shape : ',pred_score.shape)

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

    if ii == test_num: break

  ##Attaching Prototypical Network
  support_images = [t.numpy() for t in support_images]
  support_images = np.array(support_images,dtype='float64')
  support_labels = np.array(support_labels,dtype='int')

  query_images = [t.numpy() for t in query_images]
  query_images = np.array(query_images,dtype='float64')

  # print(support_images.dtype)
  # print(query_images.dtype)
  # print(support_labels.dtype)

  # print(support_images.shape)
  # print(query_images.shape)
  # print(support_labels.shape)

  query_images = at.totensor(query_images)
  support_images = at.totensor(support_images)
  support_labels = at.totensor(support_labels).int()

  # print(support_images.size())
  # print(query_images.size())
  # print(support_labels.size())

  # print(support_labels)

  convolutional_network = resnet18(pretrained=True)
  convolutional_network.fc = nn.Flatten()
  # print(convolutional_network)

  proto_model = ProtoNet(convolutional_network).cuda()
  
  proto_model.eval()
  cscores = proto_model(
      support_images.cuda().float(),
      support_labels.cuda(),
      query_images.cuda().float(),
  )

  # print('cscores shape : ',cscores.shape)

  # nps = np.array(pred_scores,dtype='float64')
  # npl = np.array(pred_labels,dtype='int')
  # nzdi = np.array(zeroDimensionImg,dtype='int')
  # print('nps shape : ',nps.shape)
  # print('zeroDimensionImg size : ',nzdi.shape)
  # print('npl[0][0] : ',npl[0][0])

  ori_pred_scores = copy.deepcopy(pred_scores)
  al = 0.01
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
    al += 0.01
  
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

  return xx,yy,ori_result


##testing on pretrained model

def load_test():
  faster_rcnn = FasterRCNNVGG16()
  mytrainer = FasterRCNNTrainer(faster_rcnn).cuda()
  lp = '/content/drive/MyDrive/simplest-faster-rcnn/weights/kaggle_13.pth'
  mytrainer.load(load_path = lp)
  print("Loaded pretrained model from %s" % lp)

  dataset = Dataset(opt)
  print('load data')
  dataloader = data_.DataLoader(dataset,batch_size=1,shuffle=True,num_workers=opt.num_workers)

  testset = TestDataset(opt)
  test_dataloader = data_.DataLoader(testset,batch_size=1,num_workers=opt.test_num_workers,shuffle=False,pin_memory=True)

  xx,yy,ores = test(test_dataloader,dataloader,faster_rcnn,test_num=10000)

  print("Best map ",max(yy)," at alpha value ",xx[yy.index(max(yy))])
  print("Without proto map is ",ores['map'])



def train():
  
  dataset = Dataset(opt)
  print('load data')
  dataloader = data_.DataLoader(dataset,batch_size=1,shuffle=True,num_workers=opt.num_workers)

  # testset = TestDataset(opt)
  # test_dataloader = data_.DataLoader(testset,batch_size=1,num_workers=opt.test_num_workers,shuffle=False,pin_memory=True)

  faster_rcnn = FasterRCNNVGG16()
  print('model construct completed')

  mytrainer = FasterRCNNTrainer(faster_rcnn).cuda()

  # if opt.load_path:
  #   mytrainer.load(opt.load_path)
  #   print('load pretrained model from %s' % opt.load_path)
  
  # mytrainer.vis.text(dataset.db.label_names, win='labels')

  mytrainer.load(load_path = "/content/drive/MyDrive/simplest-faster-rcnn/weights/lisa_50.pth")

  best_map = 0
  lr_ = opt.lr

  for epoch in range(opt.epoch):

    mytrainer.reset_meters()

    for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):

      scale = at.scalar(scale)
      img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()

      mytrainer.train_step(img, bbox, label, scale)

    print("total loss for ecpoch ",epoch," is ",mytrainer.get_meter_data()["total_loss"])
    

    if epoch == 9:
      mytrainer.faster_rcnn.scale_lr(opt.lr_decay)
      lr_ = lr_ * opt.lr_decay

  
  # print("Going For Evaluation : ")
  # ores,res = test(test_dataloader,dataloader, faster_rcnn, test_num=opt.test_num)
  mytrainer.save(save_path='/content/drive/MyDrive/simplest-faster-rcnn/weights',fn='lisa_indo_50.pth')
  
  # print('Map is : ',res['map'])
  # print('Ori Map is : ',ores['map'])
      

if __name__ == '__main__':
  # train()
  load_test()
  # print(opt.debug_file,os.path.exists(opt.debug_file))