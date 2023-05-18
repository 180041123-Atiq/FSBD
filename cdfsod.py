from data.dataset import CDDataset, preprocess, TestDataset
from data.util import read_image
from frcn.faster_rcnn_vgg16 import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from utils.config import conf
from utils import array_tool as at
from frcn.utils.bbox_tools import bbox_iou
from utils.eval_tool import eval_detection_voc
import torch
from tqdm import tqdm
import os
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

LAMDA = conf.cdfsod_lamda
ALPHA = conf.cdfsod_alpha

faster_rcnn = FasterRCNNVGG16(n_fg_class = len(conf.choosen_class), cos = conf.cdfsod_cos)
student_trainer = FasterRCNNTrainer(faster_rcnn).to(device)
teacher_trainer = FasterRCNNTrainer(faster_rcnn).to(device)

if conf.cdfsod_load_path != 'none' :
  student_trainer.load(conf.cdfsod_load_path)
  teacher_trainer.load(conf.cdfsod_load_path)

train_teacher_dataset = CDDataset(conf.cdfsod_split, transforms = False)
train_student_dataset = CDDataset(conf.cdfsod_split, transforms = True)

train_teacher_loader = DataLoader(
    train_teacher_dataset,
    batch_size = 1,
    shuffle = True,
    num_workers = conf.num_workers,
)
train_student_loader = DataLoader(
    train_student_dataset,
    batch_size = 1,
    shuffle = True,
    num_workers = conf.num_workers,
)

def biou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def train(split, cos, load_path, shots, model_name, alpha, lamda, explicit = False):


  lr_ = conf.lr

  for epoch in range(conf.epoch):

    ls = 0.0
    tn = 0

    for ii, (img, bbox_, label_, scale, fname) in tqdm(enumerate(train_student_loader)):

      scale = at.scalar(scale)
      img, bbox, label = img.to(device).float(), bbox_.to(device), label_.to(device)

      student_trainer.optimizer.zero_grad()
      losses = student_trainer.forward(img, bbox, label, scale)
      
      Ls = losses.total_loss.item()

      psuedo_labels = produce_psuedo_labels(fname[0])

      # print(psuedo_labels)

      Ld = produce_distillation_loss(psuedo_labels,img, bbox, label,scale)

      # print("Ld : ",Ld)

      L = Ls + LAMDA * Ld

      # print("L : ",L)
      # print("Ls : ",Ls)
      # print("Ld : ",Ld)

      ls += L
      tn += 1 

      with torch.no_grad():
        losses.total_loss.set_(torch.Tensor([L]).to(device)[0])

      # student_trainer.optimizer.zero_grad()
      losses.total_loss.backward()
      student_trainer.optimizer.step()

      update_parameters()

    print("total loss for ecpoch ",epoch," is ",ls/tn)
    

    if epoch == 9:
      student_trainer.faster_rcnn.scale_lr(conf.lr_decay)
      lr_ = lr_ * conf.lr_decay

  if explicit == False : 
    teacher_trainer.save(save_path=conf.weight_path,fn=model_name+'_'+shots+'.pth')
  elif explicit == True :
    teacher_trainer.save(save_path=conf.weight_path,fn=model_name+'_'+shots+'_'+str(alpha)+'_'+str(lamda)+'.pth')

def update_parameters():
  with torch.no_grad():
    for tparam,sparam in zip(teacher_trainer.faster_rcnn.parameters(),student_trainer.faster_rcnn.parameters()):
      tparam[:2] = ALPHA * tparam[:2] + (1-ALPHA) * sparam[:2]

def produce_distillation_loss(psuedo_labels,img,bbox,label,scale):

  boxes = psuedo_labels['bboxes'][0] 
  scores = psuedo_labels['scores'][0]

  mx = -1e9
  nidx = -1

  # print("boxes : ",boxes)
  # print("scores : ",scores)

  for ii in range(len(boxes)):
    iou = biou(at.tonumpy(bbox[0][0]),boxes[ii])
    
    if mx < iou :
      mx = iou
      nidx = ii

  # print("bbox size before convert : ",bbox.size())

  if len(boxes) > 0 :
    # print("Got some psuedo labels")
    # print("pseudo box : ",boxes[nidx])
    # print("gt box : ",bbox)
    # print("len(boxes) : ",len(boxes))
    vbox = np.array(boxes[nidx])
    vbox = vbox[None]
    vbox = vbox[None]
    bbox = at.totensor(vbox)
  
  # print("bbox size after convert : ",bbox.size())

  Ld = 0.0
  student_trainer.optimizer.zero_grad()
  losses = student_trainer.forward(img, bbox, label, scale)
  Ld = losses.total_loss.item()

  return Ld

def getitem_pseudo_labels(fname):

  iname = fname + conf.image_extension 
  aname = fname + '.txt'

  # print("HAHA: ", iname, aname, repo_dir) 

  img = read_image(os.path.join(conf.image_path,iname), color=True)

  ic,ih,iw= img.shape

  bbox = list()
  label = list()

  with open(os.path.join(conf.annot_path,aname)) as lines:
    
    for line in lines:
      cls,cx,cy,w,h = line.split(' ')[0],float(line.split(' ')[1]),float(line.split(' ')[2]),float(line.split(' ')[3]),float(line.split(' ')[4])
      lx = (cx*iw)-((iw*w)/2.0)
      ly = (cy*ih)-((ih*h)/2.0)
      rx = (cx*iw)+((iw*w)/2.0)
      ry = (cy*ih)+((ih*h)/2.0)
      bbox.append([ly,lx,ry,rx])
      label.append(conf.choosen_class.index(cls))
  
  bbox = np.stack(bbox).astype(np.float32)
  label = np.stack(label).astype(np.int32)  

  return img,bbox,label

def produce_psuedo_labels(fname):
  ori_img,bbox,label = getitem_pseudo_labels(fname)
  img = preprocess(ori_img)
  img = img[None]

  sizes = ori_img.shape[1:]
  sizes = [sizes[0], sizes[1]]

  pred_bboxes_, pred_labels_, pred_scores_ = teacher_trainer.faster_rcnn.predict(img, [sizes])

  psuedo_labels = {}

  psuedo_labels['bboxes'] = pred_bboxes_
  psuedo_labels['labels'] = pred_labels_
  psuedo_labels['scores'] = pred_scores_

  return  psuedo_labels


def test(split, cos, load_path, shots, model_name):

  faster_rcnn = FasterRCNNVGG16(n_fg_class = len(conf.choosen_class), cos = cos)
  inf_cdfsod_trainer = FasterRCNNTrainer(faster_rcnn).to(device)
  inf_cdfsod_trainer.load(load_path = load_path)

  testdataset = TestDataset(split = split)
  testdataloader = DataLoader(testdataset,batch_size=1,shuffle=True,num_workers=conf.num_workers)

  pred_bboxes, pred_labels, pred_scores = list(), list(), list()
  gt_bboxes, gt_labels = list(), list()

  label_acc = 0

  for ii, (imgs, sizes, gt_bboxes_, gt_labels_, fname) in tqdm(enumerate(testdataloader)):

    sizes = [sizes[0][0].item(), sizes[1][0].item()]
    pred_bboxes_, pred_labels_, pred_scores_ = inf_cdfsod_trainer.faster_rcnn.predict(imgs, [sizes])

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


if __name__ == '__main__' :
  
  # train()
  print("Inside cdfsod's main")