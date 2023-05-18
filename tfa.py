from data.dataset import Dataset, TestDataset
from frcn.faster_rcnn_vgg16 import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from utils.config import conf
from utils import array_tool as at
from utils.eval_tool import eval_detection_voc
from data.util import read_image
import torch
from torch.utils import data as data_
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parameter_list = [
  'head.classifier.0.weight',
  'head.classifier.0.bias',
  'head.classifier.2.weight',
  'head.classifier.2.bias',
  'head.cls_loc.weight',
  'head.cls_loc.bias',
  'head.score.weight',
  'head.score.bias'
]

def train(split, cos, load_path, shots, model_name):

  faster_rcnn = FasterRCNNVGG16(n_fg_class = len(conf.choosen_class), cos = cos)
  tfa_trainer = FasterRCNNTrainer(faster_rcnn).to(device)

  if load_path != 'none' :
    pre_faster_rcnn = FasterRCNNVGG16(n_fg_class = len(conf.base3_choosen_class), cos = True)
    pre_trainer = FasterRCNNTrainer(pre_faster_rcnn).to(device)

    pre_trainer.load(load_path)

    pretrained_dict = pre_trainer.faster_rcnn.state_dict()
    model_dict = tfa_trainer.faster_rcnn.state_dict()
    new_dict = tfa_trainer.faster_rcnn.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['head.cls_loc.weight','head.cls_loc.bias','head.score.weight','head.score.bias']}
    new_dict.update(pretrained_dict) 

    model_dict = {k: v for k, v in model_dict.items() if k in ['head.cls_loc.weight','head.cls_loc.bias','head.score.weight','head.score.bias']}
    new_dict.update(pretrained_dict) 
    
    tfa_trainer.faster_rcnn.load_state_dict(new_dict)

    # sd = [k for k,v in trainer.faster_rcnn.state_dict().items()]

    # print(sd)

  tfa_params = tfa_trainer.faster_rcnn.named_parameters()
  dict_tfa_params = dict(tfa_params)

  for name,param in dict_tfa_params.items():
    # print(name)
    if ( name not in parameter_list ) :
      param.requires_grad = False

  tfa_trainer.optimizer = tfa_trainer.faster_rcnn.get_optimizer()

  lr_ = conf.lr

  dataset = Dataset(split=split)
  dataloader = data_.DataLoader(dataset,batch_size=1,shuffle=True,num_workers=conf.num_workers)


  for epoch in range(conf.epoch):

    ls = 0.0
    tn = 0

    for ii, (img, bbox_, label_, scale, fname) in tqdm(enumerate(dataloader)):

      scale = at.scalar(scale)
      img, bbox, label = img.to(device).float(), bbox_.to(device), label_.to(device)

      losses = tfa_trainer.train_step(img, bbox, label, scale)

      ls += losses.total_loss.item()
      tn += 1 

    print("total loss for ecpoch ",epoch," is ", ls/tn )
    

    if epoch == 9:
      tfa_trainer.faster_rcnn.scale_lr(conf.lr_decay)
      lr_ = lr_ * conf.lr_decay

  tfa_trainer.save(save_path=conf.weight_path,fn=model_name+'_'+shots+'.pth')
  


def test(split, cos, load_path, shots, model_name):

  faster_rcnn = FasterRCNNVGG16(n_fg_class = len(conf.choosen_class), cos = cos)
  inf_tfa_trainer = FasterRCNNTrainer(faster_rcnn).to(device)
  inf_tfa_trainer.load(load_path = load_path)

  testdataset = TestDataset(split = split)
  testdataloader = data_.DataLoader(testdataset,batch_size=1,shuffle=True,num_workers=conf.num_workers)

  pred_bboxes, pred_labels, pred_scores = list(), list(), list()
  gt_bboxes, gt_labels = list(), list()

  label_acc = 0

  for ii, (imgs, sizes, gt_bboxes_, gt_labels_, fname) in tqdm(enumerate(testdataloader)):

    sizes = [sizes[0][0].item(), sizes[1][0].item()]
    pred_bboxes_, pred_labels_, pred_scores_ = inf_tfa_trainer.faster_rcnn.predict(imgs, [sizes])

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
    # if os.path.exists(my_path) == False : os.makedirs(my_path) 

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
  print('Inside tfa\'s main')