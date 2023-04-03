from util import read_image

import sys
# sys.path is a list of absolute path strings
sys.path.append('/content/drive/MyDrive/FSBD')
from utils.config import conf
sys.path.append('/content/drive/MyDrive/FSBD/data')

import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image
import cv2

class TrBboxDataset:
  def __init__(self, split='alpha'):

    data_dir = conf.data_dir
    choosen_class = conf.choosen_class

    self.files = []
    
    if split == 'alpha':
      with open(data_dir,'r') as f1:
        for line in f1:
          self.files.append(line)
    elif split == 'beta':

      fcnt = 0
      
      with open(data_dir,'r') as f1:
        for line in f1:
          self.files.append(line)

          fcnt += 1
          if fcnt == 1: break
    
    self.data_dir = data_dir
    self.label_names = tuple(choosen_class)

  def __len__(self):
    return len(self.files)
  
  def __getitem__(self,i):
    
    fname = self.files[i]
    fname = fname.split('.')[0]
      
    # print(fname)

    iname = fname + conf.image_extension 
    aname = fname + '.txt' 

    img = img = read_image(os.path.join(conf.image_path,iname), color=True)

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
        label.append(self.label_names.index(cls))
    
    bbox = np.stack(bbox).astype(np.float32)
    label = np.stack(label).astype(np.int32)  

    return img, bbox, label, fname
    



if __name__ == '__main__':

  print('Inside tr_dataset.py')

  dd = conf.one_shot_tn

  db = TrBboxDataset()
  print(db.__len__())

  img,bbox,label,fname = db.__getitem__(0)
  img /= 255
  img = img.transpose((1,2,0))

  for bb in bbox:
    cv2.rectangle(img,(int(bb[1]),int(bb[0])),(int(bb[3]),int(bb[2])),(0,0,255),2)
  plt.imshow(img)
  plt.savefig('abcd.png')