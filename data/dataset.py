import sys
from skimage.transform.radon_transform import convert_to_float
sys.path.append('/content/drive/MyDrive/FSBD')
from utils.config import conf
sys.path.append('/content/drive/MyDrive/FSBD/data')

import util
from cutout import Cutout
from tr_dataset import TrBboxDataset

import torch as t
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as T
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image,ImageDraw
import cv2



def inverse_normalize(img):
  if conf.caffe_pretrain:
      img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
      return img[::-1, :, :]
  # approximate un-normalize for visualize
  return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def pytorch_normalze(img):
    
  normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
  img = normalize(t.from_numpy(img))
  return img.numpy()


def caffe_normalize(img):
    
  img = img[[2, 1, 0], :, :]  # RGB-BGR
  img = img * 255
  mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
  img = (img - mean).astype(np.float32, copy=True)
  return img


def preprocess(img, min_size=600, max_size=1000):

  C, H, W = img.shape
  scale1 = min_size / min(H, W)
  scale2 = max_size / max(H, W)
  scale = min(scale1, scale2)
  img = img / 255.
  img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
  # both the longer and shorter should be less than
  # max_size and min_size
  if conf.caffe_pretrain:
      normalize = caffe_normalize
  else:
      normalize = pytorch_normalze
  return normalize(img)


class Transform(object):

  def __init__(self, min_size=600, max_size=1000):
    self.min_size = min_size
    self.max_size = max_size

  def __call__(self, in_data):
    img, bbox, label = in_data
    _, H, W = img.shape
    img = preprocess(img, self.min_size, self.max_size)
    _, o_H, o_W = img.shape
    scale = o_H / H
    bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

    # horizontally flip
    img, params = util.random_flip(
        img, x_random=True, return_param=True)
    bbox = util.flip_bbox(
        bbox, (o_H, o_W), x_flip=params['x_flip'])

    return img, bbox, label, scale



class Dataset:
  
  def __init__(self, split = 'beta'):
    self.db = TrBboxDataset( split = split)
    self.tsf = Transform(conf.min_size, conf.max_size)

  def __getitem__(self, idx):
    ori_img, bbox, label, fname = self.db.__getitem__(idx)
    
    img, bbox, label, scale = self.tsf((ori_img, bbox, label))

    return img.copy(), bbox.copy(), label.copy(), scale, fname


  def __len__(self):
    return len(self.db)



class TestDataset:
  def __init__(self,split = 'beta'):
      self.db = TrBboxDataset(split = split)

  def __getitem__(self, idx):
      ori_img, bbox, label, fname = self.db.__getitem__(idx)
      img = preprocess(ori_img)
      return img, ori_img.shape[1:], bbox, label, fname

  def __len__(self):
      return len(self.db)



class CDDataset:
  
  def __init__(self, split = 'beta', transforms = False):
    self.db = TrBboxDataset(split = split)
    self.tsf = Transform(conf.min_size, conf.max_size)
    self.transforms = transforms

  def __getitem__(self, idx):
    ori_img, bbox, label, fname = self.db.__getitem__(idx)
    
    img, bbox, label, scale = self.tsf((ori_img, bbox, label))

    if self.transforms:

      if randint(0, 1000) % 2 == 1 or True :

        img = img.transpose((1,2,0))
        objtotensor = T.ToTensor()
        objgaus = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
        objjit = T.ColorJitter(brightness=.5, hue=.3)
        objco = Cutout(1,8)
        img = objtotensor(img.copy())
        img = objgaus(img)
        img = objjit(img)

        img = objco(img)

        img = img.numpy()

    return img.copy(), bbox.copy(), label.copy(), scale, fname

  def __len__(self):
    return len(self.db)


class FSBDDataset2x:

  def __init__(self, split = 'beta'):
    self.db = TrBboxDataset( split = split)
    self.tsf = Transform(conf.min_size, conf.max_size)

  def __getitem__(self, idx):
    real_idx = idx//2
    ori_img, bbox, label, fname = self.db.__getitem__(real_idx)
    
    img, bbox, label, scale = self.tsf((ori_img, bbox, label))


    img = img.transpose((1,2,0))
    objtotensor = T.ToTensor()
    objgaus = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
    objjit = T.ColorJitter(brightness=.5, hue=.3)
    objco = Cutout(1,8)
    img = objtotensor(img.copy())

    if idx % 2 != 0 :
      img = objgaus(img)
      img = objjit(img)
      img = objco(img)

    img = img.numpy()

    return img.copy(), bbox.copy(), label.copy(), scale, fname

  def __len__(self):
    return (len(self.db)*2)


class FSBDDataset4x:

  def __init__(self, split = 'beta'):
    self.db = TrBboxDataset( split = split)
    self.tsf = Transform(conf.min_size, conf.max_size)

  def __getitem__(self, idx):
    real_idx = idx//4
    ori_img, bbox, label, fname = self.db.__getitem__(real_idx)
    
    img, bbox, label, scale = self.tsf((ori_img, bbox, label))


    img = img.transpose((1,2,0))
    objtotensor = T.ToTensor()
    objgaus = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
    objjit = T.ColorJitter(brightness=.5, hue=.3)
    objco = Cutout(1,8)
    img = objtotensor(img.copy())

    if idx % 4 != 0 :
      img = objgaus(img)
      img = objjit(img)
      img = objco(img)

    img = img.numpy()

    return img.copy(), bbox.copy(), label.copy(), scale, fname


  def __len__(self):
    return (len(self.db)*4)

if __name__ == '__main__':

  print("Inside dataset")

  conf.data_dir = conf.ten_shot_tn
  conf.image_path = conf.ten_shot_image
  conf.annot_path = conf.ten_shot_annot

  db = FSBDDataset2x(split='beta')
  print(db.__len__())

  for ii in range(db.__len__()):
    db.__getitem__(ii)

  img,bbox,label,scale,fname = db.__getitem__(0)
  print("Displaying img shape : ",img.shape)

  img = img.transpose((1,2,0))

  fig, ax = plt.subplots()

  for bb in bbox:
    h = bb[2]-bb[0]
    w = bb[3]-bb[1]
    rect = patches.Rectangle((bb[1], bb[0]), w, h, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

  plt.imshow(img)
  plt.savefig('abc.png')
  