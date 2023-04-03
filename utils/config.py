import os

class Config :

  data_dir = ''
  image_path = ''
  annot_path = ''

  shot_list = [1,2,3,5,10]

  data_gen_files_path = '/content/drive/MyDrive/FSBD/datasets'

  image_extension = '.png'

  choosen_class = ['3','11','14','10','8']

  #only applicable for cdfsod
  cdfsod_split = 'beta'
  cdfsod_cos = False
  cdfsod_load_path = ''
  cdfsod_alpha = 0.4
  cdfsod_lamda = 2.0


  one_shot_tn = '/content/drive/MyDrive/FSBD/datasets/fsbd-1-tn.txt'
  one_shot_ts = '/content/drive/MyDrive/FSBD/datasets/fsbd-1-ts.txt'
  one_shot_image = '/content/drive/MyDrive/FSBD/datasets/test'
  one_shot_annot = '/content/drive/MyDrive/FSBD/datasets/labels/test'

  five_shot_tn = '/content/drive/MyDrive/FSBD/datasets/fsbd-5-tn.txt'
  five_shot_ts = '/content/drive/MyDrive/FSBD/datasets/fsbd-5-ts.txt'
  five_shot_image = '/content/drive/MyDrive/FSBD/datasets/test'
  five_shot_annot = '/content/drive/MyDrive/FSBD/datasets/labels/test'

  base1 = '/content/drive/MyDrive/FSBD/datasets/GTSDB/strain.txt'
  base1_image = '/content/drive/MyDrive/FSBD/datasets/GTSDB/images'
  base1_annot = '/content/drive/MyDrive/FSBD/datasets/GTSDB/images'
  base1_choosen_class = ['9','10','11','12']

  base2 = '/content/drive/MyDrive/FSBD/datasets/LISA/lisa_train.txt'
  base2_image = '/content/drive/MyDrive/FSBD/datasets/LISA/images'
  base2_annot = '/content/drive/MyDrive/FSBD/datasets/LISA/images'
  base2_choosen_class = ['0','1','2','3','4','5','6','7','8']

  base3 = '/content/drive/MyDrive/FSBD/datasets/random/random_train.txt'
  base3_image = '/content/drive/MyDrive/FSBD/datasets/random/images'
  base3_annot = '/content/drive/MyDrive/FSBD/datasets/random/images'
  base3_choosen_class = ['13','14','15','16']

  weight_path = '/content/drive/MyDrive/FSBD/weights'

  base_path = os.path.join(weight_path, "base1_base2_base3_50.pth")

  frcnft1_path = os.path.join(weight_path,'frcnft_1.pth')
  frcnft2_path = os.path.join(weight_path,'frcnft_2.pth')
  frcnft3_path = os.path.join(weight_path,'frcnft_3.pth')
  frcnft5_path = os.path.join(weight_path,'frcnft_5.pth')
  frcnft10_path = os.path.join(weight_path,'frcnft_10.pth')

  tfa1_path = os.path.join(weight_path,'tfa_1.pth')
  tfa2_path = os.path.join(weight_path,'tfa_2.pth')
  tfa3_path = os.path.join(weight_path,'tfa_3.pth')
  tfa5_path = os.path.join(weight_path,'tfa_5.pth')
  tfa10_path = os.path.join(weight_path,'tfa_10.pth')

  tfacos1_path = os.path.join(weight_path,'tfacos_1.pth')
  tfacos2_path = os.path.join(weight_path,'tfacos_2.pth')
  tfacos3_path = os.path.join(weight_path,'tfacos_3.pth')
  tfacos5_path = os.path.join(weight_path,'tfacos_5.pth')
  tfacos10_path = os.path.join(weight_path,'tfacos_10.pth')

  cdfsod1_path = os.path.join(weight_path,'cdfsod_1.pth')
  cdfsod2_path = os.path.join(weight_path,'cdfsod_2.pth')
  cdfsod3_path = os.path.join(weight_path,'cdfsod_3.pth')
  cdfsod5_path = os.path.join(weight_path,'cdfsod_5.pth')
  cdfsod10_path = os.path.join(weight_path,'cdfsod_10.pth')

  cdfsodcos1_path = os.path.join(weight_path,'cdfsodcos_1.pth')
  cdfsodcos2_path = os.path.join(weight_path,'cdfsodcos_2.pth')
  cdfsodcos3_path = os.path.join(weight_path,'cdfsodcos_3.pth')
  cdfsodcos5_path = os.path.join(weight_path,'cdfsodcos_5.pth')
  cdfsodcos10_path = os.path.join(weight_path,'cdfsodcos_10.pth')

  test_predictions_path = '/content/drive/MyDrive/FSBD/test_predictions'


  min_size = 600  # image resize
  max_size = 1000 # image resize
  num_workers = 8
  test_num_workers = 8

  # sigma for l1_smooth_loss
  rpn_sigma = 3.
  roi_sigma = 1.

  # param for optimizer
  # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
  weight_decay = 0.0005
  lr_decay = 0.1  # 1e-3 -> 1e-4
  lr = 1e-3


  # visualization
  env = 'faster-rcnn'  # visdom env
  port = 8097
  plot_every = 1  # vis every N iter

  # preset
  data = 'voc'
  pretrained_model = 'vgg16'

  # training
  epoch = 1


  use_adam = False # Use Adam optimizer
  use_chainer = False # try match everything as chainer
  use_drop = False # use dropout in RoIHead
  # debug
  debug_file = '/content/drive/MyDrive/simplest-faster-rcnn/tmp/debugf'

  test_num = 10000
  # model
  load_path = None

  caffe_pretrain = False # use caffe pretrained model instead of torchvision
  caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'



conf = Config()



