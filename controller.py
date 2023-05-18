import argparse
import frcnft
import fsbd4x
import fsbd2x
import tfa
import test_proto
from utils.config import conf
import sys

if __name__=="__main__":

  print("I am inside Controller")
  
  train = True if input('Want to train? t or f : ') == 't' else False
  cos = True if input('Want to add cos? t or f : ') == 't' else False
  proto = True if input('Want to add proto? t or f : ') == 't' else False
  
  shots = input('Give number of shots : ')

  if shots == '1' :
    conf.image_path = conf.one_shot_image
    conf.annot_path = conf.one_shot_annot

    if train == True :
      conf.data_dir = conf.one_shot_tn
    else : conf.data_dir = conf.one_shot_ts
  elif shots == '2' :
    conf.image_path = conf.two_shot_image
    conf.annot_path = conf.two_shot_annot

    if train == True :
      conf.data_dir = conf.two_shot_tn
    else : conf.data_dir = conf.two_shot_ts
  elif shots == '3' :
    conf.image_path = conf.three_shot_image
    conf.annot_path = conf.three_shot_annot

    if train == True :
      conf.data_dir = conf.three_shot_tn
    else : conf.data_dir = conf.three_shot_ts
  elif shots == '5' :
    conf.image_path = conf.five_shot_image
    conf.annot_path = conf.five_shot_annot

    if train == True :
      conf.data_dir = conf.five_shot_tn
    else : conf.data_dir = conf.five_shot_ts
  elif shots == '10' :
    conf.image_path = conf.ten_shot_image
    conf.annot_path = conf.ten_shot_annot

    if train == True :
      conf.data_dir = conf.ten_shot_tn
    else : conf.data_dir = conf.ten_shot_ts
  else :
    print('Worng shots number, has to be one of them [1,2,3,5,10]')
    sys.exit()
    
  
  tr_mode = 'beta' if input('Want beta or alpha? b or a : ') == 'b' else 'alpha'

  model = input('Give a model name : ')

  load_in = input('Want to load? : ')
  load_path = 'none'

  if load_in == 'base' :
    load_path = conf.base_path
  elif load_in == 'frcnft' :
    if shots == '1' : load_path = conf.frcnft1_path
    elif shots == '3' : load_path = conf.frcnft3_path
    elif shots == '5' : load_path = conf.frcnft5_path
    elif shots == '10' : load_path = conf.frcnft10_path
  elif load_in == 'tfa' :
    if shots == '1' : 
      if cos == False : load_path = conf.tfa1_path
      elif cos == True : load_path = conf.tfacos1_path
    elif shots == '2' : 
      if cos == False : load_path = conf.tfa2_path
      elif cos == True : load_path = conf.tfacos2_path
    elif shots == '3' : 
      if cos == False : load_path = conf.tfa3_path
      elif cos == True : load_path = conf.tfacos3_path
    elif shots == '5' : 
      if cos == False : load_path = conf.tfa5_path
      elif cos == True : load_path = conf.tfacos5_path
    elif shots == '10' : 
      if cos == False : load_path = conf.tfa10_path
      elif cos == True : load_path = conf.tfacos10_path
  elif load_in == 'cdfsod' :
    if shots == '1' :
      if cos == False : load_path = conf.cdfsod1_path
      elif cos == True : load_path = conf.cdfsodcos1_path
    elif shots == '3' :
      if cos == False : load_path = conf.cdfsod3_path
      elif cos == True : load_path = conf.cdfsodcos3_path
    elif shots == '5' :
      if cos == False : load_path = conf.cdfsod5_path
      elif cos == True : load_path = conf.cdfsodcos5_path
    elif shots == '10' :
      if cos == False : load_path = conf.cdfsod10_path
      elif cos == True : load_path = conf.cdfsodcos10_path
  elif load_in == 'fsbd2x' :
    if shots == '1' : load_path = conf.fsbd2x1_path
    elif shots == '2' : load_path = conf.fsbd2x2_path
    elif shots == '3' : load_path = conf.fsbd2x3_path
    elif shots == '5' : load_path = conf.fsbd2x5_path
    elif shots == '10' : load_path = conf.fsbd2x10_path
  elif load_in == 'fsbd4x' :
    if shots == '1' : load_path = conf.fsbd4x1_path
    elif shots == '2' : load_path = conf.fsbd4x2_path
    elif shots == '3' : load_path = conf.fsbd4x3_path
    elif shots == '5' : load_path = conf.fsbd4x5_path
    elif shots == '10' : load_path = conf.fsbd4x10_path
  elif load_in == 'none':
    load_path = 'none'
  else :
    print('Wrong model to load')
    sys.exit()

  if model == 'frcnft' :
    if train == True : frcnft.train(split=tr_mode, cos = cos, load_path = load_path, shots = shots, model_name = model)
    if train == False : frcnft.test(split=tr_mode, cos = cos, load_path = load_path, shots = shots, model_name = model)
  elif model == 'tfa' :
    if train == True : 
      if cos == True : tfa.train(split=tr_mode, cos = cos, load_path = load_path, shots = shots, model_name = model+'cos')
      elif cos == False : tfa.train(split=tr_mode, cos = cos, load_path = load_path, shots = shots, model_name = model)
    if train == False : 
      if cos == True : tfa.test(split=tr_mode, cos = cos, load_path = load_path, shots = shots, model_name = model+'cos')
      elif cos == False : tfa.test(split=tr_mode, cos = cos, load_path = load_path, shots = shots, model_name = model)
  elif model == 'fsbd2x' :
    if train == True : fsbd2x.train(split=tr_mode, cos=cos, load_path=load_path, shots=shots, model_name=model)
    elif train == False : fsbd2x.test(split=tr_mode, cos=cos, load_path=load_path, shots=shots, model_name=model)
  elif model == 'fsbd4x' :
    if train == True : fsbd4x.train(split=tr_mode, cos=True, load_path=load_path, shots=shots, model_name=model)
    elif train == False : fsbd4x.test(split=tr_mode, cos=True, load_path=load_path, shots=shots, model_name=model)
  elif model == 'cdfsod' :
    alpha = float(input('Alpha value for cdfsod : '))
    lamda = float(input('Lamda value for cdfsod : '))

    conf.cdfsod_split = tr_mode
    conf.cdfsod_cos = cos
    conf.cdfsod_load_path = load_path
    conf.cdfsod_alpha = alpha 
    conf.cdfsod_lamda = lamda 

    import cdfsod

    if train == True :
      if cos == False :
          cdfsod.train(split=tr_mode, cos=cos, load_path=load_path, shots=shots, model_name=model, alpha=alpha, lamda=lamda )
      elif cos == True :
        cdfsod.train(split=tr_mode, cos=cos, load_path=load_path, shots=shots, model_name=model+'cos', alpha=alpha, lamda=lamda )
    elif train == False :
      if cos == False :
        if proto == False :
          cdfsod.test(split=tr_mode, cos=cos, load_path=load_path, shots=shots, model_name=model )
        elif proto == True :
          test_proto.test(split=tr_mode, cos=cos, load_path=load_path, shots=shots, model_name=model )
      elif cos == True :
        if proto == False :
          cdfsod.test(split=tr_mode, cos=cos, load_path=load_path, shots=shots, model_name=model+'cos' )
        elif proto == True :
          test_proto.test(split=tr_mode, cos=cos, load_path=load_path, shots=shots, model_name=model+'cos' )
  else :
    print('Wrong model to work with')
    sys.exit()
  
