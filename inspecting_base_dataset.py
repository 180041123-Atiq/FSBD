from utils.config import conf

import os

def inspect_class(data_dir, image_path, annot_path):
  
  files = []

  with open(data_dir,'r') as filehandle:
    for line in filehandle:
      files.append(line)

  ccls = set()

  for f1 in files:
    fname = f1.split('.')[0]
    
    with open(os.path.join(annot_path,fname+'.txt'),'r') as lines:
      for line in lines:
        cls = line.split(' ')[0]
        ccls.add(cls)

  mp = {cn:0 for cn in ccls}

  for f1 in files:
    fname = f1.split('.')[0]
    
    with open(os.path.join(annot_path,fname+'.txt'),'r') as lines:
      for line in lines:
        cls = line.split(' ')[0]
        mp[cls] += 1

  print(mp)

if __name__=="__main__":
  inspect_class(conf.base3, conf.base3_image, conf.base3_annot)