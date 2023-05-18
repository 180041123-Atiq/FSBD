from utils.config import conf

import os 
import glob

def generate_txt_file(shots,files):

  fls = []

  for f in files:
    with open(f,'r') as lines :
      ll = []
      for line in lines:
        ll.append(line)
      if len(ll) == 1 : fls.append(f)

  # print(len(fls))

  ccls = conf.choosen_class

  
  mp = {icls:[] for icls in ccls}

  for ii in range(len(fls)):
    with open(fls[ii],'r') as lines:
      for line in lines:
        cls = line.split(' ')[0]

        if cls in ccls :
          if len(mp[cls]) < shots+15 or True : mp[cls].append(fls[ii].split('/')[-1])


  with open(os.path.join(conf.data_gen_files_path,'Fsbd-'+str(shots)+'-tn.txt'),'w') as filehandle:

    ini = False
    for ic in ccls:
      for i in range(shots):

        if ini == False : 
          ini = True
        else : filehandle.write('\n')

        filehandle.write('%s' % mp[ic][i])


  with open(os.path.join(conf.data_gen_files_path,'Fsbd-'+str(shots)+'-ts.txt'),'w') as filehandle:

    ini = False
    for ic in ccls:
      for i in range(shots,len(mp[ic])):

        if ini == False : 
          ini = True
        else : filehandle.write('\n')

        filehandle.write('%s' % mp[ic][i])

  amp = {icls:0 for icls in ccls}

  with open(os.path.join(conf.data_gen_files_path,'Fsbd-'+str(shots)+'-tn.txt'),'r') as files :
    for f in files:
      with open(os.path.join(conf.annot_path,f.split('.')[0]+'.txt'),'r') as lines:
        for line in lines :
          cls = line.split(' ')[0]
          amp[cls] += 1

  print()
  print('-------5 way ',shots,'shots tn--------')
  print(amp)

  amp = {icls:0 for icls in ccls}

  with open(os.path.join(conf.data_gen_files_path,'Fsbd-'+str(shots)+'-ts.txt'),'r') as files :
    for f in files:
      with open(os.path.join(conf.annot_path,f.split('.')[0]+'.txt'),'r') as lines:
        for line in lines :
          cls = line.split(' ')[0]
          amp[cls] += 1

  print()
  print('-------5 way undeterministic shots ts--------')
  print(amp)



if __name__=="__main__":

  conf.annot_path = "/content/drive/MyDrive/FSBD/datasets/labels/test"

  files = glob.glob(os.path.join(conf.annot_path,"*.txt"))

  for it in conf.shot_list:
    generate_txt_file(it,files)