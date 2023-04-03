import argparse
import frcnft
from utils.config import conf

# Define the ArgumentParser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("shots")
parser.add_argument("tr_mode")
parser.add_argument("base")
parser.add_argument("model")
parser.add_argument("alpha")
parser.add_argument("lamda")
parser.add_argument("proto")
parser.add_argument("resnet")
parser.add_argument("pretrain")
parser.add_argument("cos")

# Indicate end of argument definitions and parse args
args = parser.parse_args()



if __name__=="__main__":
  # Access arguments by using dot syntax with their name

  if args.cos == 'cos' :
    cos = True
  else : cos = False

  load_path = 'none'

  if args.base == 'base' : 
    load_path = conf.base_path
  elif args.base == 'frcnft' :
    load_path = conf.frcnft_path 

  print("5 way",args.shots,"shots")

  if args.shots == '1' :
    conf.data_dir = conf.one_shot_tn
    conf.image_path = conf.one_shot_image
    conf.annot_path = conf.one_shot_annot
  elif args.shots == '5' :
    conf.data_dir = conf.five_shot_tn
    conf.image_path = conf.five_shot_image
    conf.annot_path = conf.five_shot_annot

  if args.model == "tfa" :
    print("training tfa")
  elif args.model == 'frcnft' :

    print("training frcnft")
    frcnft.train(split = args.tr_mode, cos = cos, load_path = load_path)

  elif args.model == 'cd' :
    print("training cd fsod with ",args.alpha," and ",args.lamda)

  if args.proto == "proto" :
    print("Prototypical network is added with resnet -",args.resnet," pretrained with ",args.pretrain)

  if args.cos == 'cos' :
    print("Instance feature normalization is also added")