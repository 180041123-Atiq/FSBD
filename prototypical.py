import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm
import numpy as np
import cv2
import os
import copy


class ProtoNet(nn.Module):
  def __init__(self, backbone: nn.Module):
    super(ProtoNet, self).__init__()
    self.backbone = backbone

  def forward(
      self,
      support_images: torch.Tensor,
      support_labels: torch.Tensor,
      query_images: torch.Tensor,
  ) -> torch.Tensor:
    """
    Predict query labels using labeled support images.
    """
    # Extract the features of support and query images
    z_support = self.backbone.forward(support_images)
    z_query = self.backbone.forward(query_images)

    # Infer the number of different classes from the labels of the support set
    n_way = len(torch.unique(support_labels))
    # Prototype i is the mean of all instances of features corresponding to labels == i
    z_proto = torch.cat(
        [
            z_support[torch.nonzero(support_labels == label)].mean(0)
            for label in range(n_way)
        ]
    )

    # Compute the euclidean distance from queries to prototypes
    # dists = torch.cdist(z_query, z_proto)

    # print('z_query size : ',z_query.size())
    # print('z_proto size : ',z_proto.size())

    # changing distant metric to cosine similarity
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    # dists = cos(z_query[0], z_proto[0])

    rs = z_query.size(dim=0)
    cs = z_proto.size(dim=0)

    cscores = np.zeros(shape=(rs,cs))

    for qtem in range(rs):
      for ptem in range(cs):
        out = cos(z_query[qtem],z_proto[ptem])
        cscores[qtem][ptem] = out

    # And here is the super complicated operation to transform those distances into classification scores!
    # scores = -dists
    #scores = dists

    return cscores