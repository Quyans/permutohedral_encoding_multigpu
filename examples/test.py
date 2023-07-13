import torch
# import permutohedral_encoding as permuto_enc
import numpy as np
import os
import tinycudann as tcnn

#create encoding

device = 1
gpu = 'cuda:1'

config = {
  "otype": "HashGrid",
  "n_levels": 16,
  "n_features_per_level": 2,
  "log2_hashmap_size": 19,
  "base_resolution": 16,
  "per_level_scale": 2.0,
 }

with torch.cuda.device(device):
    encoding = tcnn.Encoding(4, config)
            
n_output_dims = encoding.n_output_dims

nr_points=1000
points=torch.rand(nr_points, 4).to(gpu)

features=encoding(points)

print(features)