
import torch
import permutohedral_encoding as permuto_enc
import numpy as np

#create encoding

# device = 0

pos_dim=6
capacity=pow(2,18) 
nr_levels=24 
nr_feat_per_level=2 
coarsest_scale=1.0 
finest_scale=0.0001 
scale_list=np.geomspace(coarsest_scale, finest_scale, num=nr_levels)

# with torch.cuda.device(device):
#     encoding=permuto_enc.PermutoEncoding(pos_dim, capacity, nr_levels, nr_feat_per_level, scale_list).cuda()
#     #create points
#     nr_points=1000
#     points=torch.rand(nr_points, 6).cuda()


d = 0
device = torch.device('cuda', d)
torch.cuda.set_device(d)
    
# hash_encoder = HashEncoder().cuda().to(device)

encoding=permuto_enc.PermutoEncoding(pos_dim, capacity, nr_levels, nr_feat_per_level, scale_list).cuda().to(device)
    #create points
nr_points=1000
points=torch.rand(nr_points, 6).cuda().to(device)


#encode
features=encoding(points)

print(features)