import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim

from collections import OrderedDict


class LatentVectorsLearner(nn.Module):
    def __init__(self, initials=None):
        super(LatentVectorsLearner, self).__init__()

        if initials is not None:
            state_dict = torch.load(initials)
            
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            self.guidance_embeddings = nn.Parameter(new_state_dict['guidance_embeddings']).cuda()
        else:
            self.guidance_embeddings = nn.Parameter(torch.randn(2, 512))
        
        self.guidance_embeddings.requires_grad=True

    def forward(self, tensor, use_softmax=1):
        
        for i in range(tensor.shape[0]):
            image_features=tensor[i]
            nor=torch.norm(self.guidance_embeddings,dim=-1, keepdim=True)
            if not use_softmax:
                similarity = (100.0 * image_features @ (self.guidance_embeddings/nor).T)
                if(i==0):
                    probs=similarity
                else:
                    probs=torch.cat([probs,similarity],dim=0)
            else:
                similarity = (100.0 * image_features @ (self.guidance_embeddings/nor).T).softmax(dim=-1)
                if(i==0):
                    probs=similarity[:,0]
                else:
                    probs=torch.cat([probs,similarity[:,0]],dim=0)
        return probs


def init_latent_vector_learner(config):
    if config.guidance_model.load_pretrain:
        latent_learner=LatentVectorsLearner(config.guidance_model.pretrain_dir).cuda()
    else:
        latent_learner=LatentVectorsLearner().cuda()
    latent_learner =  torch.nn.DataParallel(latent_learner)

    return latent_learner