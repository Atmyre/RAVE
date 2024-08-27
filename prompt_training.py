import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim

import clip

from collections import OrderedDict


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, eos_indices):

        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), eos_indices.long()] @ self.text_projection
        
        return x


class PromptLearner(nn.Module):
    def __init__(self, config, model, initials=None):
        super(PromptLearner, self).__init__()

        self.text_encoder = TextEncoder(model)

        # determining eos indices for pseudo prompts 
        # for TextEncoder to know where CLF token in CLIP embeddings is
        tokenized_pseudo_rompts= torch.cat([clip.tokenize(p) for p in [" ".join(["X"]*config.guidance_model.length_prompt)]])
        self.eos_indices = tokenized_pseudo_rompts.argmax(dim=-1)

        if isinstance(initials, list):
            text = clip.tokenize(initials).cuda()
            self.prompt_embedding = nn.Parameter(model.token_embedding(text).requires_grad_()).cuda()
        elif isinstance(initials, str):
            prompt_path=initials

            state_dict = torch.load(prompt_path)
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            self.prompt_embedding=nn.Parameter(new_state_dict['prompt_embedding']).cuda()
            self.prompt_embedding.requires_grad = True
        else:
            self.prompt_embedding=torch.nn.init.xavier_normal_(nn.Parameter(model.token_embedding([" ".join(["X"]*config.guidance_model.length_prompt)," ".join(["X"]*config.guidance_model.length_prompt)]).requires_grad_())).cuda()

    def forward(self, tensor, use_softmax=1):
        pseudo_prompt_latent_vectors = self.text_encoder(self.prompt_embedding, self.eos_indices)
        
        for i in range(tensor.shape[0]):
            image_features=tensor[i]
            nor=torch.norm(pseudo_prompt_latent_vectors,dim=-1, keepdim=True)
            if not use_softmax:
                similarity = (100.0 * image_features @ (pseudo_prompt_latent_vectors/nor).T)
                if(i==0):
                    probs=similarity
                else:
                    probs=torch.cat([probs,similarity],dim=0)
            else:
                similarity = (100.0 * image_features @ (pseudo_prompt_latent_vectors/nor).T).softmax(dim=-1)
                if(i==0):
                    probs=similarity[:,0]
                else:
                    probs=torch.cat([probs,similarity[:,0]],dim=0)
        return probs


def init_prompt_learner(config, model):
    if config.guidance_model.load_pretrain:
        prompt_learner=PromptLearner(config, model, initials=config.guidance_model.pretrain_dir, ).cuda()
    else:
        prompt_learner=PromptLearner(config, model, initials=[" ".join(["X"]*(config.guidance_model.length_prompt))," ".join(["X"]*(config.guidance_model.length_prompt))]).cuda()
    prompt_learner =  torch.nn.DataParallel(prompt_learner)

    return prompt_learner
