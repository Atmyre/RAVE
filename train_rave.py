from math import sqrt
import numpy as np
import os
import argparse
import random
from collections import OrderedDict
import pyiqa
import shutil
from tqdm import tqdm
from PIL import Image
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim
from torch.utils.tensorboard import SummaryWriter

import dataloader_prompt_margin
import dataloader_prompt_add
import dataloader_images as dataloader_sharp 
from enhancement_model import load_enhancement_model

import clip
import clip_score


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

#load clip
model, preprocess = clip.load("ViT-B/32", device=torch.device("cpu"), download_root="./clip_model/")
model.to(device)
for param in model.parameters():
    param.requires_grad = False


def extract_embs(data_path, model):
    model_embs = None
    
    for img_name in tqdm(os.listdir(data_path)):

        try:
            img = Image.open(os.path.join(data_path, img_name))
        except Exception as e:
            # corrupted image
            continue

        # processing input and putting through a model
        inputs = preprocess(img).to(device).view([1, 3, 224, 224])
        outputs = model.visual(inputs)
        cls = outputs.cpu().detach().numpy()
        cls = cls / np.linalg.norm(cls, axis=1, keepdims=True)

        if model_embs is None:
            model_embs = cls
        else:
            model_embs = np.concatenate((model_embs, cls), axis = 0)

    return model_embs


def train(config):
    
    #load model
    exp_dir = os.path.join(config.exp.save_dir, config.exp.exp_name)
    unet_snapshots_dir = os.path.join(exp_dir, "snapshots_model")
    
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    if not os.path.exists(unet_snapshots_dir):
        os.mkdir(unet_snapshots_dir)

    U_net = load_enhancement_model(config)
    
    # load dataset for enhancement model (UNet) training
    train_dataset = dataloader_sharp.lowlight_loader(config.data.backlit_images_path, 
                                                     config.data.welllit_images_path
                                                     )
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=config.unet_model.batch_size, 
                                               shuffle=True, 
                                               num_workers=config.train.num_workers, 
                                               pin_memory=True
                                               )
    
    #loss
    L_clip = clip_score.L_clip_from_feature()
    L_clip_MSE = clip_score.L_clip_MSE()
    
    # optimizers
    train_optimizer = torch.optim.Adam(U_net.parameters(), lr=config.unet_model.lr, weight_decay=config.unet_model.weight_decay)
   
    # metric
    iqa_metric = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)

    # prepare residual vector
    tokenizer = clip.simple_tokenizer.SimpleTokenizer()

    vocab_embs = {}
    for i in tqdm(range(49408)):
        with torch.no_grad():
            token = tokenizer.decode([i])
            inputs = torch.cat([clip.tokenize(token)]).to(device)
            output = model.encode_text(inputs)[0].data.cpu().numpy()
            vocab_embs[token] = output

    
    embs_neg = extract_embs(config.data.backlit_images_path, model)
    embs_pos = extract_embs(config.data.welllit_images_path, model)

    vector_pos = np.mean(embs_pos , axis=0)
    vector_pos = vector_pos / np.linalg.norm(vector_pos)

    vector_neg = np.mean(embs_neg, axis=0)
    vector_neg = vector_neg / np.linalg.norm(vector_neg)

    residual_vector = vector_pos - vector_neg
    residual_vector = residual_vector / np.linalg.norm(residual_vector)

    if config.guidance.remove_first_n_tokens > 0:

        token_scores = {}
        for key in vocab_embs.keys():
            token_scores[key] = np.dot(vocab_embs[key] / np.linalg.norm(vocab_embs[key]), \
                                 residual_vector / np.linalg.norm(residual_vector))

        token_scores = sorted(token_scores.items(), key=lambda x:x[1])

        embs_neg_add = np.mean([vocab_embs[x]*y for x, y in token_scores[-15:][::-1]], axis=0)
        print('removing info from tokens:', ' '.join([x for x, y in token_scores[-15:][::-1]]))
        embs_neg_add = embs_neg_add / np.linalg.norm(embs_neg_add)

        embs_pos_add = np.mean([vocab_embs[x]*y for x, y in scores[:15]], axis=0)
        print('adding info from tokens:', ' '.join([x for x, y in token_scores[:15]]))
        embs_pos_add = embs_pos_add / np.linalg.norm(embs_pos_add)

        remove_emb = embs_neg_add + embs_pos_add
        remove_emb = remove_emb / np.linalg.norm(remove_emb)

        # update the residual vector
        residual_vector = residual_vector - remove_emb*np.dot(remove_emb, residual_vector)
        residual_vector = residual_vector / np.linalg.norm(residual_vector)
    
    thr = np.dot(vector_pos, residual_vector)
    residual_vector = torch.tensor(residual_vector).view(1, 512).to(device)

    #initial parameters
    total_iteration = 0
    cur_iteration = 0
    max_score_psnr = -10000
    pr_last_few_iter = 0
    score_psnr = [0]*30
    best_model = U_net
    min_prompt_loss = 100
    best_model_iter = 0
    curr_epoch = 0
    reconstruction_iter = 0
    reinit_flag = 0

    #Start training
    U_net.train()
    
    for epoch in range(config.train.num_epochs):

        if total_iteration < config.unet_model.num_reconstruction_iters:
            unet_train_iters = config.unet_model.num_reconstruction_iters
        elif cur_iteration == 0:
            unet_train_iters = 2100

        # for consistency with CLIP-LIT, 
        # though we can omit that and just train UNet for a 
        # desired number of iterations
        if cur_iteration >= unet_train_iters:
            cur_iteration=0
            max_score_psnr=-10000
            score_psnr=[0]*30
            curr_epoch += 1

        else: 
            for iteration, item in enumerate(train_loader): 
        
                # get enhancement model output for input batch of backlit images
                img_lowlight ,img_lowlight_path=item
                img_lowlight = img_lowlight.cuda()

                light_map  = U_net(img_lowlight)
                enhanced_image=torch.clamp(((img_lowlight) /(light_map+0.000000001)),0,1)
               
                # compute losses
                # guidance loss
                cliploss=16*20*L_clip(enhanced_image, residual_vector, thr=thr)
                # reconstruction loss
                clip_MSEloss = 25*L_clip_MSE(enhanced_image, img_lowlight,[1.0,1.0,1.0,1.0,0.5])

                if total_iteration >= config.unet_model.num_reconstruction_iters:
                    # training the model with cliploss and reconstruction loss
                    loss = 6*cliploss + 0.9*clip_MSEloss
                else:
                    # training the model with reconstruction loss only
                    loss = 25*L_clip_MSE(enhanced_image, img_lowlight,[1.0,1.0,1.0,1.0,1.0])

                # do training step of enhancement model
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

                # 
                with torch.no_grad():
                    if total_iteration<config.unet_model.num_reconstruction_iters:
                        score_psnr[pr_last_few_iter] = torch.mean(iqa_metric(img_lowlight, enhanced_image))
                        reconstruction_iter+=1
                        if sum(score_psnr).item()/30.0 < 8 and reconstruction_iter >100:
                            reinit_flag=1
                    else:
                        score_psnr[pr_last_few_iter] = -loss

                    pr_last_few_iter += 1
                    if pr_last_few_iter == 30:
                        pr_last_few_iter = 0
                    if (sum(score_psnr).item()/30.0) > max_score_psnr and ((total_iteration+1) % config.unet_model.display_iter) == 0:
                        max_score_psnr = sum(score_psnr).item()/30.0
                        torch.save(U_net.state_dict(), os.path.join(unet_snapshots_dir, "best_model_round"+str(curr_epoch) + '.pth'))    
                        best_model = U_net
                        best_model_iter = total_iteration+1
                        print(max_score_psnr)
                        torch.save(U_net.state_dict(), os.path.join(unet_snapshots_dir, "iter_" + str(total_iteration+1) + '.pth')) 
                
                if reinit_flag == 1:
                    print(sum(score_psnr).item()/30.0)
                    print("reinitialization...")
                    seed=random.randint(0,100000)
                    print("current random seed: ",seed)
                    torch.cuda.manual_seed_all(seed)
                    U_net=load_unet(config)
                    reconstruction_iter=0
                    train_optimizer = torch.optim.Adam(U_net.parameters(), lr=config.unet_model.lr, weight_decay=config.unet_model.weight_decay)
                    config.unet_model.num_reconstruction_iters+=100
                    reinit_flag=0
                
                # logging
                if ((total_iteration+1) % config.unet_model.display_iter) == 0:
                    print("training current learning rate: ",train_optimizer.state_dict()['param_groups'][0]['lr'])
                    print("Loss at iteration", total_iteration+1,"epoch",epoch, ":", loss.item())
                    print("loss_clip",cliploss," reconstruction loss",clip_MSEloss)
                    print(cur_iteration+1," ",total_iteration+1)
                    print(unet_train_iters)

                cur_iteration += 1
                total_iteration += 1

                if cur_iteration == unet_train_iters and total_iteration > config.unet_model.num_reconstruction_iters and (cliploss + 0.9*clip_MSEloss > config.unet_model.thr_loss):
                    unet_train_iters += 60
                elif cur_iteration == unet_train_iters:
                    break
            

if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default="./configs/train/rave.yaml") 
    args = parser.parse_args()
    
    config = OmegaConf.load(args.cfg)

    train(config)

