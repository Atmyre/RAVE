from math import sqrt
import numpy as np
import os
import argparse
import random
from collections import OrderedDict
import pyiqa
import shutil
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
from prompt_training import PromptLearner, TextEncoder, init_prompt_learner
from latent_training import LatentVectorsLearner, init_latent_vector_learner
from inference import inference

import clip
import clip_score


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

#load clip
model, preprocess = clip.load("ViT-B/32", device=torch.device("cpu"), download_root="./clip_model/")
model.to(device)
for param in model.parameters():
    param.requires_grad = False


def initialize_guidance_model(config, guidance_learner, guidance_optimizer, guidance_snapshots_dir):
    # unfreeze pseudo prompts/latent vectors for training
    if config.exp.mode == 'clip-lit':
        guidance_learner.module.prompt_embedding.requires_grad = True
    elif config.exp.mode == 'clip-lit-latent':
        guidance_learner.module.guidance_embeddings.requires_grad = True

    total_iterations = config.train.guidance_model.num_pretrain_iters
    curr_iteration = 0
    best_guidance_learner = guidance_learner
    min_prompt_loss = 100

    # load dataset and for pseudo prompts/latent vectors training
    prompt_train_dataset = dataloader_prompt_add.lowlight_loader(config.data.backlit_images_path,
                                                           config.data.welllit_images_path)
    prompt_train_loader = torch.utils.data.DataLoader(prompt_train_dataset,
                                                            batch_size=config.train.guidance_model.batch_size,
                                                            shuffle=True,
                                                            num_workers=config.train.num_workers,
                                                            pin_memory=True
                                                            )

    # pseudo prompts/latent vectors initial training 
    while curr_iteration < total_iterations:

        for iteration, item in enumerate(prompt_train_loader):
            
            # get guidance_learner output 
            img_lowlight,label = item    
            img_lowlight = img_lowlight.cuda()
            label = label.cuda()

            output = guidance_learner(img_lowlight, 0)

            # using just cross-entropy for the initial training of 
            # pseudo prompts/latent vectors pairs
            loss = F.cross_entropy(output,label)

            # training step
            guidance_optimizer.zero_grad()
            loss.backward()
            guidance_optimizer.step()
            
            # logging and saving
            if ((iteration+1) % config.train.guidance_model.display_iter) == 0:
                
                # if we've got better guidance_learner
                if loss < min_prompt_loss:
                    min_prompt_loss = loss
                    best_guidance_learner = guidance_learner
                    torch.save(guidance_learner.state_dict(), os.path.join(guidance_snapshots_dir, "best_guidance_learner_epoch"+str(0) + '.pth'))
                # logging
                print("guidance_learner current learning rate: ", guidance_optimizer.state_dict()['param_groups'][0]['lr'])
                print("Loss at iteration", curr_iteration + 1, ":", loss.item())
                #print("output",output.softmax(dim=-1),"label",label)
                print("cross_entropy_loss",loss)

            if curr_iteration + 1 == total_iterations and loss > config.train.guidance_model.thr_loss:
                total_iterations += 100

            # update iteration counters
            curr_iteration += 1

            # check if we've finished 
            if curr_iteration == total_iterations:
                # add some more training iterations to pseudo prompts/latent vectors initial training 
                # if we haven't obtained goood enough model
                if loss > config.train.guidance_model.thr_loss:
                    total_iterations += 100
                else:
                    break

    return best_guidance_learner, guidance_optimizer, min_prompt_loss
                         

def train(config):

    exp_dir = os.path.join(config.exp.save_dir, config.exp.exp_name)
    unet_snapshots_dir = os.path.join(exp_dir, "snapshots_model")
    guidance_snapshots_dir = os.path.join(exp_dir, "snapshots_guidance") 
    
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    if not os.path.exists(unet_snapshots_dir):
        os.mkdir(unet_snapshots_dir)
    if not os.path.exists(guidance_snapshots_dir):
        os.mkdir(guidance_snapshots_dir)
    
    #add pretrained model weights
    guidance_learner = None
    if config.exp.mode == 'clip-lit':
        guidance_learner = init_prompt_learner(config, model)
    elif config.exp.mode == 'clip-lit-latent':
        guidance_learner = init_latent_vector_learner(config)
    
    U_net = load_enhancement_model(config)
    
    # load dataset for enhancement model (UNet) training
    train_dataset = dataloader_sharp.lowlight_loader(config.data.backlit_images_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=config.train.unet_model.batch_size, 
                                               shuffle=True, 
                                               num_workers=config.train.num_workers, 
                                               pin_memory=True
                                               )
    
    # loss
    if config.exp.mode == 'clip-lit':
        text_encoder = TextEncoder(model)
    L_clip = clip_score.L_clip_from_feature()
    L_clip_MSE = clip_score.L_clip_MSE()
    L_margin_loss = clip_score.four_margin_loss(0.9,0.2)
    
    # optimizers
    train_optimizer = torch.optim.Adam(U_net.parameters(), lr=config.train.unet_model.lr, weight_decay=config.train.unet_model.weight_decay)
    guidance_optimizer = torch.optim.Adam(guidance_learner.parameters(), lr=config.train.guidance_model.lr, weight_decay=config.train.guidance_model.weight_decay)

    # metric
    iqa_metric = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)

    #initial parameters
    total_iteration = 0
    cur_iteration = 0
    max_score_psnr = -10000
    pr_last_few_iter = 0
    score_psnr = [0]*30
    semi_path = ['','']
    pr_semi_path = 0
    best_model = U_net
    best_prompt = guidance_learner
    min_prompt_loss = 100
    best_prompt_iter = 0
    best_model_iter = 0
    curr_epoch = 0
    reconstruction_iter = 0
    reinit_flag = 0
 
    # start training

    # initial training of guidance pseudo prompts/vectors
    guidance_learner, guidance_optimizer, min_prompt_loss = initialize_guidance_model(config, guidance_learner, guidance_optimizer, guidance_snapshots_dir)
    best_guidance_learner = guidance_learner

    for epoch in range(config.train.num_epochs):
        if total_iteration < config.train.unet_model.num_reconstruction_iters:
            unet_train_iters = config.train.unet_model.num_reconstruction_iters
            guidance_model_train_iters = 0
        elif cur_iteration == 0:
            unet_train_iters = 2100
            guidance_model_train_iters = 1000


        # if end of current epoch of training, reset local params
        # and start new epoch
        if cur_iteration >= unet_train_iters + guidance_model_train_iters:
            cur_iteration=0
            min_prompt_loss=100
            max_score_psnr=-10000
            score_psnr=[0]*30
            curr_epoch += 1

        # Unet training
        elif cur_iteration < unet_train_iters: 
            if cur_iteration==0:
                guidance_learner = best_guidance_learner

            if config.exp.mode == 'clip-lit':
                prompt_embedding=guidance_learner.module.prompt_embedding
                prompt_embedding.requires_grad = False
                tokenized_pseudo_rompts= torch.cat([clip.tokenize(p) for p in [" ".join(["X"]*config.train.guidance_model.length_prompt)]])
                eos_indices = tokenized_pseudo_rompts.argmax(dim=-1)
                guidance_embeddings = text_encoder(prompt_embedding, eos_indices)
            elif config.exp.mode == 'clip-lit-latent':
                guidance_embeddings=guidance_learner.module.guidance_embeddings
                guidance_embeddings.requires_grad = False
            
            # freeze all the parameters of pseudo prompt / latent vectors trainer
            for name, param in guidance_learner.named_parameters():
                param.requires_grad_(False)

            # unfreeze all the parameters of UNet model
            for name, param in U_net.named_parameters():
                param.requires_grad_(True)
            U_net.train()

            for iteration, item in enumerate(train_loader): 
        
                # get enhancement model output for input batch of backlit images
                img_lowlight, img_lowlight_path = item
                img_lowlight = img_lowlight.cuda()

                light_map  = U_net(img_lowlight)
                enhanced_image=torch.clamp(((img_lowlight) /(light_map+0.000000001)),0,1)
               
                # compute losses
                # guidance loss
                cliploss=16*20*L_clip(enhanced_image, guidance_embeddings)
                # reconstruction loss
                clip_MSEloss = 25*L_clip_MSE(enhanced_image, img_lowlight,[1.0,1.0,1.0,1.0,0.5])

                if total_iteration >= config.train.unet_model.num_reconstruction_iters:
                    # training the model with cliploss and reconstruction loss
                    loss = cliploss + 0.9*clip_MSEloss
                else:
                    # training the model with reconstruction loss only
                    loss = 25*L_clip_MSE(enhanced_image, img_lowlight,[1.0,1.0,1.0,1.0,1.0])
                
                # do training step of enhancement model
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()
                
                # 
                with torch.no_grad():
                    if total_iteration<config.train.unet_model.num_reconstruction_iters+config.train.guidance_model.num_pretrain_iters:
                        score_psnr[pr_last_few_iter] = torch.mean(iqa_metric(img_lowlight, enhanced_image))
                        reconstruction_iter+=1
                        if sum(score_psnr).item()/30.0 < 8 and reconstruction_iter >100:
                            reinit_flag=1
                    else:
                        score_psnr[pr_last_few_iter] = -loss

                    pr_last_few_iter += 1
                    if pr_last_few_iter == 30:
                        pr_last_few_iter = 0
                    if (sum(score_psnr).item()/30.0) > max_score_psnr and ((total_iteration+1) % config.train.unet_model.display_iter) == 0:
                        max_score_psnr = sum(score_psnr).item()/30.0
                        torch.save(U_net.state_dict(), os.path.join(unet_snapshots_dir, "best_model_round"+str(curr_epoch) + '.pth'))    
                        best_model = U_net
                        best_model_iter = total_iteration+1
                        print(max_score_psnr)
                        images_save_path = './'+config.exp.exp_name+'/result_'+config.exp.exp_name+'/result_jt_'+str(total_iteration+1)+"_psnr_or_-loss"+str(max_score_psnr)[:8]+'/'
                        inference(config.data.backlit_images_path, images_save_path, U_net, size=256)
                        if total_iteration > config.train.unet_model.num_reconstruction_iters+config.train.guidance_model.num_pretrain_iters:
                            semi_path[pr_semi_path] = images_save_path
                            print(semi_path)
                        torch.save(U_net.state_dict(), os.path.join(unet_snapshots_dir, "iter_" + str(total_iteration+1) + '.pth'))
                
                if reinit_flag == 1:
                    print(sum(score_psnr).item()/30.0)
                    print("reinitialization...")
                    seed=random.randint(0,100000)
                    print("current random seed: ",seed)
                    torch.cuda.manual_seed_all(seed)
                    U_net=load_unet(config)
                    reconstruction_iter=0
                    train_optimizer = torch.optim.Adam(U_net.parameters(), lr=config.train.unet_model.lr, weight_decay=config.train.unet_model.weight_decay)
                    config.train.unet_model.num_reconstruction_iters+=100
                    reinit_flag=0
                
                # logging
                if ((total_iteration+1) % config.train.unet_model.display_iter) == 0:
                    print("training current learning rate: ",train_optimizer.state_dict()['param_groups'][0]['lr'])
                    print("Loss at iteration", total_iteration+1,"epoch",epoch, ":", loss.item())
                    print("loss_clip",cliploss," reconstruction loss",clip_MSEloss)
                    print(cur_iteration+1," ",total_iteration+1)
                    print(unet_train_iters,' ',unet_train_iters + guidance_model_train_iters)

                cur_iteration += 1
                total_iteration += 1

                if cur_iteration == unet_train_iters and total_iteration > config.train.unet_model.num_reconstruction_iters and (cliploss + 0.9*clip_MSEloss > config.train.unet_model.thr_loss):
                    unet_train_iters += 60
                elif cur_iteration == unet_train_iters:
                    print("switch to fine-tune the prompt pair")
                    break
                
        
        # pseudo prompts/latent vectors fine-tuning
        else:
            
            if config.exp.mode == 'clip-lit':
                guidance_learner.module.prompt_embedding.requires_grad = True
            elif config.exp.mode == 'clip-lit-latent':
                guidance_learner.module.guidance_embeddings.requires_grad = True

            # load the data on the start of fine-tuning 
            if cur_iteration == unet_train_iters:
                if total_iteration >= config.train.guidance_model.num_pretrain_iters:
                    pr_semi_path = 1-pr_semi_path


                # load dataset and construct loss for pseudo prompts/latent vectors training 
                # depending on if results from previous iteration are available
                if semi_path[0]=='':
                    L_margin_loss = clip_score.four_margin_loss(1.0,0.2)
                    guidance_train_dataset = dataloader_prompt_margin.lowlight_loader(config.data.backlit_images_path,
                                                                                      config.data.welllit_images_path
                                                                                      )     
                    guidance_train_loader = torch.utils.data.DataLoader(guidance_train_dataset, 
                                                                        batch_size=config.train.guidance_model.batch_size, 
                                                                        shuffle=True, 
                                                                        num_workers=config.train.num_workers, 
                                                                        pin_memory=True)
                elif semi_path[1]=='':
                    L_margin_loss = clip_score.four_margin_loss(0.9,0.2)
                    guidance_train_dataset = dataloader_prompt_margin.lowlight_loader(config.data.backlit_images_path,
                                                                                      config.data.welllit_images_path,
                                                                                      semi_path[0]
                                                                                      )
                    guidance_train_loader = torch.utils.data.DataLoader(guidance_train_dataset, 
                                                                      batch_size=config.train.guidance_model.batch_size, 
                                                                      shuffle=True, 
                                                                      num_workers=config.train.num_workers, 
                                                                      pin_memory=True
                                                                      )
                else:
                    L_margin_loss = clip_score.four_margin_loss(0.9,0.1)
                    guidance_train_dataset = dataloader_prompt_margin.lowlight_loader(config.data.backlit_images_path,
                                                                                    config.data.welllit_images_path,
                                                                                    semi_path[1-pr_semi_path],
                                                                                    semi_path[pr_semi_path]
                                                                                    )
                    guidance_train_loader = torch.utils.data.DataLoader(guidance_train_dataset, 
                                                                      batch_size=config.train.guidance_model.batch_size, 
                                                                      shuffle=True, 
                                                                      num_workers=config.train.num_workers, 
                                                                      pin_memory=True
                                                                      )
            
            # fix enhancement model 
            U_net = best_model
            for name, param in U_net.named_parameters():
                param.requires_grad_(False)
                
            # train the guidance model
            for iteration, item in enumerate(guidance_train_loader):
                img_feature_list,labels = item 
                labels = labels.cuda()

                if len(img_feature_list) == 2:
                    inp, ref = img_feature_list
                    loss=200*L_margin_loss(guidance_learner(inp.cuda()),
                                           guidance_learner(ref.cuda()), 
                                           labels, 
                                           2
                                           )
                elif len(img_feature_list) == 3:
                    inp, semi1, ref = img_feature_list
                    loss = 200*L_margin_loss(guidance_learner(inp.cuda()),
                                             guidance_learner(ref.cuda()),
                                             labels,
                                             3,
                                             guidance_learner(semi1.cuda())
                                             )
                else:
                    inp,semi1, semi2,ref = img_feature_list
                    loss = 200*L_margin_loss(guidance_learner(inp.cuda()),
                                             guidance_learner(ref.cuda()),
                                             labels,
                                             4,
                                             guidance_learner(semi1.cuda()),
                                             guidance_learner(semi2.cuda())
                                             )
                
                guidance_optimizer.zero_grad()
                loss.backward()
                guidance_optimizer.step()
                

                # logging and saving
                if ((total_iteration + 1) % config.train.guidance_model.display_iter) == 0:
                    if loss < min_prompt_loss:
                        min_prompt_loss = loss
                        best_guidance_learner = guidance_learner
                        best_guidance_learner_iter = total_iteration+1
                        torch.save(guidance_learner.state_dict(), os.path.join(guidance_snapshots_dir, "best_guidance_learner_round"+str(curr_epoch) + '.pth'))
                    print("prompt current learning rate: ", guidance_optimizer.state_dict()['param_groups'][0]['lr'])                    
                    print("Loss at iteration", total_iteration+1, ":", loss.item())
                    print("margin_loss",loss)
                    print(cur_iteration+1," ",total_iteration+1)
                    print(unet_train_iters,' ',unet_train_iters + guidance_model_train_iters)
            

                # update iteration counters
                cur_iteration += 1
                total_iteration += 1   

                # check if we've finished 
                if cur_iteration == unet_train_iters + guidance_model_train_iters:
                    # add some more training iterations to pseudo prompts/latent vectors initial training 
                    # if we haven't obtained goood enough model
                    if loss > config.train.guidance_model.thr_loss:
                        guidance_model_train_iters += 100
                    else:
                        break
            

if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default="./configs/train/clip_lit.yaml") 
    args = parser.parse_args()
    
    config = OmegaConf.load(args.cfg)

    train(config)

