from tqdm import tqdm
import os
import numpy as np
import argparse
from PIL import Image
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torchvision.transforms as transforms

# load lpips metric
import lpips
loss_fn_alex = lpips.LPIPS(net='alex') 
loss_fn_alex.cuda()

# load ssim metric
from skimage.metrics import structural_similarity as ssim

# for psnr metric
import cv2


def get_img_names(enhanced_images_path, gt_images_path):
    img_names_enhanced= []
    img_names_gt = []

    for img_name in os.listdir(enhanced_images_path):
        # there are images with names in form of "0152(2).png" in input test images
        # we remove numbers inside () to find corresponding ground truth image
        img_name_base = img_name.split('.')[0]
        if '(' in img_name_base:
            if '(1)' not in img_name_base:
                continue
            img_name_base_w = img_name_base[:-3]
        else:
            img_name_base_w = img_name_base
        # search for corresponding ground truth image    
        for img_name_1 in os.listdir(gt_images_path):
            if img_name_base_w in img_name_1:
                img_names_enhanced.append(img_name_base+'.png')
                img_names_gt.append(img_name_1)
                break
                
    return img_names_enhanced, img_names_gt


def compute_metrics(enhanced_images_path, gt_images_path):
    
    img_names_enhanced, img_names_gt = get_img_names(enhanced_images_path, gt_images_path)

    lpips_scores = []
    psnr_scores = []
    ssim_scores = []
    
    # for AlexNet in lpips
    transform = transforms.Compose([
        transforms.Resize((1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for i, (img_name_enhanced, img_name_gt) in tqdm(enumerate(zip(img_names_enhanced, img_names_gt))):

        # compute lpips score
        img1 = Image.open(os.path.join(gt_images_path, img_name_gt))
        img2 = Image.open(os.path.join(enhanced_images_path, img_name_enhanced))
        
        lpips_scores.append(loss_fn_alex(transform(img1).cuda(), transform(img2).cuda()).data.cpu().numpy())

        # compute psnr and ssim scores
        img1 = cv2.imread(os.path.join(gt_images_path, img_name_gt), 1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

        img2 = cv2.imread(os.path.join(enhanced_images_path, img_name_enhanced), 1)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

        psnr_scores.append(cv2.PSNR(img1, img2))
        ssim_scores.append(ssim(img1, img2, data_range=img2.max() - img2.min())) 
        
    return psnr_scores, ssim_scores, lpips_scores
                 
            
if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default="./configs/inference/metrics.yaml") 
    args = parser.parse_args()
    
    config = OmegaConf.load(args.cfg)
    
    psnr_scores, ssim_scores, lpips_scores = compute_metrics(config.enhanced_images_path, config.gt_images_path)
    
    print("Mean PSNR score:", np.mean(psnr_scores))
    print("Mean SSIM score:", np.mean(ssim_scores))
    print("Mean LPIPS score:", np.mean(lpips_scores))