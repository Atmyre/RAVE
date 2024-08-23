import os
# os.environ['CUDA_VISIBLE_DEVICES']='3'
import sys
import argparse
import time
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim

from enhancement_model import load_enhancement_model
import numpy as np
from PIL import Image
import glob
import time


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='directory of input folder', default='./input/')
parser.add_argument('-o', '--output', help='directory of output folder', default='./inference_result/')
parser.add_argument('-c', '--unet_pretrain_dir', help='enhancement model pretrained ckpt path', default='./pretrained_models/enhancement_model.pth')

config = parser.parse_args()
config.load_pretrain_unet = True

U_net = load_enhancement_model(config, padding_mode='reflect')

def lowlight(image_path): 

	data_lowlight = Image.open(image_path)#.convert("RGB")
	data_lowlight = (np.asarray(data_lowlight)/255.0) 
	
	data_lowlight = torch.from_numpy(data_lowlight).float().cuda()
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.unsqueeze(0) 

	light_map = U_net(data_lowlight)
	enhanced_image = torch.clamp((data_lowlight / light_map), 0, 1)

	image_path = args.output+os.path.basename(image_path)

	image_path = image_path.replace('.jpg','.png')
	image_path = image_path.replace('.JPG','.png')
	result_path = image_path
	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')): 
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))

	torchvision.utils.save_image(enhanced_image, result_path)
	
if __name__ == '__main__':
	with torch.no_grad():
		filePath = args.input
		file_list = os.listdir(filePath)
		print(file_list)
  
		for file_name in file_list:
			image=filePath+file_name
			print(image)
			lowlight(image)

		

