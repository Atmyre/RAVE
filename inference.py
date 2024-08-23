import torch
import os
import time
from tqdm import tqdm

import numpy as np
from PIL import Image
import glob
import time
import torchvision

from enhancement_model import load_enhancement_model


def enhance_image(image_path, image_list_path, result_list_path, unet_model, size=None): 

	data_lowlight = Image.open(image_path)
	if size is not None:
		data_lowlight = data_lowlight.resize((size, size), Image.ANTIALIAS)
	data_lowlight = (np.asarray(data_lowlight)/255.0) 
	
	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0) 

	light_map = unet_model(data_lowlight)
	enhanced_image = torch.clamp((data_lowlight / light_map), 0, 1)
	
	output_path = image_path.replace(image_list_path, result_list_path)
	output_path = output_path.replace('.JPG','.png')
	if not os.path.exists(result_list_path): 
		os.makedirs(result_list_path)

	torchvision.utils.save_image(enhanced_image, output_path)


def inference(image_list_path, result_list_path, unet_model, size=None):
    with torch.no_grad():

        file_list = os.listdir(image_list_path)
        
        print("Inferencing...")
        for file_name in tqdm(file_list):
            image_path = os.path.join(image_list_path, file_name)
            enhance_image(image_path, image_list_path, result_list_path, unet_model, size=size)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', help='directory of input folder', default='./input/')
	parser.add_argument('-o', '--output', help='directory of output folder', default='./inference_result/')
	parser.add_argument('-c', '--unet_pretrain_dir', help='enhancement model pretrained ckpt path', default='./pretrained_models/enhancement_model.pth')

	args = parser.parse_args()
	args.load_pretrain_unet = True

	U_net = load_enhancement_model(args, padding_mode='reflect')

	inference(args.input, args.output, U_net)


