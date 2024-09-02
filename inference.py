import torch
import os
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from enhancement_model import load_enhancement_model


def create_transform(size=None):
    transform_list = [
        transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),  # Ensure image is in RGB
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.cuda())
    ]
    if size:
        transform_list.insert(1, transforms.Resize((size, size), Image.ANTIALIAS))  # Adjusted index for Resize
    return transforms.Compose(transform_list)


def enhance_image(image_path, output_path, model, transform):
    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0)
    print(img_tensor.shape)
    light_map = model(img_tensor)
    enhanced = torch.clamp(img_tensor / light_map, 0, 1)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_image(enhanced, output_path)


def process_directory(input_dir, output_dir, model, size=None):
    transform = create_transform(size)
    with torch.no_grad():
        for filename in tqdm(os.listdir(input_dir), desc="Enhancing images"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace('.JPG', '.png'))
            enhance_image(input_path, output_path, model, transform)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default="./configs/inference/inference.yaml")
    args = parser.parse_args()
    
    config = OmegaConf.load(args.cfg)
    model = load_enhancement_model(config, padding_mode='reflect')
    process_directory(config.data.input, config.data.output, model)
