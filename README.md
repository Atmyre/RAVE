# RAVE: Residual Vector Embedding for CLIP-Guided Backlit Image Enhancement

<div>
    :star: <strong>Accepted to ECCV 2024</strong>
</div>

<div>
    <a href='https://atmyre.github.io' target='_blank'>Tatiana Gaintseva</a>&emsp;
    <a href='https://profiles.ucl.ac.uk/95169-martin-benning' target='_blank'>Martin Benning</a>&emsp;
    <a href='https://www.qmul.ac.uk/eecs/people/profiles/slabaughgreg.html' target='_blank'>Gregory Slabaugh</a>
</div>

<div>
    <h4 align="center">
        <a href="" target='_blank'>[Project Page]</a> •
        <a href="https://arxiv.org/abs/2404.01889" target='_blank'>[arXiv]</a> •
    </h4>
</div>

[Project Page]() [[Paper]](https://arxiv.org/abs/2404.01889) 

In this paper we propose a novel modification of Contrastive Language-Image Pre-Training (CLIP) guidance for the task of unsupervised backlit image enhancement. Our work builds on the state-of-the-art CLIP-LIT approach, which learns a prompt pair by constraining the text-image similarity between a prompt (negative/positive sample) and a corresponding image (backlit image/well-lit image) in the CLIP embedding space. Learned prompts then guide an image enhancement network. Based on the CLIP-LIT framework, we propose two novel methods for CLIP guidance. First, we show that instead of tuning prompts in the space of text embeddings, it is possible to directly tune their embeddings in the latent space without any loss in quality. This accelerates training and potentially enables the use of additional encoders that do not have a text encoder. Second, we propose a novel approach that does not require any prompt tuning. Instead, based on CLIP embeddings of backlit and well-lit images from training data, we compute the residual vector in the embedding space as a simple difference between the mean embeddings of the well-lit and backlit images. This vector then guides the enhancement network during training, pushing a backlit image towards the space of well-lit images. This approach further dramatically reduces training time, stabilizes training and produces high quality enhanced images without artifacts, both in supervised and unsupervised training regimes. Additionally, we show that residual vectors can be interpreted, revealing biases in training data, and thereby enabling potential bias correction. 

![Results](assets/results.png)


## Approach

![RAVE](assets/rave.png)

In RAVE we exploit arithmetic defined in the CLIP latent space. Using well-lit and backlit training data, we construct a residual vector, which will then be used for enhancement model guidance. This is a vector that points in a direction moving from backlit images to well-lit images in the CLIP embedding space. We then use this vector as guidance for the image enhancement model during training. This will train the image enhancement model to produce images with CLIP latent vectors that are close to the CLIP latent vectors of well-lit training images. 

## Updates
- **2024.08.26**: Code for training and testing as well as model checkpoints are publicly available now.


## Usage

### Training and Testing Data:
Training and testing data can be downloaded from:
- [BAID dataset (train and test parts)](https://drive.google.com/drive/folders/14_OvT17bfoN-JEH0GTFD6bnKzjKmag_l?usp=sharing);
- [DIV2K images (well-lit images used instead of well-lit images from BAID for training models in unpaired setting)]https://drive.google.com/drive/folders/1PbWzGzxLF0OJMyA7zMj_Dd00UB_ulhb4?usp=sharing);
- [LOL-v1 dataset for low-light image enhancement task (see supplementary material of RAVE paper for results on this data)](https://drive.google.com/drive/folders/1ewmaFEVjKmzS8fLisSA7q_t1RbBdz6He?usp=sharing).


### Run Training:

#### CLIP-LIT

Train from scratch:

```
python train.py \
 -b path_to_backlit_train_images  \
 -r path_to_well_lit_train_images \
 --mode clip-lit                  \
 --exp_name clip_lit              \
 --train_lr 0.00002               \
 --prompt_lr 0.000005             \
 --num_reconstruction_iters 1000  \
 --num_clip_pretrained_iters 8000 \
 --load_pretrain_unet False       \
 --load_pretrain_guidance False   \
 ```

If you have pre-trained Unet and/or guidance model checkpoints, you can resume training as follows:

```
 python train.py \
 -b path_to_backlit_train_images  \
 -r path_to_well_lit_train_images \
 --mode clip-lit                  \
 --exp_name clip_lit              \
 --train_lr 0.00002               \
 --prompt_lr 0.000005             \
 --num_reconstruction_iters 1000  \
 --num_clip_pretrained_iters 8000 \
 --load_pretrain_unet True        \
 --unet_pretrain_dir path_to_unet_ckpt \
 --load_pretrain_guidance True    \
 --guidance_pretrain_dir path_to_guidance_model_ckpt \
 ```

 #### CLIP-LIT-Latent

Train from scratch:

```
python train.py \
 -b path_to_backlit_train_images  \
 -r path_to_well_lit_train_images \
 --mode clip-lit-latent           \
 --exp_name clip_lit-latent       \
 --train_lr 0.00002               \
 --prompt_lr 0.0005               \
 --num_reconstruction_iters 1000  \
 --num_clip_pretrained_iters 8000 \
 --load_pretrain_unet False       \
 --load_pretrain_guidance False   \
 ```

If you have pre-trained Unet and/or guidance model checkpoints, you can resume training as follows:

```
 python train.py \
 -b path_to_backlit_train_images  \
 -r path_to_well_lit_train_images \
 --mode clip-lit-latent           \
 --exp_name clip_lit              \
 --train_lr 0.00002               \
 --prompt_lr 0.0005               \
 --num_reconstruction_iters 1000  \
 --num_clip_pretrained_iters 8000 \
 --load_pretrain_unet True        \
 --unet_pretrain_dir path_to_unet_ckpt \
 --load_pretrain_guidance True    \
 --guidance_pretrain_dir path_to_guidance_model_ckpt \
 ```


 #### RAVE

Train from scratch without shifting the residual vector:

```
python train.py \
 -b path_to_backlit_train_images  \
 -r path_to_well_lit_train_images \
 --exp_name rave                  \
 --train_lr 0.00002               \
 --num_reconstruction_iters 1000  \
 --load_pretrain_unet False       \
 ```

To train RAVE with shifted residual by n tokens add the following instruction:
 ```
 --remove_first_n_tokens n         \
 ```


If you have pre-trained Unet, you can resume training as follows:

```
python train.py \
 -b path_to_backlit_train_images  \
 -r path_to_well_lit_train_images \
 --exp_name rave                  \
 --train_lr 0.00002               \
 --num_reconstruction_iters 1000  \
 --load_pretrain_unet False       \
 --unet_pretrain_dir path_to_unet_ckpt \
 ```


### Inferencing and Testing:

#### Pretrained checkpoints

Pretrained checkpoints for all the models are stored in 'pretrained_models' dir.

Models trained on paired data:
- CLIP-LIT: clip_lit_paired.pth;
- CLIP-LIT-Latent: clip_lit_latent_paired.pth;
- RAVE: rave_paired.pth.

Models trained on unpaired data:
- CLIP-LIT: clip_lit_unpaired.pth;
- CLIP-LIT-Latent: clip_lit_latent_unpaired.pth;
- RAVE without shifting the residual: rave_unpaired.pth;
- RAVE with shifting the residual by 15 tokens: rave_unpaired_shifted.pth.

#### Inferencing
To run trained model on backlit images use the following command:

```
python inference.py \
 --input path_to_input_images \
 --output path_to_output_dir  \
 --unet_pretrain_dir path_to_pretrained_ckpt
```

#### Testing (computing metrics)

To compute metrics (SSIM, PSNR, LPIPS, FID) on bunch of backlit and corresponding enhanced images, use the following command:

```
python compute_metrics.py \
 --gt_images_path path_to_ground_truth_well_lit_images \
 --enhanced_images_path path_to_ground_enhanced_images  
```

## Citation
If you find our work useful, please consider citing the paper:
```
@misc{gaintseva2024raveresidualvectorembedding,
      title={RAVE: Residual Vector Embedding for CLIP-Guided Backlit Image Enhancement}, 
      author={Tatiana Gaintseva and Martin Benning and Gregory Slabaugh},
      year={2024},
      eprint={2404.01889},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.01889}, 
}
```

### Contacts
Please feel free to reach out at `t.gaintseva@qmul.ac.uk`. 