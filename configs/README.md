Here you will find descriptions of parameters of configs for training and testing of all the models.

### Training configs

#### train/clip-lit.yaml and train/clip-lit-latent.yaml parameters:

- ```exp```:
    - ```mode```: either clip-lit or clip-lit-latent
    - ```exp_name```: name of experiment. Will affect the name of folder with experiment checkpoints
    - ```save_dir```: directory of storing experiment's checkpoints

- ```data```:
    - ```backlit_images_path```: path to the training backlit images. e.g. ./data/BAID/train/backlit/
    - ```welllit_images_path```: path to the training well-lit images. e.g. ./data/BAID/train/well-lit/

- ```train```:
    - ```num_epochs```: number of epochs to run training for. One epoch is 2100 iterations of enhancement model (UNet) training and 1000 iteratuions of - prommpts/latent vectors fine-tuning. Hyperparameters 2100 and 1000 are taken from original [CLIP-LIT](hhttps://github.com/ZhexinLiang/CLIP-LIT) code
    - ```num_workers```: number of workers for dataloader

- ```unet_model```:
    - ```thr_loss```: upper limit on loss value for UNet. If UNet loss is higher than thr_loss at the end of epoch, 60 more iterationas of UNet training are added. This is repeated until UNet loss becomes less than thr_loss;
    - ```lr```: learning rate for the optimizer of the enhancement model;
    - ```weight_decay```: weight decay for the optimizer of the enhancement model;
    - ```num_reconstruction_iters```: at the beginning of training the enhancement model is trained using only reconstruction loss for num_reconstruction_iters number of iterations;
    - ```batch_size```: batch size for the training data of the enhancement model;
    - ```display_iter```: the frequency to display the training log during the enhancement model training;
    - ```load_pretrain```: whether to resume training of the enhancement model from existing checkpoint. Note that is load_pretrain=True you need to specify num_reconstruction_iters=0;
    - ```pretrain_dir```: if load_pretrain=True, the enhancement model checkpoint will be loaded from pretrain_dir;

- ```guidance_model```:
    - ```weight_decay```: weight decay for the optimizer of the guidance model;
    - ```length_prompt```: if exp.mode=clip-lit, length_prompt specifies the number of tokens in a learned pseudo-prompt.
    - ```thr_loss```: upper limit on loss value for guidance model. If guidance model loss is higher than thr_loss at the end of epoch, 100 more iterationas of guidance model training are added. This is repeated until the loss becomes less than thr_loss;
    - ```lr```: learning rate for the optimizer of the guidance model;
    - ```num_pretrain_iters```: number of initial training iterations of guidance model. Before training the enhancement model, we need to train good enough guidance for it, so num_pretrain_iters is set to 8000;
    - ```batch_size```: batch size for the training data of the guidance model;
    - ```display_iter```: the frequency to display the training log during the guidance model training;
    - ```load_pretrain```: whether to resume training of the guidance model from existing checkpoint. Note that is load_pretrain=True you need to specify num_pretrain_iters=0;
    - ```pretrain_dir```: if load_pretrain=True, the guidance model checkpoint will be loaded from pretrain_dir;


#### train/rave.yaml parameters:

- ```exp```:
  - ```mode```: rave (the only available option)
  - ```exp_name```: name of experiment. Will affect the name of folder with experiment checkpoints
  - ```save_dir```: directory of storing experiment's checkpoints

- ```data```:
  - ```backlit_images_path```: path to the training backlit images. e.g. ./data/BAID/train/backlit/
  - ```welllit_images_path```: path to the training well-lit images. e.g. ./data/BAID/train/well-lit/
    
- ```train```:
  - ```num_epochs```: number of epochs to run training for. One epoch is 2100 iterations of enhancement model (UNet) training. Number 2100 was chosen for the consistency with training of clip-lit and clip-litl-latent models;
  - ```num_workers```: number of workers for dataloader

- ```unet_model```:
  - ```thr_loss```: upper limit on loss value for UNet. If UNet loss is higher than thr_loss at the end of epoch, 60 more iterationas of UNet training are added. This is repeated until UNet loss becomes less than thr_loss;
  - ```lr```: learning rate for the optimizer of the enhancement model;
  - ```weight_decay```: weight decay for the optimizer of the enhancement model;
  - ```num_reconstruction_iters```: at the beginning of training the enhancement model is trained using only reconstruction loss for num_reconstruction_iters number of iterations;
  - ```batch_size```: batch size for the training data of the enhancement model;
  - ```display_iter```: the frequency to display the training log during the enhancement model training;
  - ```load_pretrain```: whether to resume training of the enhancement model from existing checkpoint. Note that is load_pretrain=True you need to specify num_reconstruction_iters=0;
  - ```pretrain_dir```: if load_pretrain=True, the enhancement model checkpoint will be loaded from pretrain_dir;

- ```guidance```:
  - ```remove_first_n_tokens```: used for shifting the residual guidance vector before training (see rave-shifted in paper). The guidance vector will be shifted by additional residual vector formed using remove_first_n_tokens number of tokens with the most and least closest CLIP embeddings to the initial residual vector.


### Inference configs

#### inference.yaml:

- ```unet_model```:
  - ```load_pretrain```: True (should be true for inference)
  - ```pretrain_dir```: a path to the enhancement model checkpoint for inference. 

- ```data```:
  - ```input```: a path to foler containing backlit images for inference
  - ```output```: a path to the folder where resulting well-lit images will be stored


#### metrics.yaml:

- ```data```:
  - ```gt_images_path```: path to the folder with ground-truth well-lit images
  - ```enhanced_images_path```: path to the folder with enhanced images, for which we need to compute metrics


