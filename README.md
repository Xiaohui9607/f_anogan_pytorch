f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks
===================================================================


Code for reproducing **f-AnoGAN** training and anomaly scoring presented in [*"f-AnoGAN: Fast Unsupervised Anomaly Detection with Generative Adversarial Networks"*](https://www.sciencedirect.com/science/article/abs/pii/S1361841518302640), implemented in Pytorch.



## Requirements

scikit-learn==0.21.2<br>
torch==1.4.0<br>
torchvision==0.5.0


## Run
### Generative adversarial training
```bash
$ python ./gan_training.py     \
          --n_epochs   50      \ # number of epochs of training
          --dataset    mnist   \ # folder | cifar10 | mnist
          --latent_dim 128     \ # dimensionality of the latent space
          --img_size   32      \ # size of each image dimension
          --channels   1       \ # number of image channels
          --abnormal_class 0   \ # Anomaly class idx for mnist and cifar datasets
          --device     cuda    \ # device: cuda | cpu
          --out        ckpts \ # checkpoint directory
```
### Encoder training
```bash
$ python ./enc_training.py     \
          --n_epochs   50      \ # number of epochs of training
          --dataset    mnist   \ # folder | cifar10 | mnist
          --latent_dim 128     \ # dimensionality of the latent space
          --img_size   32      \ # size of each image dimension
          --channels   1       \ # number of image channels
          --abnormal_class 0   \ # Anomaly class idx for mnist and cifar datasets
          --device     cuda    \ # device: cuda | cpu
          --out        checkpoints \ # checkpoint directory
          --G_path     ckpts/G_epoch49.pt  \ # path to trained state dict of generator
          --D_path     ckpts/D_epoch49.pt  \ # path to trained state dict of discriminator
```
