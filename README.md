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
$ python ./gan_training.py --n_epochs   50 --dataset    mnist --latent_dim 128 --img_size   32 --channels   1 --abnormal_class 0 --device     cuda --out        ckpts
```
### Encoder training
```bash
$ python ./enc_training.py --n_epochs   5 --dataset    mnist --latent_dim 128 --img_size   32 --channels   1 --abnormal_class 0 --device     cuda --out        ckpts --G_path     ckpts/G_epoch49.pt --D_path     ckpts/D_epoch49.pt
```
