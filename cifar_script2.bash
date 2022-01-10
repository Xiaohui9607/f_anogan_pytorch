

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class airplane --name cifar10_3_0 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class airplane --name cifar10_3_0 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class automobile --name cifar10_3_1 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class automobile --name cifar10_3_1 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class bird --name cifar10_3_2 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class bird --name cifar10_3_2 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class cat --name cifar10_3_3 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class cat --name cifar10_3_3 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class deer --name cifar10_3_4 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class deer --name cifar10_3_4 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class dog --name cifar10_3_5 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class dog --name cifar10_3_5 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class frog --name cifar10_3_6 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class frog --name cifar10_3_6 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class horse --name cifar10_3_7 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class horse --name cifar10_3_7 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class ship --name cifar10_3_8 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class ship --name cifar10_3_8 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class truck --name cifar10_3_9 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class truck --name cifar10_3_9 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class airplane --name cifar10_4_0 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class airplane --name cifar10_4_0 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class automobile --name cifar10_4_1 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class automobile --name cifar10_4_1 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class bird --name cifar10_4_2 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class bird --name cifar10_4_2 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class cat --name cifar10_4_3 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class cat --name cifar10_4_3 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class deer --name cifar10_4_4 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class deer --name cifar10_4_4 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class dog --name cifar10_4_5 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class dog --name cifar10_4_5 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class frog --name cifar10_4_6 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class frog --name cifar10_4_6 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class horse --name cifar10_4_7 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class horse --name cifar10_4_7 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class ship --name cifar10_4_8 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class ship --name cifar10_4_8 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class truck --name cifar10_4_9 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class truck --name cifar10_4_9 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt
