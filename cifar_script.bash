

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class airplane --name cifar10_0_0 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class airplane --name cifar10_0_0 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class automobile --name cifar10_0_1 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class automobile --name cifar10_0_1 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class bird --name cifar10_0_2 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class bird --name cifar10_0_2 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class cat --name cifar10_0_3 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class cat --name cifar10_0_3 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class deer --name cifar10_0_4 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class deer --name cifar10_0_4 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class dog --name cifar10_0_5 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class dog --name cifar10_0_5 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class frog --name cifar10_0_6 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class frog --name cifar10_0_6 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class horse --name cifar10_0_7 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class horse --name cifar10_0_7 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class ship --name cifar10_0_8 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class ship --name cifar10_0_8 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class truck --name cifar10_0_9 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class truck --name cifar10_0_9 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class airplane --name cifar10_1_0 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class airplane --name cifar10_1_0 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class automobile --name cifar10_1_1 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class automobile --name cifar10_1_1 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class bird --name cifar10_1_2 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class bird --name cifar10_1_2 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class cat --name cifar10_1_3 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class cat --name cifar10_1_3 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class deer --name cifar10_1_4 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class deer --name cifar10_1_4 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class dog --name cifar10_1_5 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class dog --name cifar10_1_5 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class frog --name cifar10_1_6 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class frog --name cifar10_1_6 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class horse --name cifar10_1_7 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class horse --name cifar10_1_7 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class ship --name cifar10_1_8 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class ship --name cifar10_1_8 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class truck --name cifar10_1_9 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class truck --name cifar10_1_9 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class airplane --name cifar10_2_0 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class airplane --name cifar10_2_0 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class automobile --name cifar10_2_1 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class automobile --name cifar10_2_1 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class bird --name cifar10_2_2 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class bird --name cifar10_2_2 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class cat --name cifar10_2_3 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class cat --name cifar10_2_3 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class deer --name cifar10_2_4 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class deer --name cifar10_2_4 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class dog --name cifar10_2_5 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class dog --name cifar10_2_5 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class frog --name cifar10_2_6 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class frog --name cifar10_2_6 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class horse --name cifar10_2_7 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class horse --name cifar10_2_7 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class ship --name cifar10_2_8 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class ship --name cifar10_2_8 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class truck --name cifar10_2_9 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar10 --latent_dim 128 --img_size 32 --channels 3 --abnormal_class truck --name cifar10_2_9 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt
