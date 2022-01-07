

python ./gan_training.py --n_epochs 50 --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 0 --name cifar_0_0 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 0 --name cifar_0_0 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 1 --name cifar_0_1 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 1 --name cifar_0_1 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 2 --name cifar_0_2 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 2 --name cifar_0_2 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 3 --name cifar_0_3 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 3 --name cifar_0_3 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 4 --name cifar_0_4 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 4 --name cifar_0_4 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 5 --name cifar_0_5 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 5 --name cifar_0_5 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 6 --name cifar_0_6 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 6 --name cifar_0_6 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 7 --name cifar_0_7 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 7 --name cifar_0_7 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 8 --name cifar_0_8 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 8 --name cifar_0_8 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 9 --name cifar_0_9 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 9 --name cifar_0_9 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 0 --name cifar_1_0 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 0 --name cifar_1_0 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 1 --name cifar_1_1 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 1 --name cifar_1_1 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 2 --name cifar_1_2 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 2 --name cifar_1_2 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 3 --name cifar_1_3 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 3 --name cifar_1_3 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 4 --name cifar_1_4 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 4 --name cifar_1_4 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 5 --name cifar_1_5 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 5 --name cifar_1_5 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 6 --name cifar_1_6 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 6 --name cifar_1_6 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 7 --name cifar_1_7 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 7 --name cifar_1_7 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 8 --name cifar_1_8 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 8 --name cifar_1_8 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 9 --name cifar_1_9 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 9 --name cifar_1_9 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 0 --name cifar_2_0 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 0 --name cifar_2_0 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 1 --name cifar_2_1 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 1 --name cifar_2_1 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 2 --name cifar_2_2 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 2 --name cifar_2_2 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 3 --name cifar_2_3 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 3 --name cifar_2_3 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 4 --name cifar_2_4 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 4 --name cifar_2_4 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 5 --name cifar_2_5 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 5 --name cifar_2_5 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 6 --name cifar_2_6 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 6 --name cifar_2_6 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 7 --name cifar_2_7 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 7 --name cifar_2_7 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 8 --name cifar_2_8 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 8 --name cifar_2_8 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 9 --name cifar_2_9 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset cifar --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 9 --name cifar_2_9 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt
