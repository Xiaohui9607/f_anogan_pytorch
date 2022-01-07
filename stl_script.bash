

python ./gan_training.py --n_epochs 50 --dataset stl --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 0 --name stl_0 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset stl --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 0 --name stl_0 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset stl --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 1 --name stl_1 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset stl --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 1 --name stl_1 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset stl --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 2 --name stl_2 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset stl --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 2 --name stl_2 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset stl --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 3 --name stl_3 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset stl --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 3 --name stl_3 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset stl --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 4 --name stl_4 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset stl --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 4 --name stl_4 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset stl --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 5 --name stl_5 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset stl --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 5 --name stl_5 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset stl --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 6 --name stl_6 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset stl --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 6 --name stl_6 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset stl --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 7 --name stl_7 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset stl --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 7 --name stl_7 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset stl --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 8 --name stl_8 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset stl --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 8 --name stl_8 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt

python ./gan_training.py --n_epochs 50 --dataset stl --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 9 --name stl_9 --device cuda --out ckpts
python ./enc_training.py --n_epochs 5  --dataset stl --latent_dim 128 --img_size 32 --channels 3 --abnormal_class 9 --name stl_9 --device cuda --out ckpts --G_path ckpts/G_epoch49.pt --D_path ckpts/D_epoch49.pt
