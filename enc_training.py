import os
import argparse
import torch
from torchvision import datasets, transforms
from networks import Generator, Discriminator, Encoder
from torch.autograd import Variable
from torch.nn.functional import mse_loss
from dataloader.dataloader import load_data
from sklearn.metrics import roc_curve, auc
import random


os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--print_interval", type=int, default=100, help="interval of loss printing")
parser.add_argument("--dataroot", default="", help="path to dataset")
parser.add_argument("--dataset", default="cifar10", help="folder | cifar10 | mnist")
parser.add_argument("--abnormal_class", default="airplane", help="Anomaly class idx for mnist and cifar datasets")
parser.add_argument("--out", default="ckpts", help="checkpoint directory")
parser.add_argument("--device", default="cuda", help="device: cuda | cpu")
parser.add_argument("--G_path", default="ckpts/G_epoch19.pt", help="path to trained state dict of generator")
parser.add_argument("--D_path", default="ckpts/D_epoch19.pt", help="path to trained state dict of discriminator")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

generator = Generator(dim = 64, zdim=opt.latent_dim, nc=opt.channels)
discriminator = Discriminator(dim = 64, zdim=opt.latent_dim, nc=opt.channels,out_feat=True)
encoder = Encoder(dim = 64, zdim=opt.latent_dim, nc=opt.channels)

generator.load_state_dict(torch.load(opt.G_path))
discriminator.load_state_dict(torch.load(opt.D_path))
generator.to(opt.device)
encoder.to(opt.device)
discriminator.to(opt.device)

encoder.train()
discriminator.train()

dataloader = load_data(opt)

generator.eval()

Tensor = torch.cuda.FloatTensor if torch.cuda.FloatTensor if opt.device == 'cuda' else torch.FloatTensor

optimizer_E = torch.optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

max_auc = 0
for epoch in range(opt.n_epochs):

    # train
    for i, (imgs, _) in enumerate(dataloader.train):
        optimizer_E.zero_grad()
        optimizer_D.zero_grad()

        imgs = imgs.to(opt.device)
        generator.zero_grad()
        z = encoder(imgs)

        fake_imgs = generator(z)
        image_feats = discriminator(imgs)
        recon_feats = discriminator(fake_imgs)

        loss_img = mse_loss(imgs, fake_imgs)
        loss_feat = mse_loss(image_feats, recon_feats)

        e_loss = loss_img + loss_feat

        e_loss.backward()
        optimizer_E.step()
        optimizer_D.step()

        if i % opt.print_interval == 0:

            print(
                "[Epoch %d/%d] [Batch %d/%d] [E loss: %f] [D loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader.train), loss_img.item(), loss_feat.item())
            )

    # validation
    with torch.no_grad():
        scores = torch.empty(
            size=(len(dataloader.valid.dataset),),
            dtype=torch.float32,
            device=opt.device)
        labels = torch.zeros(size=(len(dataloader.valid.dataset),),
                                    dtype=torch.long, device=opt.device)

        for i, (imgs, lbls) in enumerate(dataloader.valid):
            imgs = imgs.to(opt.device)
            lbls = lbls.to(opt.device)

            labels[i*opt.batch_size:(i+1)*opt.batch_size].copy_(lbls)
            emb_query = encoder(imgs)
            fake_imgs = generator(emb_query)

            image_feats  = discriminator(imgs)
            recon_feats = discriminator(fake_imgs)


            image_distance = torch.mean(torch.pow(imgs-fake_imgs, 2), dim=[1,2,3])
            feat_distance = torch.mean(torch.pow(image_feats-recon_feats, 2), dim=1)

            # z_distance = mse_loss(emb_query, emb_fake)
            scores[i*opt.batch_size:(i+1)*opt.batch_size].copy_(feat_distance)

        labels = labels.cpu()
        # scores = torch.mean(scores,)
        scores = scores.cpu().squeeze()

        # True/False Positive Rates.
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        print(roc_auc)
        max_auc = max(roc_auc, max_auc)

print(opt.out, "[auc: ", max_auc,"]")
