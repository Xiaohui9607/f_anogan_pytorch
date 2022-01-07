"""
LOAD DATA from file.
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import os
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, ImageFolder, STL10

from dataloader.datasets import get_cifar_anomaly_dataset
from dataloader.datasets import get_mnist_anomaly_dataset
# from dataloader.kdd_dataset import get_loader

class Data:
    """ Dataloader containing train and valid sets.
    """
    def __init__(self, train, valid):
        self.train = train
        self.valid = valid

##
def load_data(opt):
    """ Load Data

    Args:
        opt ([type]): Argument Parser

    Raises:
        IOError: Cannot Load Dataset

    Returns:
        [type]: dataloader
    """

    ##
    # LOAD DATA SET
    if opt.dataroot == '':
        opt.dataroot = './data/{}'.format(opt.dataset)

    ## CIFAR
    if opt.dataset in ['cifar10']:
        transform = transforms.Compose([transforms.Resize(opt.img_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_ds = CIFAR10(root='./data', train=True, download=True, transform=transform)
        valid_ds = CIFAR10(root='./data', train=False, download=True, transform=transform)
        train_ds, valid_ds = get_cifar_anomaly_dataset(train_ds, valid_ds, train_ds.class_to_idx[opt.abnormal_class])

    ## MNIST
    elif opt.dataset in ['mnist']:
        transform = transforms.Compose([transforms.Resize(opt.img_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])

        train_ds = MNIST(root='./data', train=True, download=True, transform=transform)
        valid_ds = MNIST(root='./data', train=False, download=True, transform=transform)
        train_ds, valid_ds = get_mnist_anomaly_dataset(train_ds, valid_ds, int(opt.abnormal_class))

    #STL
    elif opt.dataset in ['stl']:
        transform = transforms.Compose([transforms.Resize(opt.img_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])

        train_ds = STL10(root='./data', split='train', download=True, transform=transform)
        valid_ds = STL10(root='./data', split='test', download=True, transform=transform)
        train_ds, valid_ds = get_stl_anomaly_dataset(train_ds, valid_ds, int(opt.abnormal_class))

    # FOLDER
    elif opt.dataset in ['OCT']:
        # TODO: fix the OCT dataset into the dataloader and return
        def white_noise(x):
            x = x + torch.randn(x.shape)*0.01
            x[x > 1] = 1
            x[x < 0] = 0
            return x
        transform_train = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(opt.img_size),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.CenterCrop(opt.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(white_noise)])

        transform_test = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(opt.img_size),
            transforms.CenterCrop(opt.img_size),
            transforms.ToTensor(),])

        train_ds = ImageFolder(os.path.join(opt.dataroot, 'train'), transform_train)
        valid_ds = ImageFolder(os.path.join(opt.dataroot, 'test'), transform_test)

    elif opt.dataset in ['KDD99']:
        train_ds = KDD_dataset(opt, mode='train')
        valid_ds = KDD_dataset(opt, mode='test')

    else:
        raise NotImplementedError

    ## DATALOADER
    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batch_size, shuffle=False, drop_last=False)

    return Data(train_dl, valid_dl)
