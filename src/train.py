'''
Code to train a variational autoencoder in pytorch
https://medium.com/dataseries/variational-autoencoder-with-pytorch-2d359cbf027b
'''

# TODO: store test metrics

import torch
from datetime import datetime
from torchvision import datasets
from torchvision import transforms
import torchvision
from torch.utils.data import random_split
from VAE import VariationalAutoencoder, PrototypeVAE
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# TODO: incorporate elastic deformation data augmentation 
# TODO: save model hyperparameters

# test to do
# TODO: increase VAE complexity
# TODO: increase nmber of parameters

l1, l2, l3 = 0.5, 0.5, 0.5

# initialise tensorboard
model_id = datetime.now().strftime('%d%m%Y%H%M%S')
writer = SummaryWriter(f"../runs/{model_id}")

def train_epoch(vae, device, dataloader, optimizer):

    # set model to train mode
    vae.train()
    train_loss = 0.0

    # iterate over dataloader
    for x, _ in dataloader:
        # move tensor to proper device
        x = x.to(device)
        x_hat = vae(x)

        # evaluate loss
        loss = ((x - x_hat) ** 2).sum() + vae.encoder.kl

        # backwards pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('\t partial train loss (single batch): %f' % (loss.item()), end="\r")
        train_loss += loss.item()

    return train_loss / len(dataloader.dataset)


def train_epoch(vae, device, dataloader, optimizer, ce):

    # set model to train mode
    vae.train()
    train_loss = 0.0
    correct = 0
    total_reconstruction_loss = 0
    total_ce_loss = 0
    total_error_1 = 0
    total_error_2 = 0

    # iterate over dataloader
    for x, y in dataloader:
        # move tensor to proper device
        y = y.to(device)
        x = x.to(device)
        x_hat, y_hat = vae(x)

        prototype_distances = vae.classifier.prototype_distances
        feature_vector_distances = vae.classifier.feature_vector_distances

        # evaluate loss
        reconstruction_loss = ((x - x_hat) ** 2).sum() + vae.encoder.kl
        ce_loss = ce(y_hat, y)
        error_1 = torch.mean(torch.min(feature_vector_distances, dim = 1).values)
        error_2 = torch.mean(torch.min(prototype_distances, dim=1).values)

        # lambda values are all 0.5 in paper
        # spend time optimizing these values
        loss = l1*reconstruction_loss + ce_loss + l2*error_1 + l3*error_2

        # calculate accuracy
        correct += (torch.max(y_hat, 1).indices == y).float().sum()

        # backwards pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('\t partial train loss (single batch): %f' % (loss.item()), end="\r")
        train_loss += loss.item()

        # for metric tracking
        total_reconstruction_loss += reconstruction_loss.item()
        total_ce_loss += ce_loss.item()
        total_error_1 += error_1
        total_error_2 += error_2
    
    # calculate metrics
    avg_loss = train_loss / len(dataloader.dataset)
    avg_reconstruction = total_reconstruction_loss / len(dataloader.dataset)
    avg_ce = total_ce_loss / len(dataloader.dataset)
    avg_error_1 = total_error_1 / len(dataloader.dataset)
    avg_error_2 = total_error_2 / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)

    # write metrics
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Reconstruction/train", avg_reconstruction, epoch)
    writer.add_scalar("CE/train", avg_ce, epoch)
    writer.add_scalar("E1/train", avg_error_1, epoch)
    writer.add_scalar("E2/train", avg_error_2, epoch)
    writer.add_scalar("Accuracy/train", accuracy, epoch)

    return avg_loss, accuracy

def test_epoch(vae, device, dataloader):

    # set to evaluation mode
    vae.eval()
    val_loss = 0.0
    # no need to track gradients
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            x_hat = vae(x)
            loss = ((x - x_hat)**2).sum() + vae.encoder.kl
            val_loss += loss.item()

    return val_loss / len(dataloader.dataset)

def test_epoch(vae, device, dataloader, ce, test_set=False):

    # set to evaluation mode
    vae.eval()
    val_loss = 0.0
    correct = 0
    total_reconstruction_loss = 0
    total_ce_loss = 0
    total_error_1 = 0
    total_error_2 = 0
    # no need to track gradients
    with torch.no_grad():
        for x, y in dataloader:
            y = y.to(device)
            x = x.to(device)
            x_hat, y_hat = vae(x)

            prototype_distances = vae.classifier.prototype_distances
            feature_vector_distances = vae.classifier.feature_vector_distances

            # evaluate loss
            reconstruction_loss = ((x - x_hat) ** 2).sum() + vae.encoder.kl
            ce_loss = ce(y_hat, y)
            error_1 = torch.mean(torch.min(feature_vector_distances, dim = 1).values)
            error_2 = torch.mean(torch.min(prototype_distances, dim=1).values)

            # lambda values are all 0.5 in paper
            # spend time optimizing these values
            loss = l1*reconstruction_loss + ce_loss + l2*error_1 + l3*error_2

            # calculate accuracy
            correct += (torch.max(y_hat, 1).indices == y).float().sum()
            
            # for metric tracking
            val_loss += loss.item()
            total_reconstruction_loss += reconstruction_loss.item()
            total_ce_loss += ce_loss.item()
            total_error_1 += error_1
            total_error_2 += error_2
    
    # calculate metrics
    avg_loss = val_loss / len(dataloader.dataset)
    avg_reconstruction = total_reconstruction_loss / len(dataloader.dataset)
    avg_ce = total_ce_loss / len(dataloader.dataset)
    avg_error_1 = total_error_1 / len(dataloader.dataset)
    avg_error_2 = total_error_2 / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)

    if test_set:
        writer.add_scalar("Loss/test", avg_loss)
        writer.add_scalar("Reconstruction/test", avg_reconstruction)
        writer.add_scalar("CE/test", avg_ce)
        writer.add_scalar("E1/test", avg_error_1)
        writer.add_scalar("E2/test", avg_error_2)
        writer.add_scalar("Accuracy/test", accuracy)
    else:
        # write metrics
        writer.add_scalar("Loss/val", avg_loss, epoch)
        writer.add_scalar("Reconstruction/val", avg_reconstruction, epoch)
        writer.add_scalar("CE/val", avg_ce, epoch)
        writer.add_scalar("E1/val", avg_error_1, epoch)
        writer.add_scalar("E2/val", avg_error_2, epoch)
        writer.add_scalar("Accuracy/val", accuracy, epoch)

    return avg_loss, accuracy


def show_image(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    return npimg

# load dataset
data_dir = "../data"

# download MNIST dataset
train_dataset = datasets.MNIST(data_dir, train=True, download=True)
test_dataset  = datasets.MNIST(data_dir, train=False, download=True)

# initialise data to tensor
train_transform = transforms.Compose([
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

# apply transform
train_dataset.transform = train_transform
test_dataset.transform = test_transform

# 80/20 train/val split
n = len(train_dataset)
train_data, val_data = random_split(train_dataset, [int(n-n*0.2), int(n*0.2)])
batch_size = 256

# load data
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# SETUP TRAINING
# initialise model
latent_dims = 4
n_prototypes = 10
# vae = VariationalAutoencoder(latent_dims)
vae = PrototypeVAE(latent_dims, n_prototypes)


# initialise optimizer
lr = 1e-3
optim = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)

# initialise MSE
ce = torch.nn.CrossEntropyLoss()

# initialise device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')
vae.to(device)

# train model
num_epochs = 200

for epoch in range(num_epochs):
    # train_loss = train_epoch(vae, device, train_loader, optim)
    # val_loss = test_epoch(vae, device, valid_loader)
    train_loss, train_acc = train_epoch(vae, device, train_loader, optim, ce)
    val_loss, val_acc = test_epoch(vae, device, valid_loader, ce)
    print('\n EPOCH {}/{} \t train loss {:.3f} \t train acc: {:.3f} \t val loss {:.3f} \t val acc: {:.3f}'.format(epoch + 1, num_epochs,train_loss, train_acc, val_loss, val_acc))

vae.eval()

test_epoch(vae, device, test_loader, ce, True)

with torch.no_grad():

    # sample latent vectors from the normal distribution
    latent = torch.randn(128, latent_dims, device=device)

    # reconstruct images from the latent vectors
    img_recon = vae.decoder(latent)
    img_recon = img_recon.cpu()

    fig, ax = plt.subplots(figsize=(20, 8.5))
    img = show_image(torchvision.utils.make_grid(img_recon.data[:100],10,5))
    writer.add_image("Reconstruction", img)
    plt.show()

# view prototypes
with torch.no_grad():

    prototypes = vae.classifier.prototypes.to(device)
    prototypes = vae.decoder(prototypes)

    prototypes = prototypes.cpu()

    fig, ax = plt.subplots(figsize=(20, 8.5))
    img = show_image(torchvision.utils.make_grid(prototypes.data,5,2))
    writer.add_image("Prototypes", img)
    plt.show()

# close tensorboard writer
writer.flush()
writer.close()

# save model
torch.save(vae.state_dict(), f"../models/{model_id}.pt")