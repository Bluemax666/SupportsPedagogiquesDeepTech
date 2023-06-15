# -*- coding: utf-8 -*-
"""
@author: Maxime FAURENT
"""

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from model_VAE_64 import VAE
import numpy as np
import cv2
import os

#résolution des images
images_size = 64
#définit le nombre de valeurs qui vont réprensenter une image après la compression
latent_size = 4  #***1-32***

#nom du dossier contenant les image utilisées pour l'entraînement
dataset_folder= "images" #***
#nom du modèle une fois sauvegardé
output_model_name = "model64.torch" #***
#proportion des exemple qui ne seront utilisés que pour tester la performance du modèle
test_data_proportion = 0.2 #***0.05-0.5***
#divise la valeur des pixels par 255 avant de les passer dans le réseau de neurones
normalize_data = True 
#nombre d'éléments qui sont passés en même temps dans le modèle pendant l'entrainement
batch_size = 64  #***4-256***
#nombre de fois que l'apprentissage va être fait une fois pour tout les image du dataset
epochs = 20 #***5-200*** pour changer le temps d'entraînement il faut varier ce paramètre


#si il y a une carte graphique compatible et que PyTorch a été installé avec CUDA
cuda = torch.cuda.is_available()
torch.manual_seed(123)
torch.backends.cudnn.benchmark = True

#affiche si on utilise le processeur (cpu) ou la carte graphique (cuda)
device = torch.device("cuda" if cuda else "cpu")
print("device used : ", device)

#on charge toutes les images
os.chdir(dataset_folder)
imgs = []
for img_path in os.listdir():
    img = cv2.imread(img_path)
    img = cv2.resize(img, (images_size, images_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgs.append(img)

#On tranforme les images et on affiche le nombre d'images    
data = np.array(imgs)
print("data shape : ", data.shape)
n_samples = data.shape[0]
n_test_samples = int(n_samples*test_data_proportion)

#on sépare le dataset en un train dataset et un test dataset
train_data = np.array([i for i in  data[:-n_test_samples]], dtype=np.float32).reshape(-1,3,64,64)
test_data = np.array([i for i in  data[-n_test_samples:]], dtype=np.float32).reshape(-1,3,64,64)

if normalize_data:
    train_data /= 255
    test_data /= 255

#on met les données sous la forme d'un dataloader pour les passer dans le modèle
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=True)

#on charge le modele définit dans le fichier model_VAE_64.py
model = VAE(3, latent_size).to(device)
optimizer = optim.Adam(model.parameters())

#fonction qui définit l'erreur du model
def loss_function(recon_x, x, mu, logsigma):
    beta_factor = 1
    BCE = F.mse_loss(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return BCE + beta_factor * KLD

#entraîne le modèle pendant une epoch
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 40 == 39:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))

#test les performances du modèle
def test():
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

#boucle d'entraînemenent
for epoch in range(1, epochs + 1): 
    train(epoch)
    test_loss = test()

#les poids du modèle sont sauvegardés 
os.chdir("..")    
torch.save(model.state_dict(), output_model_name)
