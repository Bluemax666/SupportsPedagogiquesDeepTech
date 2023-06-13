""" Training VAE """
import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from model_VAE_64 import VAE
import numpy as np
import cv2
import os


images_size = 64 #resolution des images
latent_size = 1

folder_name = "images"
output_model_name = "model64.torch"
normalize_data = True #divide pixel values by 255
test_data_proportion = 0.2

batch_size = 64
epochs = 20
beta_factor = 1

cuda = torch.cuda.is_available()
torch.manual_seed(123)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if cuda else "cpu")
print("device used : ", device)

os.chdir("images")
imgs = []
for img_path in os.listdir():
    img = cv2.imread(img_path)
    img = cv2.resize(img, (images_size, images_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgs.append(img)
    
data = np.array(imgs)
print("data shape : ", data.shape)
n_samples = data.shape[0]
n_test_samples = int(n_samples*test_data_proportion)

train_data = np.array([i for i in  data[:-n_test_samples]], dtype=np.float32).reshape(-1,3,64,64)
test_data = np.array([i for i in  data[-n_test_samples:]], dtype=np.float32).reshape(-1,3,64,64)

if normalize_data:
    train_data /= 255
    test_data /= 255

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=True)

model = VAE(3, latent_size).to(device)
optimizer = optim.Adam(model.parameters())
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logsigma):
    """ VAE loss function """
    BCE = F.mse_loss(recon_x, x, size_average=False)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return BCE + beta_factor * KLD


def train(epoch):
    """ One training epoch """
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


def test():
    """ One test epoch """
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


for epoch in range(1, epochs + 1): 
    train(epoch)
    test_loss = test()

os.chdir("..")    
torch.save(model.state_dict(), output_model_name)
