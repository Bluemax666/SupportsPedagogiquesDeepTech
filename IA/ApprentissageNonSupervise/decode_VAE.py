# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 17:59:50 2020

@author: Maxime FAURENT
"""

import cv2
import numpy as np
import tkinter as tk
from model_VAE_64 import VAE
import torch


model_name = "model64.torch"
images_size = 64 
latent_size = 1

    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = VAE(3, latent_size).to(device)
vae.load_state_dict(torch.load(model_name), strict=False)
        
class Image_decoder:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("imageDecoder")
        for i in range(latent_size):
            setattr(self, "slider{}".format(i), tk.DoubleVar())
            tk.Scale(self.root, from_=-100,to=100,length=200,orient=tk.HORIZONTAL,
                     variable=getattr(self, "slider{}".format(i)), command=self.show_img).pack()
            
        
        self.root.mainloop()
        
    
    def show_img(self, args):
        k = 20
        z_latent = []
        for i in range(latent_size):
            z_latent.append(getattr(self, "slider{}".format(i)).get())
        z_latent = np.array(z_latent, dtype=np.float32) / k
        z_latent = torch.Tensor([z_latent]).view(-1, latent_size).to(device)
        image_tensor = vae.decode(z_latent)
        image = image_tensor.cpu().detach().numpy()[0]
        image = np.reshape(image,(64,64,3))
        image = cv2.resize(image,(256,256),interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        
        cv2.imshow("decoded image",image)
        
        cv2.waitKey(10)

            
d = Image_decoder()



