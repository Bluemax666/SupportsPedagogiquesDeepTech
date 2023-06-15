"""
@author: Maxime FAURENT
"""

import cv2
import numpy as np
import os

size = 64
#dossier où les images seront enregistrées
dataset_folder = "images" #***

idx = 0
#permet de capturer la webcam
cap = cv2.VideoCapture(0)
os.chdir(dataset_folder)
#tant que le programme est actif
while cap.isOpened():
    success, image = cap.read()
    if not success:
       pass
    
    #les images sont réduites à une taille de 64x64 pixels
    image = cv2.resize(image, (size, size))
    show_img = cv2.resize(image, (256, 256), cv2.INTER_AREA)
    cv2.imshow("image", show_img)
    
    k = cv2.waitKey(10)
    if  k & 0xFF == 27 or k == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
    
    #si on appuye sur "r" les images sont sauvegardées
    elif k == ord('r'):
        cv2.imwrite(f"train_{idx}.png", image)
        idx += 1
        
