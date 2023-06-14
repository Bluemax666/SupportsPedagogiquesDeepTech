"""
@author: Maxime FAURENT
"""

import cv2
import numpy as np
import os

size = 64

idx = 0
cap = cv2.VideoCapture(0)
os.chdir("images")
while cap.isOpened():
    success, image = cap.read()
    if not success:
       pass
   
    image = cv2.resize(image, (256, 256))
    cv2.imshow("image", image)
    
    k = cv2.waitKey(10)
    if  k & 0xFF == 27 or k == ord('r'):
        cap.release()
        cv2.destroyAllWindows()
        break
    
    elif k == ord('s'):
        cv2.imwrite(f"train_{idx}.png", image)
        
        idx += 1
        
