# -*- coding: utf-8 -*-
"""
Created on Mon May  1 01:08:26 2023

@author: maxim
"""

"""
sudo -i
sudo /home/maxime/anaconda3/bin/python /home/maxime/Code/Projet_labo/projet2/hand_detect.py
"""
import cv2
import numpy as np
import torch
from torch import nn
import keyboard
from torch.utils.data import Dataset, DataLoader, TensorDataset
import mediapipe as mp


class_dict = {
"Bonjour" : "b",
"Oui" : "o",
"Non" : "n",
"Rien" : "r" 
}
model_path = "/home/maxime/Code/Projet_labo/projet2/"
model_name = "Base"

# class_dict = {k:k for k in "abcdefghijklmnopqrstuvwxyz"}
# model_name = "Alphabet"

# class_dict = {k:k for k in "azertyuiop"}
# model_name = "numb"

def idx2label(idx):
    if idx==-1:
        return ""
    
    return list(class_dict.keys())[idx]



nb_features = 21*2*2 #21 points x 2 axis x 2 hands  
nb_class = len(class_dict)

X, Y = [], []
batch_size = 4

hidden_dim = 512
model = nn.Sequential(
    nn.Linear(nb_features, hidden_dim),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, nb_class),
    nn.Softmax(dim=1)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

def get_train_and_test_loader(X, Y):
    X = np.array(X, dtype=np.float32).reshape(-1, nb_features)
    Y = np.array(Y)              
    
    dataset = TensorDataset(torch.Tensor(X), torch.LongTensor(Y))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader
    

def train_model(X, Y, epochs=20):
    train_loader, test_loader = get_train_and_test_loader(X, Y)
    for epoch in range(epochs):
        running_train_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            # set optimizer to zero grad to remove previous epoch gradients
            optimizer.zero_grad()
            # forward propagation
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # backward propagation
            loss.backward()
            # optimize
            optimizer.step()
            running_train_loss += loss.item()
        
        running_test_loss = 0.0
        for i, data in enumerate(test_loader, 0):
            with torch.no_grad():
                inputs, labels = data
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_test_loss += loss.item()
        # display statistics
        print(f'[{epoch + 1} loss: {running_train_loss / len(train_loader.dataset):.5f}, \
                                   {running_test_loss / len(test_loader.dataset):.5f}')
        

def predict_label_prob(net_input):
    net_input = net_input.astype(np.float32).reshape(-1, nb_features)
    net_input = torch.Tensor(net_input)
    prediction = model(net_input)[0].detach().numpy()
    return prediction

def get_pred_label_name(prediction):
    pred_label = prediction.argmax(0)
    pred_label_name = idx2label(pred_label)
    return pred_label_name
    

def get_label_from_key():
    label = -1
    for i, key in enumerate(class_dict.values()):
        if keyboard.is_pressed(key):
            label = i
            break
        
    return label
    
def process_landmarks(pts):
    pts = np.array(pts)
    rel_pts = (pts - pts[0]) / np.expand_dims(pts[:,2], axis=1)
    return rel_pts[:,:2].flatten() / 30
    
def add_label_to_image(image, label_name, rec_label):
    label_img = np.zeros((250, image.shape[1], 3), np.uint8)
    cv2.putText(label_img, label_name,
                org=(50,150), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=3, color=(255, 200, 200))
    
    if rec_label != "":
        cv2.putText(label_img, rec_label,
                    org=(350,200), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=2, color=(255, 100, 100))
    
    image = np.vstack((label_img, image))
    return image

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

current_pred = np.zeros(nb_class)

cap = cv2.VideoCapture(0)
frame_idx = 0
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue
    
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        
        left_hand_points = []
        right_hand_points = []
        nb_hands = 0
        rec_label = -1
        if results.multi_hand_landmarks is not None and len(results.multi_hand_landmarks) > 0:
            nb_hands = len(results.multi_hand_landmarks)
            for hand_idx in range(min(nb_hands, 2)):
                handedness = results.multi_handedness[hand_idx].classification[0].label
                for landmark in results.multi_hand_landmarks[hand_idx].landmark:
                    x = landmark.x
                    y = landmark.y
                    z = landmark.z
                    if len(left_hand_points) >= nb_features//4:
                        right_hand_points += [[x, y, z]]
                    
                    elif len(right_hand_points) >= nb_features//4:
                        left_hand_points += [[x, y, z]]
                    
                    elif handedness == "Left":
                        left_hand_points += [[x, y, z]]
                    elif handedness == "Right":
                        right_hand_points += [[x, y, z]]
            
            
            if left_hand_points == [] or right_hand_points == []:
                if left_hand_points != []:
                    net_input = process_landmarks(left_hand_points)
                else:
                    net_input = process_landmarks(right_hand_points)
                    
                net_input = np.concatenate((net_input, np.zeros(int(nb_features/2))))
                    
            else:
                net_input = np.concatenate((process_landmarks(left_hand_points),
                                            process_landmarks(right_hand_points)))
            
            #print(net_input)
            rec_label = get_label_from_key()
            if rec_label != -1 and frame_idx % 2 == 0:
                X.append(net_input)
                Y.append(rec_label)
            else:
                cv2.waitKey(5)
             
                
        if keyboard.is_pressed('t'):
            train_model(X, Y)
            
        elif keyboard.is_pressed('s'):
            torch.save(model.state_dict(), model_path+model_name)
            print(f"model : {model_name} saved")
        
        elif keyboard.is_pressed('l'):
            model.load_state_dict(torch.load(model_path+model_name), strict=False)
            print(f"model : {model_name} loaded")
            
    
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
        # Flip the image horizontally for a selfie-view display.
        image = cv2.flip(image, 1)
        label_name = ""
        if nb_hands > 0:
            prediction = predict_label_prob(net_input)
            current_pred = current_pred*0.8 + prediction*0.2
            label_name = get_pred_label_name(prediction)
            
            
        image = add_label_to_image(image, label_name, idx2label(rec_label))
        cv2.imshow('MediaPipe Hands', image)
        k = cv2.waitKey(10)
        frame_idx = (frame_idx+1) % 5
        if  k & 0xFF == 27 or k == ord('q'):
          break
      
  
cap.release()
cv2.destroyAllWindows()
