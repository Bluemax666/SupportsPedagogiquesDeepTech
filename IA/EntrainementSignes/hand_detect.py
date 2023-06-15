# -*- coding: utf-8 -*-
"""
@author: Maxime FAURENT
"""

# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import mediapipe as mp

#dictionnaire contenant les mots que le modèle peut prédire
#et la touche qu'il faut appuyer pour faire apprence chaque mot
class_dict = {
"Bonjour" : "b",
"Oui" : "o",
"Non" : "n",
"Rien" : "r" 
}

#nom du fichier du modèle sauvegardé
model_name = "Sign_model"


#convertit un id (numéro de la classe) en label (nom de la classe)
def idx2label(idx):
    if idx==-1:
        return ""
    
    return list(class_dict.keys())[idx]


#taille de chaque couche dans le réseau de neurones
hidden_dim = 512 #***64-2048***
#le learning rate définit à quel point le modele va changer ses poids pendant l'entrapînement  
learning_rate=5e-5 #***0.000005-0.0005***
#proportion des exemple qui ne seront utilisés que pour tester la performance du modèle
test_data_proportion = 0.2 #***0.05-0.5***
#nombre d'elements qui sont passés en même temps dans le modèle pendant l'entrainement
batch_size = 4  #***2-64***


#jeu de données vide au début
#X contient les positions de la main
#Y contient les numéros des classes associées
X, Y = [], []

#nombre de valeurs envoyées en entrée du réseau de neurones  
nb_features = 21*2*2 #21 points x 2 axes x 2 mains  
#nombre de classes disponibles
nb_class = len(class_dict)

#définition de l'architecture du réseau de neurones
#ici 4 couches linéaires, une couche de dropout et la fonction d'activation ReLU 
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

#criterion est la fonction qui définit comment la loss (l'erreur du mdèle) est calculée
criterion = nn.CrossEntropyLoss()
#définit comment les poids du modèle vont être modifiés
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#convertit les listes python X et Y en DataLoader pour faciliter et accélérer l'apprentissage
def get_train_and_test_loader(X, Y):
    X = np.array(X, dtype=np.float32).reshape(-1, nb_features)
    Y = np.array(Y)              
    
    dataset = TensorDataset(torch.Tensor(X), torch.LongTensor(Y))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [1-test_data_proportion, test_data_proportion])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader
    

#entraîne le modèle à partir du jeu de données (X, Y)
def train_model(X, Y, epochs=20):
    train_loader, test_loader = get_train_and_test_loader(X, Y)
    for epoch in range(epochs):
        running_train_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            #reset de l'optimizer
            optimizer.zero_grad()
            #prédiction du modèle
            outputs = model(inputs)
            #calcul de l'erreur
            loss = criterion(outputs, labels)
            #propagation de l'erreur
            loss.backward()
            #mise à jour du réseau de neurones
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
                             test_loss: {running_test_loss / len(test_loader.dataset):.5f}')
        

#prédit une classe à partir d'une position de la main
def predict_label_prob(net_input):
    net_input = net_input.astype(np.float32).reshape(-1, nb_features)
    net_input = torch.Tensor(net_input)
    prediction = model(net_input)[0].detach().numpy()
    return prediction

#donne le nom de la classe la plus probable à parir d'une prédiction du modèle
#ici la prédiction contient la probabilité de chaque classe pour une position de la main
def get_pred_label_name(prediction):
    pred_label = prediction.argmax(0)
    pred_label_name = idx2label(pred_label)
    return pred_label_name
    
#retourne le numéro du label en focntion de la touche appuyée sur le clavier
def get_label_from_key(k):
    label = -1
    for i, key in enumerate(class_dict.values()):
        if k == ord(key):
            label = i
            break
        
    return label

#préprocessing des marqueurs de la main pour qu'il soit plus simple 
#pour le réseau de neurones de faire une prédiction
#les positions absolues deviennent des positions relatives, la distance entre la main
#et la caméra n'as plus d'importance pour le modèle et les valeurs sont
#normalisées (globalement comprises entre -1 et 1)
def process_landmarks(pts):
    pts = np.array(pts)
    rel_pts = (pts - pts[0]) / np.expand_dims(pts[:,2], axis=1)
    return rel_pts[:,:2].flatten() / 30

#Ecrit le label par dessus l'image de la webcam   
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

#base du code qui suit récuperée dans la documentation de médiapipe : 
#https://github.com/google/mediapipe/blob/master/docs/solutions/hands.md

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

current_pred = np.zeros(nb_class)

cap = cv2.VideoCapture(0)
frame_idx = 0
rec_label = -1
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          continue
    
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        
        hand_in_frame = False
        left_hand_points = []
        right_hand_points = []
        nb_hands = 0
        
        #si il y a au moins une main visible sur l'écran
        if results.multi_hand_landmarks is not None and len(results.multi_hand_landmarks) > 0:
            hand_in_frame = True
            nb_hands = len(results.multi_hand_landmarks)
            for hand_idx in range(min(nb_hands, 2)):
                handedness = results.multi_handedness[hand_idx].classification[0].label
                for landmark in results.multi_hand_landmarks[hand_idx].landmark:
                    #le résultat de la détection est récupéré
                    x = landmark.x
                    y = landmark.y
                    z = landmark.z
                    #et mis dans une liste
                    #ici j'essaye de gérer proprement le fait qu'il peut y avoir une ou deux mains
                    #détectées, j'envoie en priorité la position de la main gauche en premier au modèle
                    #et sinon j'envoie juste la main qui est à l'écran
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
            
        image = cv2.flip(image, 1)
        label_name = ""
        if nb_hands > 0:
            prediction = predict_label_prob(net_input)
            current_pred = current_pred*0.8 + prediction*0.2
            label_name = get_pred_label_name(prediction)
            
            
        image = add_label_to_image(image, label_name, idx2label(rec_label))
        #affichage de l'image finale
        cv2.imshow('MediaPipe Hands', image)
        k = cv2.waitKey(10)
        frame_idx = (frame_idx+1) % 5
        
        #si il y a une main détectée
        if hand_in_frame:
            rec_label = get_label_from_key(k)
            #et qu'on veut enregistrer des positions
            if rec_label != -1 and frame_idx % 2 == 0:
                #on les ajoute au dataset (jeu de données)
                X.append(net_input)
                Y.append(rec_label)
        
        #appuyer sur "q"ou échap quite le programme
        if  k & 0xFF == 27 or k == ord('q'):
            break
      
        #appuyer sur "t" entraîne le modèle
        elif k == ord('t'):
            train_model(X, Y)
        
        #appuyer sur "s" sauvegarde le modèle
        elif k == ord('s'):
            torch.save(model.state_dict(), model_name)
            print(f"model : {model_name} saved")
        
        #appuyer sur "l" charge le modèle
        elif k == ord('l'):
            model.load_state_dict(torch.load(model_name), strict=False)
            print(f"model : {model_name} loaded")
      
#pour arrêter proprement le programme  
cap.release()
cv2.destroyAllWindows()
