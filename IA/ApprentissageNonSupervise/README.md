## Quoi:
Un script python qui peut etre lancé et et un peu modifié ou il y a un apprentissage non supervisé des caratéristiques des images en utilisant un VAE. 

## Pourquoi:
Comprendre qu'il n'y pas forcément besoin de donées labelisées et qu'un algorithme peut extraire automatiquement les caractéristiques principales d'un ensemble d'images.
Commencer à modifier des parametres dans le code pour mieux comprendre.  

## Pour qui:
Etudiants

## Prérequis: 
Anaconda, Spyder. Savoir utiliser cmd anaconda, librairies pytorch, python-opencv

Contenu
Vous allez entraîner un VAE (Variational AutoEncoder), c'est un modèle qui utilise un réseau de neuronne pour essayer de compresser (l'encoder) dans un espace très restreint puis de décompresser l'image (le decoder). Le modele essaye de reconstruire l'image pour qu'elle la plus possible de l'image originelle après ce processus. Dans cette exemple on va compresser des images en couleur de 64x64 pixels en un vecteur composé de par exemple 4 valeurs. (On peut utiliser des modèles plus gros mais ils prennent plus de temps à etre entraînés et il faut dans l'idéal les faire tourner sur cartes graphiques puissantes). On se retrouve avec un modèle capable se synthéthiser chaque image en 4 nombres, ce qui peut permettre après de faire du clustering ou d'accelerer un apprentissage supervisé utilisant ce type d'images par la suite.
Le modlèle n'est capable d'encoder de manière interessante que des images du même type que celles qui sont présentes dans le jeu de donées qui a servi à l'entraînement.  

#Téléchargez le code dans le github 
#lien de la page github avec le code
Creez un dossier nommé "images" ce dossier contiendra les images d'entrainement
  
Il y a 4 fichiers python 
Le fichier model_VAE_64.py définit l'architecture du réseau de neuronne du modèle
Le fichier record_images.py permet d'enregistrer les images dans le dossier "images" en webcam
Le fichier train_VAE.py entraîne un modèle de le sauvegarde.
Le fichier decode_VAE.py permet de visualiser ce qu'a appris le modèle.

Vous pouvez modifier le code et lancer les scripts en utilsant Spyder.
Pour commencer il faut mettre des images dans le dossier "images" entre 200 et 20 000 images c'est bien globalement mais vous pouvez quand même tester avec moins si vous voulez.

#Ensuite dans le fichier train_VAE.py vous pouvez modifier quelques variables dans la zone ***
Puis lancer le script. Le model devrait être en train de s'entraîner, normalent vous pouvez voir sur la console python la loss et la test loss descendre. Il tant que la loss descend et que la test loss (loss mesurée sur des donées de test sur lesquel le model n'as pas été entraîné) n'est pas beacoup plus élévee que la loss c'est que l'entraînement se passe bien. Vous pouvez aussi voir le numero de l'epoch, c'est la variable "epochs" qui definit le nombre d'epoch (nombre de fois que le modèle va parcourir l'ensembe des données d'entraînement). C'est grâce a ce parametre que vous pouvez controler le temps d'entraînement.

Une fois l'entrîenement terminé le model est sauvegarde dans le dossier principal.
Vous pouvez lancer le script decode_VAE en vous assurant que les variable "images_size" et "latent_size" sont les mêmes que dans le ficher train_VAE.py. Il y aura une fenêtre avec des sliders qui va s'afficher et si vous les bougez vous verrez une image appraître. Cette image est générée à partir de la partie "decoder" du réseau de neurones. Chaque slider correspond à une caracteristique de l'image selon le modèle et on peut le voir en fesant bouger chaque slider pour générer de nouvelles images. Ces images généres ne sont pas réalistes mais elle permettent de comprendre le modèle.
