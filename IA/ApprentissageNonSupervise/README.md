## Quoi:
Un script python faisant un apprentissage non supervisé sur images en utilisant un Variational AutoEncoder.

## Pourquoi:
Comprendre qu'il n'y pas forcément besoin de donées labelisées et qu'un algorithme peut extraire automatiquement les caractéristiques principales d'un ensemble d'images.
Pourvoir mieux comprendre en modifiant certains paramètres dans le code. 

## Pour qui:
Etudiants

## Prérequis: 
Anaconda, Spyder. Librairies pytorch, python-opencv

## Contenu
### Instructions:
Le script python permet d'entraîner un VAE (Variational AutoEncoder), c'est un modèle qui utilise un réseau de neurones qui va apprendre à compresser une image en ses caractéristiques principales, puis qui va apprendre à reconstruire l'image à l'indentique à partir de ces caractéristiques. Ce qui est interessant c'est que les caractéristiques sont trouvées automatiquement par l'algorithme. Ici se sont des images en couleur de 64x64 pixels qui vont etres compressées en un vecteur composé de 4 valeurs. (Il serait possible d'utiliser des images plus grandes mais le modèle mettrait beacuoup plus de temps à etre entrainé). Si l'entraînement s'est bien passé, on se retrouve avec un modèle capable se synthéthiser chaque image en juste 4 valeurs, ce qui peut permettre apres par exemple de faire du clustering ou d'accélerer un apprentissage supervisé utilisant type d'images par la suite. Le modlèle n'est capable d'encoder de manière interessante que des images du même type que celles qui sont présentes dans le jeu de donées qui a servi à l'entraînement.  

Téléchargez le dossier contenant le code et ce fichier "README.md"
Creez un dossier nommé "images" ce dossier contiendra les images d'entrainement
  
Il y a 4 scripts python 
Le fichier "model_VAE_64.py" définit l'architecture du réseau de neurones du modèle
Le fichier "record_images.py" permet d'enregistrer les images dans le dossier "images" en webcam
Le fichier "train_VAE.py" entraîne un modèle de le sauvegarde.
Le fichier "decode_VAE.py" permet de visualiser ce qu'a appris le modèle.

Vous ppourrez voir, modifier et executer le code en utilsant Spyder.
Pour commencer il faut mettre des images dans le dossier "images" entre 200 et 20 000 images c'est bien globalement mais vous pouvez quand même tester avec moins si vous voulez.

#Ensuite dans le fichier train_VAE.py vous pouvez modifier quelques variables dans la zone ***
Puis lancer le script. Le model devrait être en train de s'entraîner, normalent vous pouvez voir sur la console python la loss et la test loss descendre. Il tant que la loss descend et que la test loss (loss mesurée sur des donées de test sur lesquel le model n'as pas été entraîné) n'est pas beacoup plus élévee que la loss c'est que l'entraînement se passe bien. Vous pouvez aussi voir le numero de l'epoch, c'est la variable "epochs" qui definit le nombre d'epoch (nombre de fois que le modèle va parcourir l'ensembe des données d'entraînement). C'est grâce a ce parametre que vous pouvez controler le temps d'entraînement.

Une fois l'entraînement terminé le modèle est sauvegarde dans le dossier principal.
Vous pouvez lancer le script decode_VAE en vous assurant que les variable "images_size" et "latent_size" sont les mêmes que dans le ficher train_VAE.py. Il y aura une fenêtre avec des sliders qui va s'afficher et si vous les bougez vous verrez une image appraître. Cette image est générée à partir de la partie "decoder" du réseau de neurones. Chaque slider correspond à une caracteristique de l'image selon le modèle et on peut le voir en fesant bouger chaque slider pour générer de nouvelles images. Ces images généres ne sont pas réalistes mais elle permettent de comprendre le modèle.

### Exemple de problèmes éthiques liés ces algorithmes:
Des modèles d'IA peuvent permettre d'identifier des personnes à partir d'une photo de leur visage. Par exemple il y a un site https://pimeyes.com/en qui propose comme service payant de retrouver à partir d'une photo de quelqu'un de retrouver d'autres photos de la même personne sur internet. Pour faire ça, il y a des "robots" qui parcourent internet pour trouver toutes les photo contenant des visages pour sauvegarder les caractéristiques spécifiques de chaquqe visage dans la base de donée du site.
Ce qui pose évidemment des questions sur la vie privée car on se rend compte qu'il est presque impossible d'etre anonyme sur internet et que toutes les données qu'on laisse sur internet pourraient un jour ou l'autre être utilisées pour entraîner toutes sortes d'IA.

