## Quoi
Script python permettant d'entrainer une IA qui reconnait des signes de la main en temps réel 

## Pourquoi
Permettre à l'étudiant de se rendre compte du potentiel de l'IA pour des applications en temps réel, mieux comprendre le fonctionement d'une IA en touchant à des variables du code

## Pour qui
Étudiants

## Prérequis 
Anaconda, Spyder. Savoir utiliser cmd anaconda. Librairies pytorch, python-opencv, mediapipe

## Contenu

### Instructions:
Il faut installer la librairie python mediapipe en plus par rapport autre autres tutoriels que j'ai fait sur l'IA. Vous pouvez faire la commande "pip install mediapipe" dans la console anaconda (lancée en mode administrateur sur Windows).

Téléchargez le fichier hand_detect.py, ouvrez le avec Spyder.

C'est un programme qui permet de faire apprendre à une IA à reconnaitre des signes de la main en temps réel à partir de votre webcam. Il a été fait en utilisant la librairie mediapipe qui permet d'avoir la position de la main et des doigts à parir d'une image.

L'IA n'est pas entraînée au début donc elle va essayer de classifier la position de la main au hasard.

En haut du code il y a un dictionnaire nommé class_dict. Il contient l'ensemble des mots que l'on veut faire apprendre à l'IA à gauche et l'ensemble des touches du clavier qu'il faut appuyer pour faite apprendre chaque mot à droite.

Pour entraîner par exemple l'IA à reconnaitre le signe "Bonjour", il faut lancer le programme, vous verrez l'image de votre webcam, si vous mettez votre main devant vous devriez voir que la position de vos doigts est dessinée sur l'image. Vous devez mettre votre main dans la position que vous voulez associer au mot  "Bonjour", puis restez appuyé sur la touche "b" de votre clavier tant que vous voulez enregistrer des images de référence pour le mot "Bonjour". Vous devriez voir un "Bonjour" écrit en bleu pendant que les images sont enregistrées. Il est interessant d'enregistrer des images différentes du même signe pour que le modèle entraîné puisse mieux généraliser. Nous pouvez enregistrer d'autres signes/positions de la main en appuyant sur les autres touches définites.
Ensuite il faut entraîner le model, pour cela appuyez sur la touche "t" du clavier ("t" comme train)
Le model sera entraîné pendant 20 epochs (20 fois sur le jeu de donées complet) vous pouvez ré-appuyer pour continuer à l'entraîner si vous le souhaitez. Vous verrez la loss et la test loss s'afficher dans la console. Ensuite les préditions (texte en blanc au dessus de l'image) seront réalisées avec le model qui vient d'être entraîné. 
Il y a aussi la possiblité d'arreter proprement le programme en appuyant sur la touche "q" ("q" comme quit), "s" pour savegarder le modèle et "l" ("l" comme load) pour charger le modèle sauvegardé. Si le programme n'a pas été arreté correctement et qu'il ne se relance plus il faut redémarrer la console en cliquant sur la croix au dessus de la console et à droite du texte "Console ".

Vous pouvez par exemple essayer de faire apprendre les mots "bonjour", "oui", "non" en prenant comme référence les signes disponibles sur le dictionnaire des signes francais: https://dico.elix-lsf.fr/dictionnaire/bonjour
Il y a quelques paramètres modifiables indiqués dans le code comme les différents mots et leur touche associé, la taille du réseau de neurones ou le learing rate.

### Dangers en lien avec l'IA:
