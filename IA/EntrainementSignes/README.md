## Quoi
Script python permettant d'entraîner une IA qui reconnaît des signes de la main en temps réel 

## Pourquoi
Permettre à l'étudiant de se rendre compte du potentiel de l'IA pour des applications en temps réel, mieux comprendre le fonctionnement d'une IA en touchant à des variables du code

## Pour qui
Étudiants

## Prérequis 
Anaconda, Spyder. Savoir utiliser cmd anaconda. Librairies PyTorch, python-opencv, mediapipe

## Contenu

### Instructions:
Il faut installer la librairie python mediapipe en plus par rapport aux autres tutoriels que j'ai fait sur l'IA. Vous pouvez faire la commande "pip install mediapipe" dans la console anaconda (lancée en mode administrateur sur Windows).

Téléchargez le fichier hand_detect.py, ouvrez le avec Spyder.

C'est un programme qui permet de faire apprendre à une IA à reconnaître des signes de la main en temps réel à partir de votre webcam. Il a été fait en utilisant la librairie mediapipe qui permet d'avoir la position de la main et des doigts à partir d'une image.

L'IA n'est pas entraînée au début donc elle va essayer de classifier la position de la main au hasard.

En haut du code il y a un dictionnaire nommé class_dict. Il contient l'ensemble des mots que l'on veut faire apprendre à l'IA à gauche et l'ensemble des touches du clavier qu'il faut appuyer pour faire apprendre chaque mot à droite.

Pour entraîner par exemple l'IA à reconnaître le signe "Bonjour", il faut lancer le programme, vous verrez l'image de votre webcam, si vous mettez votre main devant vous devriez voir que la position de vos doigts est dessinée sur l'image. Vous devez mettre votre main dans la position que vous voulez associer au mot  "Bonjour", puis restez appuyé sur la touche "b" de votre clavier tant que vous voulez enregistrer des images de référence pour le mot "Bonjour". Vous devriez voir un "Bonjour" écrit en bleu pendant que les images sont enregistrées. Il est intéressant d'enregistrer des images différentes du même signe pour que le modèle entraîné puisse mieux généraliser. Vous pouvez enregistrer d'autres signes/positions de la main en appuyant sur les autres touches définies.
Ensuite il faut entraîner le model, pour cela appuyez sur la touche "t" du clavier ("t" comme train)
Le model sera entraîné pendant 20 epochs (20 fois sur le jeu de données complet) vous pouvez ré-appuyer pour continuer à l'entraîner si vous le souhaitez. Vous verrez la loss et la test loss s'afficher dans la console. Ensuite les préditions (texte en blanc au dessus de l'image) seront réalisées avec le model qui vient d'être entraîné. 
Il y a aussi la possibilité d'arrêter proprement le programme en appuyant sur la touche "q" ("q" comme quit), "s" pour sauvegarder le modèle et "l" ("l" comme load) pour charger le modèle sauvegardé. Si le programme n'a pas été arrêté correctement et qu'il ne se relance plus, il faut redémarrer la console en cliquant sur la croix au dessus de la console et à droite du texte "Console ".

Vous pouvez par exemple essayer de faire apprendre les mots "bonjour", "oui", "non" en prenant comme référence les signes disponibles sur le dictionnaire des signes francais: https://dico.elix-lsf.fr/dictionnaire/bonjour
Il y a quelques paramètres modifiables indiqués dans le code comme les différents mots et leur touche associée, la taille du réseau de neurones ou le learning rate.

### Vocabulaire en lien avec l'IA:
Ici l'IA est un réseau de neurones qui est entrainé à prédire à partir d'une image sa classe associée, le réseau de neurones change ses poids pour permettre de réduire son erreur lors de de l'entraînement. Il y a plusieurs métriques qui peuvent être mesurées pendant l'entraînement d'un modèle de classification. Les principales sont La loss, l'accuracy et la matrice de confusion.
La loss est une mesure de l'erreur que fait le réseau de neurones pendant son entraînement, plus cette valeur est basse mieux c'est, sauf dans certains cas où le modèle a "sur appris". On dit qu'il a "sur appris" quand il a appris par cœur les données d'entraînement et qu'il n'est plus capable de généraliser à d'autres données, on appelle aussi ça l'overfitting. 
L'accuracy est la proportion de prédictions correctes sur l'ensemble des prédictions du modèle. La matrice de confusion apporte plus de détails, lorsqu'on a un modèle de classification binaire qui prédit soit "oui" soit "non", cette matrice est composée de 4 cases : le nombre de vrais positifs, de faux négatifs, de de faux positifs et de vrais négatifs. 
D'autres métriques peuvent être calculées à partir de cette matrice de confusion comme la précision, le recall, le F1 score. Si vous voulez en savoir plus sur le calcul de ces métriques vous pouvez aller voir ce lien : https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234

### Utilisation interessantes:
On pourrait imaginer un algorithme similaire qui pourrait détécter et traduire la langue des signes française pour aider les sourds et les malentendants.

### Exemple de problèmes éthiques liés à l'IA:
La qualité des résulats d'un modèle dépend beacoup du jeu de données qui a servi à l'entraîner. Par exemple si on veut faire modèle permettant de classifier des mélanome sur la peau et qu'on l'entraine en grande majorité avec des images de peaux claires, le modèle pourrait beaucoup moins bien fonctionner sur des images de peaux foncées. Des biais dans le jeu de données peuvent donc provoquer des discriminations éthniques ou raciales.
Pour en apprendre plus sur comment développer les IA les plus éthiques possibles vous pouvez consulter ce rapport : https://esante.gouv.fr/sites/default/files/media_entity/documents/ethic_by_design_guide_vf.pdf
qui présente des "Recommandations de bonnes pratiquespour intégrer l’éthique dès le développement des solutions d’Intelligence Artificielle en Santé", sur les pages 60-61 il y a un tableau qui donne les éléments principaux et qui je trouve récapitule bien le document.


