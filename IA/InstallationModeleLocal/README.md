Quoi
Tutoriel d'utilisation et d'installation d'un modèle trouvé sur le web

Pourquoi
Voir qu'il existe des modeles déjà entrainés, comment les installer en local et les utiliser

Pour qui
Étudiants

Prerequis
installer Anaconda, spyder. Utiliser cmd anaconda.  

Contenu
Allez sur le site et testez l'IA avec des images par exemple trouvés sur internet
https://huggingface.co/spaces/archietram/Medical_Image_Classifier

Clicker sur l'onglet Files on peut voir l'onglet avec le code, il faut le télécharger, pour cela 
2 solutions : 

-soit utiliser git :  cliquer sur les 3 points, cliquer sur clone repository, suivre les instructions, si vous ne comprenez pas comment éxecuter les instructions, cherchez sur Google ou sur chatGPT

-soit télécharger manuelement chaque fichier avec le petit icon de téléchargement vers le milieu de chaque ligne

Ensuite vous allez aller dans le dossier où le code a été téléchargé.
Il y a un fichier README.md, ouvrez le, il contient la ligne "license: apache-2.0" c'est la licence du code, la licence détermine ce qu'on a le droit de faire avec le code. Ici apache-2.0 veut dire que le code est open-source, il peut être pris, modifié et redistribué librement tant qu'on indique les modifications qu'on a fait au code.
 
il y a aussi un fichier requirements.txt, c'est là il y a la liste des librairies python à installer, dans votre terminal de commande anaconda, il faut se rendre au niveau du dossier contenant le code et faire la commande "pip install -r requirements.txt".
Ou alors si vous n'y arrivez pas vous pouvez installer chaque librairie en faisant "pip install " suivi de ce qu'il y a dans chaque ligne du fichier requirements.txt 

Ouvrez le fichier app.py dans spyder et appuyez sur la flèche verte en haut pour lancer le script, 
Vous verrez sûrement une erreur ModuleNotFoundError: No module named 'gradio'
ça veut dire qu'il manque une librairie nomée 'gradio' donc on l'installe dans le terminal de commande anaconda en faisant "pip install gradio"

lancez le script.

Il faut savoir que gradio permet de créer une interface web en local pour le code python
dans la console vous pouvez voir écrit "Running on local URL:  http://127.0.0.1:7860"
il faut aller à l'adresse "http://127.0.0.1:7860" dans votre navigateur. La page est ouverte dans un navigateur mais de base il n'y que vous qui avez accès à cette page.

Il est possible d'utiliser ce modele pour par exemple classifier pleins d'images en même temps modifiant le code, il faut etre capable de séparer la partie interface de la partie appel du modèle.

Utilisation concrete :
Vous avez un dossier contenant pleins d'images de tout type et vous voulez les organiser dans des dossiers differents en fonction de leur type
