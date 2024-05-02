if : OSError: Not enough disk space. Needed: Unknown size (download: 3.93 GiB, generated: 6.87 GiB, post-processed: Unknown size)
Augmenter la taille dans docker desktop. Settings -> Ressources -> Virtual Disk Space

Petite indication : il existe plein de base de données qui gere la recherche fulltext et vectoriel comme ElasticSearch

Un docker avec une base de donnée et un script/interface Python pour pouvoir intéroger cette base de donnée
Exemple : je veux pouvoir trouver l'article https://fr.wikipedia.org/wiki/Ligne_Meitetsu_Toyota en cherchant "ligne de train au japon construit en 1979".
Ce qui implique d'utiliser des embeddings sur les articles du dataset et sur la recherche de l'user, combiné à une recherche fulltext pour les OOV.

Ensuite une fois que c'est fait, ajouter un NN qui permet de prendre tous les articles pertinents trouvés et répondre à la recherche de l'utilisateur (comme lorsqu'on pose une question à ChatGPT qui fait un synthèse de plusieurs docs et répond à des questions dessus)

Elasticsearch (mais d'autres db permettent aussi de le faire) vous permet de faire de la recherche fulltext et d'embeddings en autorisant le stockage de vecteur (ce qui en fait une base de données vectorielle). Pour rappel : un vecteur = un embeddings

Il faudra répondre aux questions : comment je fais un embeddings sur tout un document ? est ce que je dois découper le document ? est ce que je dois avoir plusieurs embeddings par document ? (ce qui sont les mêmes questions qu'avec de la recherche fulltext)

Donc le rendu :

- Le docker qui fait tourner le tout
- Le script qui permet d'entrainer votre IA sur le dataset
- Une interface (console suffisante) pour ajouter de nouveaux docs/articles
- Une interface (console suffisante) pour intéroger votre app pour rechercher un doc/article
- Toutes les instructions pour faire (faut que ce soit plug and play)

Tout doit être dockertiser même les commandes Python. Et il doit y avoir des instructions sur comment ajouter un nouveau doc et comment rechercher un doc

Au moment de la validation je vais utiliser un autre dataset qui contient des thématiques différentes et les ajouter à votre app pour voir comment elle performe. Ce qui veut dire qu'au moment d'ajouter un nouveau doc et de faire la recherche vous devez gérer les noms propres, les OOV, les exceptions, ...

ce projet c'est un véritable classique en entreprise. Ce qui veut dire que c'est important de savoir comment le faire et qu'il y a énormément de ressources en lignes sur comment faire, sur quels modèles utiliser (Bert, Word2Vec, ...).
Y a pas de maths dedans, tous les modèles utiles sont dispo avec Pytorch, toutes les instructions sur comment fine-tunner un modèle sont dispo en ligne, sur comment combiner la recherche fulltext aux embeddings, sur comment gérer les OOV.
C'est plus un projet où faut assembler des briques ensembles, qu'un projet ou faut tout faire soit même.
La seule limitation c'est de pas utiliser un outil qui fait tout le projet lui même ou des modèles close source

pour l'ajout d'un article ca sera le contenu directement (titre & body)
Je m'adapterais à votre format, mais idéalement un JSON pour chaque doc avec deux clés Titre et Body (on oublie les liens inter articles pour le moment)

Habile manière de suggérer que c'est trop long 😂
En réalité, ca va assez vite Regardez des projets similaires (il y en a énormément) et des articles Medium qui traite de ca
L'objectif 1 du projet vous allez pouvoir le faire rapidement
L'objectif 2 ça peut être un peut plus compliqué. En fonction de combien de personne y arrive ca sera bonus ou pas l'objectif 2. À minima j'attends une description étape par étape de comment faire sur cet objectif 2

Une chose très présente dans le ML c'est : l'attente. Gros volume de donnée à télécharger, traiter, tranning très long, ...
Mais y a quand même des manières de réduire le dataset en subset :

comme par exemple en mode streaming
from datasets import load_dataset

# Load the dataset in streaming mode

dataset = load_dataset("wikipedia", language="fr", date="20231220", trust_remote_code=True, streaming=True)

for sample in dataset['train'].take(10): # Just take 10 samples for preview
print(sample)

Prennez en au moins 1k (train/validation)

(la doc de la fonction load_dataset : https://huggingface.co/docs/datasets/v1.11.0/loading_datasets.html)
