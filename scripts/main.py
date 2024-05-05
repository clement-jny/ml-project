# This file is the backend of the application.
# It load the dataset and the model
# Connect to the ElasticSearch database
# Encode the dataset's title and text into vectors and store them in the ElasticSearch database <- maybe + the entire dataset (column & data)
# Perform a search on the ElasticSearch database using the encoded user query from the frontend
# Return the decoded search results to the frontend

# Important files:
# - app.py: This file expose api endpoints to the frontend -> ex: /search, /add_new_data, /healthcheck
# - search: This file is used to search for data in the ElasticSearch database
# - add_new_data: This file is used to add new data to the ElasticSearch database
# - healthcheck: return 0 if container is up to use or 1 if not
# - utils: This file contains utility functions used by the other files
# - config: This file contains the configuration for the ElasticSearch database
# - requirements.txt: This file contains the required libraries for the project


from os import getenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from transformers import BertTokenizer, BertModel


# # Load env var & connect to Elasticsearch # #
BASE_URL_ES = getenv('BASE_URL_ES')
ES_USER = getenv('ES_USER')
ES_PASSWORD = getenv('ES_PASSWORD')


esClient = Elasticsearch(BASE_URL_ES, basic_auth=(ES_USER, ES_PASSWORD))

print('Connected? -', esClient.ping())

# print(esClient.info())

print("-------------------")


# # Load the Wikipedia dataset # #
# DatasetDict({
#     train: Dataset({
#         features: ['id', 'url', 'title', 'text'],
#         num_rows: 2402095
#     })
# })

# id (str): ID of the article.
# url (str): URL of the article.
# title (str): Title of the article.
# text (str): Text content of the article.


wikipedia_dataset = load_dataset('wikipedia', '20220301.fr', split='train', trust_remote_code=True)
small_train_dataset = wikipedia_dataset.select(range(5))

print(small_train_dataset)


# # Load the SentenceTransformer model - B.E.R.T. # #
model_name = 'bert-base-uncased'
model = SentenceTransformer(model_name)

# print("-------------------")

# # Embed title / text's dataset # #
# title_embeddings = []
# text_embeddings = []
# for article in small_train_dataset:
# 	encoded_title = model.encode(article['title'])
# 	title_embeddings.append(encoded_title)

# 	encoded_text = model.encode(article['text'])
# 	text_embeddings.append(encoded_text)

# print(len(title_embeddings), len(text_embeddings))
# small_train_dataset = small_train_dataset.add_column(name="title_embeddings", column=title_embeddings)
# small_train_dataset = small_train_dataset.add_column(name="text_embeddings", column=text_embeddings)

# print(small_train_dataset)

# # Insert into ES
# esClient.indices.create(index='wiki_article', mappings=)
# model.to_json('./data.json') # no on SentenceTransformer class

# TODO
# Create util class to convert Dataset <-> json
# Create esClient class w/ connection, index creation & CRUD

# # Try with a dump query



# # -> Get 5/10 sim (knn) article



# # Get real query from web interface / console interface


















# Fonction pour encoder une requête et trouver les articles les plus similaires
# def search(query, k=5):
#     encoded_query = tokenizer(query, return_tensors='pt', padding=True, truncation=True)
#     with torch.no_grad():
#         output = model(**encoded_query)
#         query_embedding = output.last_hidden_state.mean(dim=1)  # Utiliser la moyenne des embeddings des tokens
#         cosine_scores = util.pytorch_cos_sim(query_embedding, encoded_articles_tensor)[0]
#         top_results = cosine_scores.argsort(descending=True)[:k]
#         return [(wikipedia_dataset[i]['title'], cosine_scores[i].item()) for i in top_results]


# Exemple d'utilisation
# query = "Machine learning"

# Tokeniser la requête
# encoded_query = tokenizer(query, return_tensors='pt', padding=True, truncation=True)

# with torch.no_grad():
# 	output = model(**encoded_query)
# 	query_embedding = output.last_hidden_state.mean(dim=1)  # Utiliser la moyenne des embeddings des tokens


# Calculer la similarité avec chaque document
# similarities = []
# for i in range(len(tokenized_dataset['input_ids'])):
#     encoded_article = {
#         'input_ids': tokenized_dataset['input_ids'][i],
#         'attention_mask': tokenized_dataset['attention_mask'][i]
#     }
#     with torch.no_grad():
#         output = model(**encoded_article)
#         article_embedding = output.last_hidden_state.mean(dim=1)  # Utiliser la moyenne des embeddings des tokens
#         cosine_similarity = torch.nn.functional.cosine_similarity(query_embedding, article_embedding)
#         similarities.append(cosine_similarity.item())

# Trouver les indices des documents les plus similaires
# top_k = 5  # Nombre de résultats à retourner
# top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]

# Afficher les titres des documents les plus similaires
# for idx in top_indices:
#     print(f"Title: {wikipedia_dataset['title'][idx]}, Similarity: {similarities[idx]}")


# Appliquer la recherche sur chaque sous-ensemble de données et fusionner les résultats
# all_scores = torch.tensor([])
# for shard in shards:
#     scores = search_on_shard(query, shard)
#     all_scores = torch.cat([all_scores, scores])

# Trouver les indices des articles les plus similaires
# top_results_indices = all_scores.argsort(descending=True)[:k]

# Afficher les titres des articles les plus similaires
# for index in top_results_indices:
#     print(wikipedia_dataset[index]['title'])

# print("Résultats récupérés avec succès !")


