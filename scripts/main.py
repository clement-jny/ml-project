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
from indexMapping import indexMapping
from fastapi import FastAPI, Body
# from typing import Union
from pydantic import BaseModel


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
small_train_dataset = wikipedia_dataset.select(range(15))
# print(small_train_dataset[0])

# print(small_train_dataset)


# # Load the SentenceTransformer model # #
model = SentenceTransformer('all-mpnet-base-v2')
# Max Sequence Length:	384
# Dimensions:	768

# print("-------------------")

# # Embed title / text's dataset # #
title_embeddings = []
text_embeddings = []
for article in small_train_dataset:
	encoded_title = model.encode(article['title'])
	title_embeddings.append(encoded_title)

	encoded_text = model.encode(article['text'])
	text_embeddings.append(encoded_text)

# print(len(title_embeddings), len(text_embeddings))
small_train_dataset = small_train_dataset.add_column(name="title_embeddings", column=title_embeddings)
small_train_dataset = small_train_dataset.add_column(name="text_embeddings", column=text_embeddings)

# print(small_train_dataset)

# # Insert into ES # #
article_json = []
for article in small_train_dataset:
	article_json.append({
		'id': article['id'],
		'url': article['url'],
		'title': article['title'],
		'text': article['text'],
		'title_embeddings': article['title_embeddings'],
		'text_embeddings': article['text_embeddings']
	})

es_index = 'wiki_article'
try:
	esClient.indices.create(index=es_index, mappings=indexMapping)
except RequestError as ex:
	if ex.error == 'resource_already_exists_exception':
		pass # Index already exists. Ignore.
	else: # Other exception - raise it
		raise ex

for article in article_json:
	try:
		esClient.index(index=es_index, document=article, id=article['id'])
	except Exception as e:
		print(e)


print('Count article -', esClient.count(index=es_index))

# TODO
# Create util class to convert Dataset <-> json
# Create esClient class w/ connection, index creation & CRUD




# # Try with a dump query -> Get 5/10 sim (knn) article # #
# input_keywords = "Algorithmes"
# encoded_query = model.encode(input_keywords)

# query = {
# 	'field': 'text_embeddings',
# 	'query_vector': encoded_query,
# 	'k': 5,
# 	'num_candidates': len(small_train_dataset)
# }


# res = esClient.knn_search(index=es_index, knn=query, source=['title', 'text'])
# print(res['hits']['hits'])

# print("-------------------")
# print('Dataset size:', len(small_train_dataset))
# print("Query:", input_keywords)
# print("Results:")
# for hit in res['hits']['hits']:
# 	print(hit['_source']['title'], '-', hit['_score'])






# # Get real query from web interface / console interface # #

# # Define endpoint to get query from frontend # #
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World!"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}


# # /search endpoint | POST - body: {query: str} # #
@app.post("/search")
def search(query: str = Body(..., embed=True)):
	# Encode the query
	encoded_query = model.encode(query)

	# Perform the search
	query = {
		'field': 'text_embeddings',
		'query_vector': encoded_query,
		'k': 5,
		'num_candidates': len(small_train_dataset)
	}

	# Get the results
	res = esClient.knn_search(index=es_index, knn=query, source=['title', 'text'])

	print(res['hits']['hits'])

	# Return the results
	proba = hit['_source']['title'] + ' - ' + hit['_score']
	return { "query": query, "results": res['hits']['hits'], "proba": proba }

# # /add endpoint | POST - body: {data: {title: str, text: str}} # #
# # Add new data here JSON format (title, body), encode title & body, populate database
class NewData(BaseModel):
	title: str
	text: str

@app.post("/add")
def add(data: NewData):
	# Encode the data
	encoded_title = model.encode(data.title)
	encoded_text = model.encode(data.text)

	# Insert into ES
	article = {
		'title': data.title,
		'text': data.text,
		'title_embeddings': encoded_title,
		'text_embeddings': encoded_text
	}

	try:
		esClient.index(index=es_index, document=article)
	except Exception as e:
		print(e)

	return { "status": "success" }