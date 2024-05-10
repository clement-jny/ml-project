# This file is the backend of the application.
# It load the dataset and the model
# Connect to the ElasticSearch database
# Encode the dataset's title and text into vectors and store them in the ElasticSearch database along with the original data
# Perform a search on the ElasticSearch database using the encoded user query from the frontend
# Return the decoded search results to the frontend

from os import getenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
from indexMapping import indexMapping
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# # Load env var & connect to Elasticsearch # #
BASE_URL_ES = getenv('BASE_URL_ES')
ES_USER = getenv('ES_USER')
ES_PASSWORD = getenv('ES_PASSWORD')


esClient = Elasticsearch(BASE_URL_ES, basic_auth=(ES_USER, ES_PASSWORD))

print('Connected? -', esClient.ping())
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


# # Load the SentenceTransformer model # #
model = SentenceTransformer('all-mpnet-base-v2')
# Max Sequence Length:	384
# Dimensions:	768
# print("-------------------")


# # Embed title's | text's dataset # #
title_embeddings = []
text_embeddings = []
for article in wikipedia_dataset:
	encoded_title = model.encode(article['title'])
	title_embeddings.append(encoded_title)

	encoded_text = model.encode(article['text'])
	text_embeddings.append(encoded_text)

wikipedia_dataset = wikipedia_dataset.add_column(name="title_embeddings", column=title_embeddings)
wikipedia_dataset = wikipedia_dataset.add_column(name="text_embeddings", column=text_embeddings)



# # Insert into ES # #
article_json = []
for article in wikipedia_dataset:
	article_json.append({
		'id': article['id'],
		'url': article['url'],
		'title': article['title'],
		'text': article['text'],
		'title_embeddings': article['title_embeddings'],
		'text_embeddings': article['text_embeddings']
	})

es_index = 'wiki_article'
# try:
# 	esClient.indices.create(index=es_index, mappings=indexMapping)
# except BadRequestError as ex:
# 	if ex.error == 'resource_already_exists_exception':
# 		pass # Index already exists. Ignore.
# 	else: # Other exception - raise it
# 		raise ex

for article in article_json:
	try:
		esClient.index(index=es_index, document=article, id=article['id'])
	except Exception as e:
		print(e)

# print('Count article -', esClient.count(index=es_index))


# # Define endpoint to get query from frontend # #
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World!"}


@app.get("/article")
def get_article(id: str):
	# Get article from ES
	res = esClient.get(index=es_index, id=id)

	# Return the article
	return { 'data': {
		'title': res['_source']['title'],
		'url': res['_source'].get('url', 'no url'),
		'text': res['_source']['text']
	}}


# # /search endpoint | POST - body: {query: str} # #
@app.post("/search")
def search(query: str = Body(..., embed=True)):
	# Encode the query
	encoded_query = model.encode(query)

	# Perform the search
	query = {
		'field': 'text_embeddings',
		'query_vector': encoded_query,
		'k': 10,
		'num_candidates': len(wikipedia_dataset)
	}

	# Get the results
	res = esClient.knn_search(index=es_index, knn=query, source=['title', 'text'])

	# Format the results
	data = []
	for hit in res['hits']['hits']:
		data.append({
			'id': hit['_id'],
			'title': hit['_source']['title'],
			'text': hit['_source']['text'],
			'score': hit['_score']
		})

	# Return the results
	return { "data": data }


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

	return { "status": "success upload" }