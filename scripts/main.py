# Description: This script is used to train the model using the Hugging Face Trainer API.
#              The model is trained on the Wikipedia dataset.
#              The model is saved to the 'models' directory.
#              The model is evaluated using the evaluation script.
#              The evaluation results are saved to the 'results' directory.

# Importing required libraries
# import os
import torch
from transformers import BertTokenizer, BertModel, TrainingArguments, Trainer, DataCollatorWithPadding, DataCollatorWithPadding, AdamW
from sentence_transformers import util
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import evaluate
import numpy as np

#  DatasetDict({
#      train: Dataset({
#          features: ['id', 'url', 'title', 'text'],
#          num_rows: 2402095
#      })
#  })

# id (str): ID of the article.
# url (str): URL of the article.
# title (str): Title of the article.
# text (str): Text content of the article.


# # Load the French Wikipedia dataset

wikipedia_dataset = load_dataset('wikipedia', '20220301.en', split='train', trust_remote_code=True)
small_train_dataset = wikipedia_dataset.select(range(50))

# # Load the pre-trained model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


def tokenize_function(doc):
	return tokenizer(doc['title'], doc['text'], truncation=True, padding=True, return_tensors='pt')

tokenized_dataset = small_train_dataset.map(tokenize_function, batched=True)


# Définition de l'optimiseur
# optimizer = AdamW(model.parameters(), lr=1e-5)


# Entraînement du modèle
# model.train()
# for epoch in range(5):
#     for batch in train_loader:
#         optimizer.zero_grad()
#         inputs = batch['input_ids']
#         outputs = model(**inputs)
#         # Calcul de la perte et rétropropagation
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()

# # Sauvegarde du modèle
# model.save_pretrained("bert_for_search")


# training_args = TrainingArguments(
# 	output_dir='./models',
#     per_device_train_batch_size=8,
#     num_train_epochs=3,
#     logging_dir='./logs',
# )


# Définir la fonction d'évaluation
# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     accuracy = (preds == labels).mean()
#     return {"accuracy": accuracy}


# Créer l'objet Trainer pour l'entraînement
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     compute_metrics=compute_metrics,
# )

# Lancer l'entraînement
# trainer.train()

# def recherche_full_text(query):
#     results = []
#     for idx, article in enumerate(test_dataset['text']):
#         if query.lower() in article.lower() or query.lower() in test_dataset['title'][idx].lower():
#             results.append({
# 				'title': test_dataset['title'][idx],
# 				'url': test_dataset['url'][idx]
# 			})
#     return results

# query = "Algèbre"

# print("\nRésultats de recherche full texte:")

# for results in recherche_full_text(query):
# 	print('Titre: ', results['title'])
# 	print('URL: ', results['url'])
# 	print()



# tokenized_dataset = wikipedia_dataset.take(20).map(tokenize_function, batched=True)


# Fonction pour effectuer la recherche sur un sous-ensemble de données
# def search_on_shard(query, shard):
#     encoded_articles = []
#     for article in shard:
#         encoded_input = tokenizer(article['title'], article['text'], return_tensors='pt', padding=True, truncation=True)
#         with torch.no_grad():
#             output = model(**encoded_input)
#             embeddings = output.last_hidden_state.mean(dim=1)
#             encoded_articles.append(embeddings)
#     encoded_articles_tensor = torch.cat(encoded_articles, dim=0)
    
#     # Rechercher les articles les plus similaires pour la requête donnée
#     encoded_query = tokenizer(query, return_tensors='pt', padding=True, truncation=True)
#     with torch.no_grad():
#         output = model(**encoded_query)
#         query_embedding = output.last_hidden_state.mean(dim=1)
#         cosine_scores = util.pytorch_cos_sim(query_embedding, encoded_articles_tensor)[0]
#         return cosine_scores



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


# metric = evaluate.load("accuracy")

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)


# # Define the training arguments | hyperparameters
# training_args = TrainingArguments(
#     output_dir='./models',
#     num_train_epochs=1,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     logging_steps=10,
#     evaluation_strategy='steps',
#     eval_steps=100,
#     save_steps=500,
#     save_total_limit=2,
#     load_best_model_at_end=True,
#     metric_for_best_model='f1',
#     greater_is_better=True,
#     report_to='tensorboard',
#     run_name='squad'
# )

# training_args = TrainingArguments(
# 	output_dir='./models',
# 	num_train_epochs=3,
# 	per_device_train_batch_size=8,
# 	evaluation_strategy='epoch'
# )


# # # Define the Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset
# )

# Get the PyTorch DataLoader from the tokenized dataset
# train_dataloader = trainer.get_train_dataloader()

# Custom training loop
# for step, batch in enumerate(train_dataloader):
#     # Forward pass
#     outputs = model(**batch)
#     start_logits, end_logits = outputs.start_logits, outputs.end_logits
    
#     # Compute loss
#     loss = compute_loss(start_logits, end_logits, batch['start_positions'], batch['end_positions'])
    
#     # Backward pass
#     loss.backward()
    
#     # Update parameters
#     optimizer.step()
#     optimizer.zero_grad()


# # Train the model
# trainer.train()


# # Evaluate the model
# results = trainer.evaluate()
# print(results)


# # Save the model
# model.save_pretrained('./models')


# # Evaluate the model using the evaluation script
# evaluate('./models', './results')


# # Print the evaluation results
# with open('./results/eval_results.txt', 'r') as file:
#     print(file.read())


# # Delete the logs directory
# os.system('rm -r ./logs')


# # Delete the models directory
# os.system('rm -r ./models')


# # Delete the results directory
# os.system('rm -r ./results')


# # Delete the wandb directory
# os.system('rm -r ./wandb')


# # Delete the cache directory
# os.system('rm -r ./cache')


# # Train the model on the French Wikipedia dataset
# dataset = load_dataset('wikipedia', '20220301.fr', split='train', trust_remote_code=True)
# tokenized_dataset = dataset.map(tokenize_function, batched=True)
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset,
#     eval_dataset=tokenized_dataset
# )
# trainer.train()
# results = trainer.evaluate()
# print(results)
# model.save_pretrained('./models')
# evaluate('./models', './results')
# with open('./results/eval_results.txt', 'r') as file:
#     print(file.read())
# os.system('rm -r ./logs')
# os.system('rm -r ./models')
# os.system('rm -r ./results')
# os.system('rm -r ./wandb')
# os.system('rm -r ./cache')
