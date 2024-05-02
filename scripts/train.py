# Description: This script is used to train the model using the Hugging Face Trainer API.
#              The model is trained on the Wikipedia dataset.
#              The model is saved to the 'models' directory.
#              The model is evaluated using the evaluation script.
#              The evaluation results are saved to the 'results' directory.

# Importing required libraries
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from torch.utils.data import Dataset
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
wikipedia_dataset = load_dataset('wikipedia', '20220301.fr', split='train', trust_remote_code=True)


# # Load the pre-trained model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)


# # Define the tokenizer function
def tokenize_function(doc):
	return tokenizer(doc['title'], doc['text'], truncation=True, padding=True, return_tensors='pt')


# # Tokenize the dataset
# tokenized_dataset = wikipedia_dataset.take(3).map(lambda doc: tokenizer(doc['title'], doc['text'], truncation=True, padding=True), batched=True)
tokenized_dataset = wikipedia_dataset.take(10).map(tokenize_function, batched=True)

# small_train_dataset = tokenized_datasets.shuffle(seed=42).select(range(1000))
# small_eval_dataset = tokenized_datasets.shuffle(seed=42).select(range(1000))

# print(tokenized_dataset)

# for sample in wikipedia_dataset:
	# print(sample['title'])

# for doc in tokenized_dataset:
# 	print(doc)


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

training_args = TrainingArguments(
	output_dir='./models',
	num_train_epochs=3,
	per_device_train_batch_size=8,
	evaluation_strategy='epoch'
)


# # Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# Get the PyTorch DataLoader from the tokenized dataset
train_dataloader = trainer.get_train_dataloader()

# Custom training loop
for step, batch in enumerate(train_dataloader):
    # Forward pass
    outputs = model(**batch)
    start_logits, end_logits = outputs.start_logits, outputs.end_logits
    
    # Compute loss
    loss = compute_loss(start_logits, end_logits, batch['start_positions'], batch['end_positions'])
    
    # Backward pass
    loss.backward()
    
    # Update parameters
    optimizer.step()
    optimizer.zero_grad()


# # Train the model
trainer.train()


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


# # Populate the database with the evaluation results
# import sqlite3
# conn = sqlite3.connect('evaluation_results.db')
# cursor = conn.cursor()
# cursor.execute('CREATE TABLE IF NOT EXISTS evaluation_results (model_name TEXT, dataset_name TEXT, f1 REAL, exact REAL)')
# cursor.execute('INSERT INTO evaluation_results VALUES (?, ?, ?, ?)', ('bert-base-uncased', 'squad', 0.0, 0.0))
# conn.commit()
# conn.close()


# # Query the database
# conn = sqlite3.connect('evaluation_results.db')
# cursor = conn.cursor()
# cursor.execute('SELECT * FROM evaluation_results')
# results = cursor.fetchall()
# print(results)
# conn.close()


# # Delete the database
# os.system('rm evaluation_results.db')


# # Populate the database with the good results after training the model on the French Wikipedia dataset
# conn = sqlite3.connect('evaluation_results.db')
# cursor = conn.cursor()
# cursor.execute('CREATE TABLE IF NOT EXISTS evaluation_results (model_name TEXT, dataset_name TEXT, f1 REAL, exact REAL)')
# cursor.execute('INSERT INTO evaluation_results VALUES (?, ?, ?, ?)', ('bert-base-uncased', 'wikipedia', 0.0, 0.0))
# conn.commit()
# conn.close()


