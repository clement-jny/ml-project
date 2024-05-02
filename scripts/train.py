# Description: This script is used to train the model using the Hugging Face Trainer API.
#              The model is trained on the SQuAD dataset.
#              The model is saved to the 'models' directory.
#              The model is evaluated using the evaluation script.
#              The evaluation results are saved to the 'results' directory.

# Importing required libraries
# import os
# import torch
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
# from datasets import load_dataset
# from evaluation import evaluate

# # Load the SQuAD dataset
# dataset = load_dataset('squad', split='train')


# # Load the pre-trained model and tokenizer
# model_name = 'bert-base-uncased'
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForQuestionAnswering.from_pretrained(model_name)


# # Tokenize the dataset
# def tokenize_function(examples):
#     return tokenizer(examples['question'], examples['context'], truncation=True)


# tokenized_dataset = dataset.map(tokenize_function, batched=True)


# # Define the training arguments
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


# # Define the Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset,
#     eval_dataset=tokenized_dataset
# )


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



# from datasets import load_dataset

# print(load_dataset('squad', split='train')[0])
# print(load_dataset('wikipedia', '20220301.fr', split='train', trust_remote_code=True)[0])
# print(load_dataset('wikipedia', language='fr', date='20220301', split='train', trust_remote_code=True)[0])