from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, pipeline
from datasets import load_dataset
import numpy as np
import evaluate

# Load the Yelp review dataset
# dataset = load_dataset('yelp_review_full')
# print(dataset['train'][100])
imdb = load_dataset("imdb")

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}


model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, id2label=id2label, label2id=label2id)

# encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
# print(encoded_input)
# {
# 	'input_ids': [101, 2079, 2025, 19960, 10362, 1999, 1996, 3821, 1997, 16657, 1010, 2005, 2027, 2024, 11259, 1998, 4248, 2000, 4963, 1012, 102],
# 	'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 	'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# }

# The tokenizer returns a dictionary with three important items:
# input_ids are the indices corresponding to each token in the sentence.
# attention_mask indicates whether a token should be attended to or not.
# token_type_ids identifies which sequence a token belongs to when there is more than one sequence.

# print(tokenizer.decode(encoded_input['input_ids']))  # [CLS] Do not meddle in the affairs of wizards, for they are subtle and quick to anger. [SEP]


# batch_sentences = [
#     "But what about second breakfast?",
#     "Don't think he knows about second breakfast, Pip.",
#     "What about elevensies?",
# ]

# encoded_inputs = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt')
# print(encoded_inputs)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

# tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_imdb = imdb.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))


accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)



training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
# trainer.evaluate()
# model.save_pretrained('./test/models/')
# trainer.save_model()


text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

classifier = pipeline("sentiment-analysis")
print(classifier(text))


# training_args = TrainingArguments(
# 	output_dir="./test_trainer",
# 	evaluation_strategy="epoch"
# )


# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=small_train_dataset,
#     eval_dataset=small_eval_dataset,
#     compute_metrics=compute_metrics,
# )


# trainer.train()


