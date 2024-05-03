from transformers import pipeline

text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

classifier = pipeline("sentiment-analysis", model="model")
print(classifier(text))