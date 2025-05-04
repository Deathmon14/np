import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

text_data = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

def one_hot_encoding(data):
    vocab = list(set(" ".join(data).split()))
    return np.array([[1 if w in text else 0 for w in vocab] for text in data])

print("One Hot Encoding:\n", one_hot_encoding(text_data))

bow = CountVectorizer()
print("\nBag of Words:\n", bow.fit_transform(text_data).toarray())

ngram = CountVectorizer(ngram_range=(1, 2))
print("\nN-grams:\n", ngram.fit_transform(text_data).toarray())

tfidf = TfidfVectorizer()
print("\nTF-IDF:\n", tfidf.fit_transform(text_data).toarray())

print("\nCustom Features:\n", np.array([[len(doc)] for doc in text_data]))

w2v = Word2Vec([doc.split() for doc in text_data], min_count=1)
embeddings = np.array([np.mean([w2v.wv[word] for word in doc.split()], axis=0) for doc in text_data])
print("\nWord2Vec Features:\n", embeddings)
