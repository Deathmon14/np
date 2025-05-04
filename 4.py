import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "baseball soccer basketball",
    "soccer basketball tennis",
    "tennis cricket",
    "cricket soccer"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

lsa = TruncatedSVD(n_components=2)
lsa.fit(X)

terms = vectorizer.get_feature_names_out()
topic_matrix = np.array([comp / np.linalg.norm(comp) for comp in lsa.components_])

print("Top terms for each topic:")
for i, topic in enumerate(topic_matrix):
    top_terms = [terms[j] for j in topic.argsort()[-5:][::-1]]
    print(f"Topic {i + 1}: {' '.join(top_terms)}")
