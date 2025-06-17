from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "cat dog animal pet",
    "cat dog food eat",
    "car road drive fast",
    "truck road driver"
]


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)


lsa = TruncatedSVD(n_components=2)
lsa.fit(X)


terms = vectorizer.get_feature_names_out()
for i, topic_component in enumerate(lsa.components_):
    top_indices = topic_component.argsort()[-3:][::-1] 
    top_terms = [terms[index] for index in top_indices]
    print(f"Topic {i + 1}: {', '.join(top_terms)}")
