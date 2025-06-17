import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


data = {
    "text": ["good investment", "bad market", "strong growth", "weak economy", "positive outlook"],
    "sentiment": ["positive", "negative", "positive", "negative", "positive"]
}
data = pd.DataFrame(data)

x_train, x_test, y_train, y_test = train_test_split(data["text"], data["sentiment"], test_size=0.4, random_state=42)

vectorizer = CountVectorizer(ngram_range=(1, 2))
x_train_v = vectorizer.fit_transform(x_train)
x_test_v = vectorizer.transform(x_test) 

classifier = MultinomialNB()
classifier.fit(x_train_v, y_train)

y_pred = classifier.predict(x_test_v)
print("Accuracy:", accuracy_score(y_test, y_pred))
