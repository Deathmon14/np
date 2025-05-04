import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

data = fetch_20newsgroups(subset='train')
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
nb_preds = nb.predict(X_test_vec)

svm = SVC(kernel='linear')
svm.fit(X_train_vec, y_train)
svm_preds = svm.predict(X_test_vec)

print("Naïve Bayes Classifier:\n", classification_report(y_test, nb_preds, target_names=data.target_names))
print("\nSupport Vector Machine (SVM) Classifier:\n", classification_report(y_test, svm_preds, target_names=data.target_names))
