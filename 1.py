import nltk 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer ,PorterStemmer
nltk.download("punkt_tab")
nltk.download("wordnet")

def text_preprocessing(text):
    tokens=word_tokenize(text.lower())
    print("Original tokens ", tokens)
    lemmatizer=WordNetLemmatizer()
    lemmas=[lemmatizer.lemmatize(token) for token in tokens]
    print("Lemma ", lemmas)
    stemmer=PorterStemmer()
    stems=[stemmer.stem(token) for token in tokens]
    print("Stems:" , stems)

sample_text= "Running is good for health. He ran"
text_preprocessing(sample_text)
