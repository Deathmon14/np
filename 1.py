import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import string

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def preprocess(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    return [t for t in tokens if t not in string.punctuation and t not in stop_words]

def lemmatize(tokens):
    return [WordNetLemmatizer().lemmatize(t) for t in tokens]

def stem(tokens):
    return [PorterStemmer().stem(t) for t in tokens]

if __name__ == "__main__":
    text = "Tokenization is the process of breaking down text into words and phrases. Stemming and Lemmatization are techniques used to reduce words to their base form."
    tokens = preprocess(text)
    print("Lemmatization:\n", lemmatize(tokens))
    print("\nStemming:\n", stem(tokens))
