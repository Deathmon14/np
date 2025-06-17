import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
nltk.download('averaged_perceptron_tagger')

text = "The cat sat on the mat."
tokens = word_tokenize(text)


    tagged = []
    for word in t:
        if word.lower() == "cat":
            tagged.append((word, "NN")) 
        elif word.lower() == "sat":
            tagged.append((word, "VB")) 
        elif word.lower() == "the":
            tagged.append((word, "DT")) 
        else:
            tagged.append((word, "UNKNOWN"))
    return tagged

print("Rule-based PoS tagging:", rule_based_pos(tokens))

statistical_tags = pos_tag(tokens)
print("Statistical PoS tagging:", statistical_tags)
