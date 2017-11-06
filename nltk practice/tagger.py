import nltk
from nltk import pos_tag, word_tokenize
text = word_tokenize("And now for somthing completely diffrent")
print(nltk.pos_tag(text))

text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())
print(text.similar('women'))