import nltk
sentence = """At eight o'clock on Thursday morning
... Arthur didn't feel very good."""
tokens = nltk.word_tokenize(sentence)
print(tokens)
tagged = nltk.pos_tag(tokens)
print(tagged[0:6])
entities = nltk.chunk.ne_chunk(tagged)
print(entities)
from nltk.draw.tree import draw_trees
draw_trees(entities)

