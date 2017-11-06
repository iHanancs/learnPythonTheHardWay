from nltk.corpus import movie_reviews
documents = [(list(movie_reviews.words(fileid)), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]
import random
random.shuffle(documents)

#define a feature extractor for documents
# constructing a list of the 2000 most frequent words in the overall corpus
import nltk
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000] 
#define a feature extractor
def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features
    
print(document_features(movie_reviews.words('pos/cv957_8737.txt')))

#divide result example to training and test set
featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]

#train classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)

#evaluate the classifier on a much larger quantity of unseen data
print(nltk.classify.accuracy(classifier, test_set))

#determine which features it found most effective for distinguishing the names' genders
print(classifier.show_most_informative_features(5))