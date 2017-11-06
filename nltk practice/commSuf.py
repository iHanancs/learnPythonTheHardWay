from nltk.corpus import brown
import nltk
suffix_fdis = nltk.FreqDist()
print(suffix_fdis)

for word in brown.words():
	word = word.lower()
	suffix_fdis[word[-1:]] +=1
	suffix_fdis[word[-2:]] +=1
	suffix_fdis[word[-3:]] +=1
	
common_suffixes = [suffix for (suffix, count) in suffix_fdis.most_common(100)]
print(common_suffixes)

def pos_features(word):
     features = {}
     for suffix in common_suffixes:
         features['endswith({})'.format(suffix)] = word.lower().endswith(suffix)
     return features
     
tagged_words = brown.tagged_words(categories='news')
featuresets = [(pos_features(n), g) for (n,g) in tagged_words]
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]

classifier = nltk.DecisionTreeClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))

print(classifier.classify(pos_features('cats')))
print(classifier.classify(pos_features('hanan')))
print(classifier.classify(pos_features('reed')))