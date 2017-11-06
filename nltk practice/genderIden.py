#feature extractor
def gender_features(word): return {'last_letter': word[-1]}

#prepare example with its label
from nltk.corpus import names
labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
[(name, 'female') for name in names.words('female.txt')])
import random
random.shuffle(labeled_names)

print(labeled_names)

#divide result example to training and test set
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]
#train classifier by train set
import nltk
classifier = nltk.NaiveBayesClassifier.train(train_set)

#test
print(classifier.classify(gender_features('Hanan')))
#evaluate the classifier on a much larger quantity of unseen data
print(nltk.classify.accuracy(classifier, test_set))
#determine which features it found most effective for distinguishing the names' genders
print(classifier.show_most_informative_features(5))
#^
#likelihoo ratio: useful for comparing different feature-outcome relationships.