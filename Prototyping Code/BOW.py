# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 21:23:07 2018

@author: Sean
"""
import json
import os
import nltk
import random
from nltk.corpus import stopwords
from nltk.corpus import opinion_lexicon
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

x = []
y = []

file_list = os.listdir('data')
for file in file_list:
    with open('data/' + file, 'r') as f:
        transcripts = json.load(f)
        x.extend(transcripts['text'].values())
        y.extend(transcripts['sentiment'].values())
        
#clean all the transcripts (stop words and stemming)
stop_words = set(stopwords.words('english'))
stop_words.add(',')
stop_words.add('.')
ps = PorterStemmer()
filtered_x = []

for t in x:
     t = t.lower()
     t_tokens = word_tokenize(t)
     stopworded_t = [w for w in t_tokens if not w in stop_words]
     stemmed_t = []
     for word in stopworded_t:
        stemmed_t.append(ps.stem(word))
     filtered_x.append(stemmed_t)
     
# collect fetures
all_words = []

for row in filtered_x:
    for w in row:
        all_words.append(w)

all_words = nltk.FreqDist(all_words)


word_features = list(all_words.keys())[:1500]

# create featuresets
def find_features(t):
    features = {}
    for w in word_features:
        features[w] = (w in t)

    return features

featuresets = []
for i in range(len(y)):
    featuresets.append((find_features(filtered_x[i]), y[i]))



random.shuffle(featuresets)


#partitioning(80/20, 622 reviews in total)
training_set = featuresets[:497]
testing_set =  featuresets[498:]


# Model and test
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)
