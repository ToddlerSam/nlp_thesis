import re
import nltk
import tweepy
import csv
try:
    import numpy
except ImportError:
    pass
from nltk.corpus import stopwords
import json
import time
import collections
import tempfile
from nltk import precision
from nltk import recall
import os
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import gzip
from collections import defaultdict
from pymongo import MongoClient
from nltk.util import OrderedDict
from nltk.probability import DictionaryProbDist
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
from nltk.classify.api import ClassifierI
from nltk.classify.util import attested_labels, CutoffChecker, accuracy, log_likelihood
from nltk.classify.megam import call_megam, write_megam_file, parse_megam_weights
from nltk.classify.tadm import call_tadm, write_tadm_file, parse_tadm_weights

client = MongoClient()
db = client.thesis
stopWords = set(stopwords.words('english'))
#start process_tweet
def processTweet(tweet):
    # process the tweets
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

    if tweet=="af":
        tweet = "as fuck"
    #trim
    tweet = tweet.strip('\'"')
    return tweet
#end
#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#end

#start getStopWordList
def getStopWordList(stopWordListFileName):
    #read the stopwords file and build a list
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
#end
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)
    return all_words
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features
#start getfeatureVector
def getFeatureVector(tweet):
    featureVector = []
    #split tweet into words
    words = tweet.split()
    for w in words:
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #ignore if it is a stop word
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector
#end

#Read the tweets one by one and process it
def bigram_classifier(tweet):
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet)
    bigram_finder = BigramCollocationFinder.from_words(featureVector)
    try:
        bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 500)
    except:
        print ''
    bigram_word_feats = dict([(ngram, True) for ngram in itertools.chain(featureVector, bigrams)])
    return bigram_word_feats

trainfeats = []
testfeats = []
check = []
inpTweets = db.trainingSet.find()
for row in inpTweets:
    sentiment = row["sentiment"]
    tweet = row["tweet"]
    try:
        bigram_features = bigram_classifier(tweet)
    except:
        print ''
    trainfeats.append((bigram_features, sentiment))

word_features = get_word_features(get_words_in_tweets(trainfeats))
testTweets = db.testSet.find()
NBClassifier = nltk.NaiveBayesClassifier.train(trainfeats)
for doc in testTweets:
    testTweet = doc["tweet"]
    try:
        bigram_features = bigram_classifier(testTweet)
    except:
        print ''
    polarity = NBClassifier.classify(bigram_features)
    testfeats.append((bigram_features, polarity))


refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

for i, (feats, label) in enumerate(trainfeats):
    refsets[label].add(i)

for i, (feats, label) in enumerate(testfeats):
    observed = NBClassifier.classify(feats)
    testsets[observed].add(i)


print(nltk.classify.accuracy(NBClassifier, testfeats))
print(refsets['neutral'])
print(testsets['neutral'])
# print(precision(refsets['pos'], testsets['pos']))
# print('pos precision:', nltk.metrics.precision(refsets['pos'], testsets['pos']))
# print(recall(refsets['pos'], testsets['pos']))
# print(precision(refsets['neg'], testsets['neg']))
# print('neg recall:', nltk.metrics.recall(refsets['neg'], testsets['neg']))
#print(nltk.classify.accuracy(NBClassifier, testTweet))
NBClassifier.show_most_informative_features()
