import re
import nltk
import tweepy
import csv
try:
    import numpy
except ImportError:
    pass

import json
import time
import tempfile
import os
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
    #trim
    tweet = tweet.strip('\'"')
    return tweet
#end

#Read the tweets one by one and process it
fp = open('sampleTweets.txt', 'r')
line = fp.readline()

while line:
    processedTweet = processTweet(line)
    #print processedTweet
    line = fp.readline()
#end loop
fp.close()

stopWords = []

#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
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
fp = open('sampleTweets.txt', 'r')
line = fp.readline()

st = open('stopwords.txt', 'r')
stopWords = getStopWordList('stopwords.txt')

while line:
    processedTweet = processTweet(line)
    featureVector = getFeatureVector(processedTweet)
    #print featureVector
    line = fp.readline()
#end loop
fp.close()

inpTweets = csv.reader(open('sampleTweets.csv', 'rb'), delimiter=',', quotechar='|')
tweets = []
for row in inpTweets:
    sentiment = row[0]
    category = row[1]
    tweet = row[2]
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet)
    tweets.append((featureVector, sentiment));

def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features

inpTweets = csv.reader(open('sampleTweets.csv', 'rb'), delimiter=',', quotechar='|')
stopWords = getStopWordList('stopwords.txt')
featureList = []

# Get tweet words
tweets = []
for row in inpTweets:
    sentiment = row[0]
    category = row[1]
    tweet = row[2]
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet)
    featureList.extend(featureVector)
    tweets.append((featureVector, sentiment));
#end loop

# Remove featureList duplicates
featureList = list(set(featureList))
# Extract feature vector for all tweets in one shote
training_set = nltk.classify.util.apply_features(extract_features, tweets)
# NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
processedTestTweet = processTweet('All muslims are terrorists')
# polarity = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet)))
# print(polarity)


import svm
from svmutil import *
def getSVMFeatureVectorAndLabels(tweets, featureList):
    sortedFeatures = sorted(featureList)
    map = {}
    feature_vector = []
    labels = []
    for t in tweets:
        label = 0
        map = {}
        #Initialize empty map
        for w in sortedFeatures:
            map[w] = 0

        tweet_words = t[0]
        tweet_opinion = t[1]
        #Fill the map
        for word in tweet_words:
            #process the word (remove repetitions and punctuations)
            word = replaceTwoOrMore(word)
            word = word.strip('\'"?,.')
            #set map[word] to 1 if word exists
            if word in map:
                map[word] = 1
        #end for loop
        values = map.values()
        feature_vector.append(values)
        if(tweet_opinion == 'positive'):
            label = 0
        elif(tweet_opinion == 'negative'):
            label = 1
        elif(tweet_opinion == 'neutral'):
            label = 2
        labels.append(label)
    #return the list of feature_vector and labels
    return {'feature_vector' : feature_vector, 'labels': labels}
#end
#training data
labels = [0, 1, 1, 2]
samples = [[0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0]]

#SVM params
param = svm_parameter()
param.C = 10
param.kernel_type = LINEAR
#instantiate the problem
problem = svm_problem(labels, samples)
#train the model
model = svm_train(problem, param)
# saved model can be loaded as below
#model = svm_load_model('model_file')

#save the model
svm_save_model('model_file', model)

#test data
test_data = [[0, 1, 1], [1, 0, 1]]
#predict the labels
p_labels, p_accs, p_vals = svm_predict([0]*len(test_data), test_data, model)
print p_labels


#Train the classifier
result = getSVMFeatureVectorAndLabels(tweets, featureList)
problem = svm_problem(result['labels'], result['feature_vector'])
#'-q' option suppress console output
param = svm_parameter('-q')
param.kernel_type = LINEAR
classifier = svm_train(problem, param)
svm_save_model('model_file', classifier)

#Test the classifier
test_feature_vector = getSVMFeatureVectorAndLabels(processedTestTweet, featureList)
#p_labels contains the final labeling result
p_labels, p_accs, p_vals = svm_predict([0] * len(test_feature_vector),test_feature_vector, classifier)

# ckey = 'ClkspMmRMOTAJvysssMpylY56'
# csecret = '34qxajeFKcPTvMocYT5xMdOTZrQohbOAO1btgpRihJuA1tGxlM'
# atoken = '138086873-jMTpMLwOTPKRRDQ9MnlgUvwIollZTadG3i9ZnRlz'
# asecret = 'xFdxY1bV2aRaxvy5AJ16i7kCxsVNe6MzThTr5QgJxJzwJ'

# client = MongoClient()
# db = client.thesis

# class listener(StreamListener):

#     def on_data(self, data):
#         if 'retweeted_status' not in data:
#             allData = json.loads(data)
#             #print(allData)
#             tweet = allData["text"]
#             geo = allData["coordinates"]
#             result = db.tweets.insert_one( { "tweet": tweet, "location": geo} )
#             cursor = db.tweets.find()
#             for document in cursor:
#                 testTweet = document["tweet"]
#                 location = document["location"]
#                 processedTestTweet = processTweet(testTweet)
#                 print processedTestTweet
#                 if any(c in processedTestTweet for c in ("ugly", "black", "racist")):
#                     polarity = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet)))
#                     db.data.insert_one( {"tweet": processedTestTweet, "location": location, "polarity": polarity, "category":"Racial Bias" } )
#                 if any(c in processedTestTweet for c in ("muslim", "terrorist")):
#                     polarity = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet)))
#                     db.data.insert_one( {"tweet": processedTestTweet, "location": location, "polarity": polarity, "category":"Religious Bias" } )
#                 if any(c in processedTestTweet for c in ("gay", "homo")):
#                     polarity = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet)))
#                     db.data.insert_one( {"tweet": processedTestTweet, "location": location, "polarity": polarity, "category":"Gender Bias" } )

#                 ### HERE: this is the main loop ###
#         return(True)

#     def on_error(self, status):
#         print(status)

# auth = OAuthHandler(ckey, csecret)
# auth.set_access_token(atoken, asecret)
# twitterStream = Stream(auth, listener())
# twitterStream.filter(track=["ugly black","racist"])