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


ckey = 'ClkspMmRMOTAJvysssMpylY56'
csecret = '34qxajeFKcPTvMocYT5xMdOTZrQohbOAO1btgpRihJuA1tGxlM'
atoken = '138086873-jMTpMLwOTPKRRDQ9MnlgUvwIollZTadG3i9ZnRlz'
asecret = 'xFdxY1bV2aRaxvy5AJ16i7kCxsVNe6MzThTr5QgJxJzwJ'

client = MongoClient()
db = client.thesis

class listener(StreamListener):

    def on_data(self, data):
        allData = json.loads(data)
        # print(allData)
        tweet = allData["text"]
        user = allData["user"]["screen_name"]
        result = db.tweets.insert_one( { "tweet": tweet, "user": user } )
        return(True)

    def on_error(self, status):
        print(status)


auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)
twitterStream = Stream(auth, listener())
twitterStream.filter(track=["ugly black", "jihad", "muslim terrorist", "gay nigguh"])

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
    print processedTweet
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
    print featureVector
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

NBClassifier = nltk.NaiveBayesClassifier.train(training_set)

# Test the classifier
testTweet = 'Congrats @ravikiranj, i heard you wrote a new tech post on sentiment analysis'
processedTestTweet = processTweet(testTweet)
print NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet)))

# max entropy-------------------------
# from nltk.classify import MaxentClassifier
# MaxEntClassifier = MaxentClassifier.train(training_set)
# testTweet = 'Congrats @ravikiranj, i heard you wrote a new tech post on sentiment analysis'
# processedTestTweet = processTweet(testTweet)
# print MaxEntClassifier.classify(extract_features(getFeatureVector(processedTestTweet)))
#
# print MaxEntClassifier.show_most_informative_features(10)
# max entropy-------------------------
