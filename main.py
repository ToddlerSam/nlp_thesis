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
import tempfile
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
    bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 40)
    bigram_word_feats = dict([(ngram, True) for ngram in itertools.chain(featureVector, bigrams)])
    return bigram_word_feats


inpTweets = csv.reader(open('sampleTweets.csv', 'rb'), delimiter=',', quotechar='|')
tweets = []
check = []
for row in inpTweets:
    sentiment = row[0]
    category = row[1]
    tweet = row[2]
    bigram_features = bigram_classifier(tweet)
    tweets.append((bigram_features, sentiment))

word_features = get_word_features(get_words_in_tweets(tweets))

# Extract feature vector for all tweets in one shote
# training_set = nltk.classify.util.apply_features(extract_features, tweets)

NBClassifier = nltk.NaiveBayesClassifier.train(tweets)
testTweet = "Muslims are terrorists"
bigram_features = bigram_classifier(testTweet)
# polarity = NBClassifier.classify(extract_features(getFeatureVector(bigram_word_feats)))
polarity = NBClassifier.classify(bigram_features)
print(polarity)
NBClassifier.show_most_informative_features()
ckey = 'ClkspMmRMOTAJvysssMpylY56'
csecret = '34qxajeFKcPTvMocYT5xMdOTZrQohbOAO1btgpRihJuA1tGxlM'
atoken = '138086873-jMTpMLwOTPKRRDQ9MnlgUvwIollZTadG3i9ZnRlz'
asecret = 'xFdxY1bV2aRaxvy5AJ16i7kCxsVNe6MzThTr5QgJxJzwJ'

client = MongoClient()
db = client.thesis

# class listener(StreamListener):
#
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
#                 print(processedTestTweet)
#                 if any(c in processedTestTweet for c in ("ugly", "black", "racist")):
#                     polarity = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet)))
#                     db.data.insert_one( {"tweet": processedTestTweet, "location": location, "polarity": polarity, "category":"Racial Bias" } )
#                 if any(c in processedTestTweet for c in ("muslim", "terrorist")):
#                     polarity = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet)))
#                     db.data.insert_one( {"tweet": processedTestTweet, "location": location, "polarity": polarity, "category":"Religious Bias" } )
#                 if any(c in processedTestTweet for c in ("gay", "homo")):
#                     polarity = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet)))
#                     db.data.insert_one( {"tweet": processedTestTweet, "location": location, "polarity": polarity, "category":"Gender Bias" } )
#
#                 ### HERE: this is the main loop ###
#         return(True)
#
#     def on_error(self, status):
#         print(status)
#
# auth = OAuthHandler(ckey, csecret)
# auth.set_access_token(atoken, asecret)
# twitterStream = Stream(auth, listener())
# twitterStream.filter(track=["ugly black","racist"])
