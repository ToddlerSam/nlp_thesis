import nltk
import pymongo
import re
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import state_union
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from pymongo import MongoClient
import tweepy
import preprocess
try:
    import numpy
except ImportError:
    pass

import json
import time
import tempfile
import os
import gzip

client = MongoClient()
db = client.thesis

ckey = 'ClkspMmRMOTAJvysssMpylY56'
csecret = '34qxajeFKcPTvMocYT5xMdOTZrQohbOAO1btgpRihJuA1tGxlM'
atoken = '138086873-jMTpMLwOTPKRRDQ9MnlgUvwIollZTadG3i9ZnRlz'
asecret = 'xFdxY1bV2aRaxvy5AJ16i7kCxsVNe6MzThTr5QgJxJzwJ'

def filterUnimportantWords(sentence):
	filtered_sent=[]
	tokenized_sentence = TweetTokenizer().tokenize(sentence)
	stop_words= set(stopwords.words('english'))
	for word in tokenized_sentence:
		if word not in stop_words:
			filtered_sent.append(word)
	return filtered_sent


def createList(filename):
    list = []
    file = open(filename, 'r')
    for line in file:
        line = re.sub('[\n]','',line)
        list.append(line.strip())
    return list

def labelTweets(testSentence):
    positiveWordsFilename='positive_words.txt'
    negativeWordsFileName='negative_words.txt'
    positiveWords=createList(positiveWordsFilename)
    negativeWords=createList(negativeWordsFileName)
    posscore = negscore = 0
    testTokens = filterUnimportantWords(testSentence)
    for token in testTokens:
        if token in positiveWords:
            posscore += 1
        if token in negativeWords:
            negscore += 1

    if posscore > negscore:
            return 'pos'
    if negscore > posscore:
            return 'neg'
    if posscore==negscore:
            return 'neutral'

class listener(tweepy.StreamListener):

    def on_data(self, data):
        if 'retweeted_status' not in data:
            allData = json.loads(data)
            tweet = allData["text"]
            geo = allData["geo"]
            userLocation = allData["user"]["location"]
            username = allData["user"]["screen_name"]
            processedTweet = preprocess.processTweet(tweet)
            sentiment = labelTweets(processedTweet)
            print(processedTweet+'--'+sentiment)
            result = db.trainingSet.insert_one({"tweet": processedTweet,"location": geo,"username":username,"userLocation":userLocation,"sentiment": sentiment})

        return(True)

    def on_error(self, status):
        print(status)

auth = tweepy.OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)
twitterStream = tweepy.Stream(auth, listener())
twitterStream.filter(languages=["en"],track=['ugly black','muslim terrorist','jihad','stopislam','black nigguh','isis'])
