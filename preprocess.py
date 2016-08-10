import re
import sys
import HTMLParser
#emoticons
mycompile = lambda pat:  re.compile(pat,  re.UNICODE)
SMILEY = mycompile(r'[:=].{0,1}[\)dpD]')
MULTITOK_SMILEY = mycompile(r' : [\)dp]')

NormalEyes = r'[:=]'
Wink = r'[;]'

NoseArea = r'(|o|O|-)'   ## rather tight precision, \S might be reasonable...

HappyMouths = r'[DvV\)\]]'
SadMouths = r'[\(\[]'
Tongue = r'[pP]'
OtherMouths = r'[doO/\\]'  # remove forward slash if http://'s aren't cleaned

Happy_RE =  mycompile( '(\^_\^|' + NormalEyes + NoseArea + HappyMouths + ')')
Sad_RE = mycompile(NormalEyes + NoseArea + SadMouths)

Wink_RE = mycompile(Wink + NoseArea + HappyMouths)
Tongue_RE = mycompile(NormalEyes + NoseArea + Tongue)
Other_RE = mycompile( '('+NormalEyes+'|'+Wink+')'  + NoseArea + OtherMouths )

Emoticon = (
    "("+NormalEyes+"|"+Wink+")" +
    NoseArea +
    "("+Tongue+"|"+OtherMouths+"|"+SadMouths+"|"+HappyMouths+")"
)
Emoticon_RE = mycompile(Emoticon)
#end

#start process_tweet
def processTweet(tweet):
    # process the tweets
     #Convert to lower case
    tweet.replace(" r ", " are ")
    tweet.replace(" u ", " you ")
    tweet.replace(" nt ", " not ")
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    string= ""
    for i in tweet.split():
        if Happy_RE.search(i):
            string+=" happy "
        elif Sad_RE.search(i):
            string+=" sad "
        elif Wink_RE.search(i):
            string+=" wink "
        elif Tongue_RE.search(i):
            string+=" tongue "
        elif Other_RE.search(i):
            string+=" neutral "
        else:
            string+=" "+i

    tweet= string
#Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    #replace emoticons with words
    tweet = HTMLParser.HTMLParser().unescape(tweet)
    return tweet
#end
