# coding: utf-8
#CatherineS 20180727
##################################
#READ ME BEFORE USE THIS
""" this .py file input is one month.csv
    output is the df which contains the column of sentiment score
    change relevant to this file
    #inputParameters.read_directory
"""

#CHANGE THIS PART ABOVE
##################################
#part1: data pre processing
##################################
'''import inputParameters
import pandas as pd
import numpy as np
from textblob import TextBlob
import os
import re
import string
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import spacy
from spacy.lemmatizer import Lemmatizer
#for sentiment analysis
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from textblob.sentiments import NaiveBayesAnalyzer
from textblob import Blobber
###Read file'''
from lib_import import *

### Step2: cleaning FUNCTIONS
#1.Remove Unicode Strings and Noise and lower the text
def removeUnicode(text):
    """ Lower the text and Removes unicode strings like "\u002c" and "x96" """
    text = text.lower()
    text = re.sub(r'(\\u[0-9A-Fa-f]+)',r'', text)
    text = re.sub(r'[^\x00-\x7f]',r'',text)
    return text

#2. Replace URLs, User Mentions and Hashtags
def replaceURL_AT_HASH(text):
    """ Replaces url address with the string"url" """
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','url',text)
    text = re.sub('@[^\s]+','atUser',text)
    text = re.sub(r'#([^\s]+)', r'\1', text)
    return text

#3. Replace Repetitions of Punctuation
# Replace Negations with Antonyms
def replaceMultiMarks(text):
    """ Replaces repetitions of exlamation marks """#multiExclamation
    text = re.sub(r"(\!)\1+", ' multiExclamation ', text)
    #""" Replaces repetitions of question marks """# multiQuestion
    text = re.sub(r"(\?)\1+", ' multiQuestion ', text)
    #""" Replaces repetitions of stop marks """#multiStop
    text = re.sub(r"(\.)\1+", ' multiStop ', text)
    return text

#4.Replace Contractions#web search
""" Replaces contractions from a string to their equivalents """
#this can be replaced in the future by a list
contraction_patterns = [ (r'won\'t', 'will not'), (r'can\'t', 'cannot'), (r'i\'m', 'i am'), (r'ain\'t', 'is not'), (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'),
                         (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'s', '\g<1> is'), (r'(\w+)\'re', '\g<1> are'), (r'(\w+)\'d', '\g<1> would'), (r'&', 'and'), (r'dammit', 'damn it'), (r'dont', 'do not'), (r'wont', 'will not') ]
def replaceContraction(text):
    patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]
    for (pattern, repl) in patterns:
        (text, count) = re.subn(pattern, repl, text)
    return text

#5. Replcae Slang and Abbreviations
df= pd.read_csv('slang.txt',sep='\t',header=None,names=['abv','full'])
df.dropna(inplace=True)
df['full']=df['full'].apply(removeUnicode)
abvdic = df.set_index('abv')['full'].to_dict()#this list remains to be fullfilled
slang_words = list(abvdic.keys())
#construct the dictionary
def replaceABV(text):
    tokens = nltk.word_tokenize(text)
    replacedoutput = []
    for word in tokens:
        if word in slang_words:
            replacedoutput.append(abvdic.get(word))
        else:
            replacedoutput.append(word)
    return ' '.join(replacedoutput)

#6. Replace Negations with Antonyms
def replace(word, pos=None):
    """ Creates a set of all antonyms for the word and if there is only one antonym, it returns it """
    antonyms = set()
    for syn in wordnet.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            for antonym in lemma.antonyms():
                antonyms.add(antonym.name())
    if len(antonyms) == 1:
        return antonyms.pop()
    else:
        return None

def replaceNegations(text):
    """ Finds "not" and antonym for the next word and if found, replaces not and the next word with the antonym """
    tokens = nltk.word_tokenize(text)
    i, l = 0, len(tokens)
    words = []
    while i < l:
        word = tokens[i]
        if word == 'not' and i+1 < l:
            ant = replace(tokens[i+1])
            if ant:
                words.append(ant)
                i += 2
                continue
        words.append(word)
        i += 1
    return ' '.join(words)

#7.Elongated word removement
word = 'slangggggg'
def replaceElongated(word):
    """ Replaces an elongated word with its basic form, unless the word exists in the lexicon """

    repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
    repl = r'\1\2\3'
    if wordnet.synsets(word):
        return word
    repl_word = repeat_regexp.sub(repl, word)
    if repl_word != word:
        return replaceElongated(repl_word)
    else:
        return repl_word

#8. spacy for lemmitizr

nlp = spacy.load('en_core_web_sm')
def spacy_lemmatizer(text):
    """lemmetizr"""
    sent = []
    doc = nlp(text)
    for word in doc:
        sent.append(word.lemma_)
    return " ".join(sent)

#9.remove stopwords
#9. Remove stopwords and punctuations after this #this should be the last step of all


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
exclude = set(string.punctuation)


def removeStopwords_Punc(text):
    """remove stopwords"""
    r =  [word for word in text.split() if word not in stop_words]
    s = [ch for ch in r if ch not in exclude]
    return ' '.join(s)

def textProcessing_Pipeline(text,Logger):
    """Finally: Process ALL Step"""
    #1""" Lower the text and Removes unicode strings like "\u002c" and "x96" """
    try:
        text = removeUnicode(text)
    except Exception as e:
        Logger.write("removeUnicode failed, Error: {}".format(e))
    #2 """ Replaces url address with the string"url" """
    try:
        text = replaceURL_AT_HASH(text)
    except Exception as e:
        Logger.write("replace url failed, Error: {}".format(e))
    #3""" Replaces repetitions of exlamation marks """#multiExclamation
    try:
        text = replaceMultiMarks(text)
    except Exception as e:
        Logger.write("replace multimarks failed: {}".format(e))
    #4""" Replaces contractions from a string to their equivalents """
    try:
        text = replaceContraction(text)
    except Exception as e:
        Logger.write("rreplace Contraction failed: {}".format(e))
    #5""" Replcae Slang and Abbreviations"""
    text = replaceABV(text)
    #6""" Finds "not" and antonym for the next word and if found, replaces not and the next word with the antonym """
    word = replaceNegations(text)
    #7 """ Replaces an elongated word with its basic form, unless the word exists in the lexicon """
    text = replaceElongated(word)
    #8 """chosing spacy for lemmitizer"""
    text = spacy_lemmatizer(text)
    #9 """emove stopwords and punctuations after this """
    text = removeStopwords_Punc(text)
    return text

#Below is run the specific code
tb = Blobber(analyzer=NaiveBayesAnalyzer())
def GetSentiment_final(inputfilename,Logger):
    """use textprocessingsentiment.GetSentiment_final(inputfilename) in the run files
       input .csv -> output df
    """
    df = pd.read_csv(os.path.join(inputParameters.read_directory, inputfilename))
    print('Readin csv file successfully {}'.format(inputfilename))
    Logger.write("Readin csv file successfully")
    #the wrong one: df = pd.read_csv(inputParameters.read_directory+inputfilename)# ,engine='python' , encoding= 'uft-8')
    ### Step1: filter out the tweet without the @inside
    filtered_df = df[df['mentions'].notnull()]
    #add a new columns by process this
    #before:     filtered_df['clean_text']=filtered_df.tweet.apply(lambda x: textProcessing_Pipeline(x))
    #filtered_df['clean_text']=filtered_df.loc[:,('tweet')].apply(lambda x: textProcessing_Pipeline(x))
    #.loc[:,('one','second')]
    filtered_df['clean_text']=filtered_df.tweet.apply(lambda x: textProcessing_Pipeline(x,Logger))
    Logger.write("textProcessing_Pipeline: finished")

    ### Step2: data sentiment analysis
    #add the general sentiment columns
    try:
        filtered_df[['nb_sentiment', 'nb_pos', 'nb_neg']] = filtered_df.clean_text.apply(lambda v: pd.Series(tb(v).sentiment))
    except Exception as e:
        print('Sentiment analysis with tb naiveBayesAnalyzer has problems {}'.fsormat(e))
        Logger.write("Sentiment analysis with tb naiveBayesAnalyzer successfully {}".format(e))

    return filtered_df, Logger
