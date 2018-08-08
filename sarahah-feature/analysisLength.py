import csv
import numpy as np
from os.path import isfile, join
from six.moves import cPickle as pickle

import string
import pandas as pd

from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import os
import datetime
import time

import enchant


tokenizer = RegexpTokenizer(r'\w+')  # r'[a-zA-Z]+'

fileList = ['EnglishData.txt', 'NonEnglishData.txt', 'UserSpecificData.txt']
header = ['id', 'original_tweet', 'user_comment']

for file in fileList:
    data = open(join('dataset/',file), 'r').readlines()
    len_list = []
    for line in data:
        record = line.strip().split('\t')
        tweet = record[1]
        tokens = tokenizer.tokenize(tweet)
        len_list.append(len(tokens))

print np.mean(len_list)