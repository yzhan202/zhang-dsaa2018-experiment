import csv
import numpy as np
from os.path import isfile, join
from six.moves import cPickle as pickle

import string
import pandas as pd

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

stem_dict = {}
stem_file = './dataset/stem_data.txt'
stem_data = open(stem_file, 'r').readlines()
for line in stem_data:
    record = line.strip().split('\t')
    seq_num = int(record[0])
    tweet = record[1]
    try:
        comment = record[2]
    except:
        comment = ''
    stem_dict[seq_num] = (tweet, comment)


seqList = []
topic0_file = 'topics/explicit_se*_topic.txt'
topic1_file = 'topics/haterAndAbuse_topic.txt'
fileList = [topic0_file, topic1_file]

for f in fileList:
    topic_data = open(f, 'r').readlines()
    for line in topic_data:
        record = line.strip().split('\t')
        seq_num = int(record[0])
        seqList.append(seq_num)
seqList = list(set(seqList))

output = 'topics/flirting_topic.txt'
out = open(output, 'w+')
for (seq_num, value) in stem_dict.items():
    if seq_num not in seqList:
        tweet = value[0]
        comment = value[1]
        out.write('%d\t%s\t%s\t\n' % (seq_num, tweet, comment))
out.close()


