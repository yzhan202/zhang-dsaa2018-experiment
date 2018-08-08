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


def remove_non_ascii_characters(string_in):
    stripped = [c for c in string_in if 0 < ord(c) < 127]
    return ''.join(stripped)


tweet_list = []

# file = '/home/yue/Public/C_program/cl2_project/SeededLDA/data/sarahah_data/new_result3/topics/topic0.txt'
       # './topics/haterAndAbuse_topic.txt'
# file = './dataset/stem_data.txt'
# file = './genders/male.txt'
# file = './userComments/comments0.txt'

file = './formspring_data.txt'
header = ['database_id', 'userid', 'originalText', 'ProcessedText', 'asker']

data = open(file, 'r').readlines()
idx = -1
for line in data:
    idx += 1
    if idx == 0:
        continue
    record = line.strip().split('\t')
    # print record
    seq_num = record[0]
    clean_tweet = record[3]
    # print clean_tweet
    # if len(clean_tweet) == 0:
    #     print
    tweet_list.append(clean_tweet)



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

no_features = 2000
# tf_vectorizer = CountVectorizer(stop_words=stopwords_set, ngram_range=(1, 1), encoding='utf-8', lowercase=True)
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(tweet_list)
tf_feature_names = tf_vectorizer.get_feature_names()

out_dir = '/home/yue/Public/C_program/cl2_project/SeededLDA/data/formspring_data'   # /result3/subtopic1'
docIdx_file = join(out_dir, 'docIdx.txt')
wordIdx_file = join(out_dir, 'wordIdx.txt')
srcWords_file = join(out_dir, 'srcWords.txt')

docIdx_out = open(docIdx_file, 'w+')
wordIdx_out = open(wordIdx_file, 'w+')
srcWords_out = open(srcWords_file, 'w+')

for i in range(len(tf_feature_names)):
    feature = str(tf_feature_names[i])
    srcWords_out.write('%s\n' % feature)
srcWords_out.close()

tf_dense = tf.toarray()
doc_num = tf_dense.shape[0]
print doc_num

for i in range(doc_num):
    doc_tf = tf_dense[i, :]
    nonzeroIdx = (np.nonzero(doc_tf))[0]
    frequency = doc_tf[nonzeroIdx]
    for j in range(len(nonzeroIdx)):
        idx = nonzeroIdx[j]
        freq = frequency[j]
        for t in range(freq):
            wordIdx_out.write('%d\n' % (idx + 1))
            docIdx_out.write('%d\n' % (i + 1))
wordIdx_out.close()
docIdx_out.close()
