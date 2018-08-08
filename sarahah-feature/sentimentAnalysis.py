import numpy as np

import time
import datetime
import sys, glob, os
from os import listdir
from os.path import isfile, join
import dill
from six.moves import cPickle as pickle

import string

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

import dill
from nltk.corpus import sentiwordnet as swn
import spacy

file_num = 6
for i in range(file_num):
    commentFile = join('userComments', 'comments'+str(i)+'.txt')
    comment_data = open(commentFile, 'r').readlines()


    neg_output = './sentiment/negSentiment' + str(i) + '.txt'
    neg_out = open(neg_output, 'w+')

    pos_output = './sentiment/posSentiment' + str(i) + '.txt'
    pos_out = open(pos_output, 'w+')

    for line in comment_data:
        record = line.strip().split('\t')
        seq_num = int(record[0])
        comment = record[1]
        text = comment.split(' ')

        pos_sentiment_score = []
        neg_sentiment_score = []

        for word in text:
            tweet_synsets = list(swn.senti_synsets(word))
            pos_scores = []
            neg_scores = []

            for s in tweet_synsets:
                neutral_score = s.obj_score()
                if neutral_score >= 0.9:
                    continue
                pos = s.pos_score()
                neg = s.neg_score()

                pos_scores.append(pos)
                neg_scores.append(neg)

            if len(neg_scores) >0:
                neg_sentiment_score.append(np.average(neg_scores))
                pos_sentiment_score.append(np.average(pos_scores))
        if len(pos_sentiment_score) != 0:
            pos_avg = np.average(pos_sentiment_score)
            if not (np.isnan(pos_avg)):
                # pos_sentiment_score.append((seq_num, pos_avg, comment))
                pos_out.write('%f\n' % (pos_avg))
        if len(neg_sentiment_score) != 0:
            neg_avg = np.max(neg_sentiment_score)
            if not (np.isnan(neg_avg)):
                # neg_sentiment_score.append((seq_num, neg_avg, comment))
                neg_out.write('%f\n' % (neg_avg))

    pos_out.close()
    neg_out.close()

