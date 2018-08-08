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


true_post = 0
true_nega = 0
false_post = 0
false_nega = 0

topic_dir = './topics/' # genderTopic
femaleTopic_file = join(topic_dir, 'topic0.txt')
maleTopic_file = join(topic_dir, 'topic1.txt')

groundTruth_dir = './genders'
femaleTruth_file = join(groundTruth_dir, 'female.txt')
maleTruth_file = join(groundTruth_dir, 'male.txt')

femaleTopic_data = open(femaleTopic_file, 'r').readlines()
maleTopic_data = open(maleTopic_file, 'r').readlines()

femaleTruth_data = open(femaleTruth_file, 'r').readlines()
maleTruth_data = open(maleTruth_file, 'r').readlines()

femaleTruth_ids = []
for line in femaleTruth_data:
    record = line.strip().split('\t')
    id = int(record[0])
    femaleTruth_ids.append(id)

maleTruth_ids = []
for line in maleTruth_data:
    record = line.strip().split('\t')
    id = int(record[0])
    maleTruth_ids.append(id)

# femaleTopic_ids = []
for line in femaleTopic_data:
    record = line.strip().split('\t')
    id = int(record[0])
    if id in femaleTruth_ids:
        true_nega += 1
    elif id in maleTruth_ids:
        false_nega += 1

for line in maleTopic_data:
    record = line.strip().split('\t')
    id = int(record[0])
    if id in maleTruth_ids:
        true_post += 1
    elif id in femaleTruth_ids:
        false_post += 1

print('TN: %d, FN: %d\n' % (true_nega, false_nega))
print('TP: %d, FP: %d\n' % (true_post, false_post))

recall_man = (true_post)* 1.0 / (true_post+ false_nega)

precision_man = (true_post)* 1.0 / (true_post+ false_post)

recall_weman = (true_nega)* 1.0 / (false_post+ true_nega)

precision_weman = (true_nega)* 1.0 / (true_nega+ false_nega)


acc = (true_post+true_nega)*1.0 / (true_post+true_nega+ false_post+ false_nega)

print('For man recall: %f, precision: %f\n' % (recall_man, precision_man))
print('For women recall: %f, precision: %f\n' % (recall_weman, precision_weman))

print('accuracy:%f' % acc)