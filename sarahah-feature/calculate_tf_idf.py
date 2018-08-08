

import nltk
import string
import os

from os.path import isfile, join

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer


tfidf = TfidfVectorizer(stop_words='english')


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


groundTruth_dir = './genders'
femaleTruth_file = join(groundTruth_dir, 'female.txt')
maleTruth_file = join(groundTruth_dir, 'male.txt')

femaleTruth_data = open(femaleTruth_file, 'r').read().translate(None, string.punctuation)
maleTruth_data = open(maleTruth_file, 'r').read().translate(None, string.punctuation)


token_dict = {femaleTruth_file: femaleTruth_data, maleTruth_file: maleTruth_data}

tfs = tfidf.fit_transform(token_dict.values())
# print tfs[0]

tfs_female = tfs[0].toarray()[0]
tfs_male = tfs[1].toarray()[0]
# print tfs_female

male_idxs = tfs_male.argsort()[-20:][::-1]
female_idxs = tfs_female.argsort()[-20:][::-1]

# print tfs_male[male_idxs]

feature_names = tfidf.get_feature_names()

male_words = []
female_words = []

for id in male_idxs:
    word = feature_names[id]
    male_words.append(word)

for id  in female_idxs:
    word = feature_names[id]
    female_words.append(word)

print male_words
print female_words

