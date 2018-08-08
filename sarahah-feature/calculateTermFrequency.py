import nltk
import string
import os

from os.path import isfile, join

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

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


topic0_file = '/home/yue/Public/C_program/cl2_project/SeededLDA/data/sarahah_data' \
              '/new_result3/topics/topic0.txt'
#'topics/explicit_se*_topic.txt'
topic1_file = '/home/yue/Public/C_program/cl2_project/SeededLDA/data/sarahah_data' \
              '/new_result3/topics/topic1.txt'
#'topics/haterAndAbuse_topic.txt'
topic2_file = '/home/yue/Public/C_program/cl2_project/SeededLDA/data/sarahah_data' \
              '/new_result3/subtopic2/topics/topic2.txt'
#'topics/flirting_topic.txt'
topic3_file = '/home/yue/Public/C_program/cl2_project/SeededLDA/data/sarahah_data' \
              '/new_result3/subtopic2/topics/topic3.txt'

nonBully_file1 = '/home/yue/Public/C_program/cl2_project/SeededLDA/data/sarahah_data' \
                 '/new_result3/subtopic2/topics/rest_topic.txt'
rest = './dataset/removed_data.txt'

no_features = 5000
tf_vectorizer = CountVectorizer(max_features=no_features, stop_words='english')

topic0CommentList = []

topic0_data = open(topic1_file, 'r').readlines()

idx = 0
for line in topic0_data:
    record = line.strip().split('\t')
    seq_num = int(record[0])
    value = stem_dict[seq_num]
    comment = value[1]
    topic0CommentList.append(comment)
topic0CommentList = ''.join(topic0CommentList)

# print topic0CommentList

# idx = 0
# rest_data = open(topic1_file, 'r').readlines()
# for line in rest_data:
#     record = line.strip().split('\t')
#     seq_num = int(record[0])
#     try:
#         comment = record[2]
#         topic0CommentList.append(comment)
#         idx += 1
#     except:
#         continue
# topic0CommentList = ''.join(topic0CommentList)

tf = tf_vectorizer.fit_transform([topic0CommentList])
tf_feature_names = tf_vectorizer.get_feature_names()
tf_dense = tf.toarray()[0]

#
# idxs = tf_dense.argsort()[-20:][::-1]
#
# word_list = []
# tf_list = []
# for id in idxs:
#     freq = tf_dense[id]
#     print id, freq
#     word = str(tf_feature_names[id])
#     word_list.append(word)
#     tf_list.append(freq)
#
# print word_list
# print tf_list

wordList = ['sex','shit','ass','fuck','gay','bitch','butt','boob','hate','suck',
                   'bulli', 'hater', 'sad', 'disgust', 'bad', 'ugli', 'wtf']
counts = 0
for w in wordList:
    try:
        idx = tf_feature_names.index(w)
        num = tf_dense[idx]
        counts += num
    except:
        print w

print counts

# topic0: 73
# topic1: 46
