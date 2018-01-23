# -*- coding:utf-8 -*-
import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk.text import TextCollection
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import jieba
import codecs
from nltk import FreqDist, ConditionalFreqDist
import pickle
import os
from nltk.classify.scikitlearn import SklearnClassifier
import sklearn
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import itertools
import zipfile

import sys
from os import path
cur_file = path.dirname(__file__)
sys.path.append(cur_file)
from readfile_test import *


# 解压数据包
def unpack_data(zip_file, data_dir):
    if path.isdir(data_dir):
        return
    else:
        os.mkdir(data_dir)
    f = zipfile.ZipFile(zip_file, 'r')
    for zfile in f.namelist():
        f.extract(zfile, data_dir)
    f.close()


# 单词
def bag_of_word(words):
    return dict([(word, True) for word in words])


# 双词
def bigram(words, score_fn = BigramAssocMeasures.chi_sq, n = 2000):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_word(bigrams)


# 单词和双词
def bigram_words(words, score_fn = BigramAssocMeasures.chi_sq, n = 2000):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_word(words + bigrams)
    #return bigram(words + bigrams)


# 结巴分词
def get_jieba_cut_words(filename):
    with codecs.open(filename, 'r', "GBK") as f:
        tmp_line = f.read()
        jieba_cut = jieba.cut(tmp_line)
        ans = ' '.join(jieba_cut).strip().replace("\n", "").replace("\r", "")
        return ans


# 结巴分词
def get_jieba_cut_words1(filename):
    [status, utf8_data, gbk_data] = get_utf8_data(filename)
    if not status:
        raise Exception("can not decode text")
    jieba_cut = jieba.cut(gbk_data)
    ans = ' '.join(jieba_cut).strip().replace("\n", "").replace("\r", "")
    return ans


# 结巴分词
def get_jieba_cut_words2(text):
    jieba_cut = jieba.cut(text)
    ans = ' '.join(jieba_cut).strip().replace("\n", "").replace("\r", "")
    return ans

# 加载停用词
def load_stopwords(filename):
    with codecs.open(filename, 'r', "utf-8") as f:
        lines = f.readlines()
        return [line.strip().replace("\r", "").replace("\n", "") for line in lines]

# 获取词频
def get_word_freq(words):
    return FreqDist(words)


def get_word_scores(pos_words, neg_words):
    pos_words_plain = list(itertools.chain(*pos_words))
    neg_words_plain = list(itertools.chain(*neg_words))
    word_fd = FreqDist(pos_words_plain + neg_words_plain)  # 可统计所有词的词频
    pos_word_fd = FreqDist(pos_words_plain)
    neg_word_fd = FreqDist(neg_words_plain)

    pos_word_count = pos_word_fd.N()  # 积极词的数量
    neg_word_count = neg_word_fd.N()  # 消极词的数量
    #total_word_count = pos_word_count + neg_word_count
    total_word_count = word_fd.N()

    word_scores = {}
    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(pos_word_fd[word], (freq, pos_word_count), total_word_count)  # 计算积极词的卡方统计量，这里也可以计算互信息等其它统计量
        neg_score = BigramAssocMeasures.chi_sq(neg_word_fd[word], (freq, neg_word_count), total_word_count)  # 同理
        word_scores[word] = pos_score + neg_score  # 一个词的信息量等于积极卡方统计量加上消极卡方统计量
    return word_scores  # 包括了每个词和这个词的信息量


def get_word_bigram_scores(pos_words, neg_words):
    pos_words_plain = list(itertools.chain(*pos_words))
    neg_words_plain = list(itertools.chain(*neg_words))

    bigram_finder = BigramCollocationFinder.from_words(pos_words_plain)
    pos_bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)
    bigram_finder = BigramCollocationFinder.from_words(neg_words_plain)
    neg_bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)

    pos = pos_words_plain + pos_bigrams  # 词和双词搭配
    neg = neg_words_plain + neg_bigrams
    all_words = pos + neg

    pos_word_fd = FreqDist(pos)
    neg_word_fd = FreqDist(neg)
    word_fd = FreqDist(all_words)

    pos_word_count = pos_word_fd.N()  # 积极词的数量
    neg_word_count = neg_word_fd.N()  # 消极词的数量
    #total_word_count = pos_word_count + neg_word_count
    total_word_count = word_fd.N()

    word_scores = {}
    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(pos_word_fd[word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(neg_word_fd[word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score
    return word_scores


def find_best_words(word_scores, number):
    # 把词按信息量倒序排序。number是特征的维度，是可以不断调整直至最优的
    best_vals = sorted(word_scores.items(), key=lambda w_s: w_s[1], reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words


def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])


def get_pos_features(feature_extraction_method, pos_review):
    features = []
    for i in pos_review:
        pos_words = [feature_extraction_method(i), 'pos']    # 为积极文本赋予"pos"
        features.append(pos_words)
    return features


def get_neg_features(feature_extraction_method, neg_review):
    features = []
    for j in neg_review:
        neg_words = [feature_extraction_method(j), 'neg']    # 为消极文本赋予"neg"
        features.append(neg_words)
    return features


def get_dataset(filelist):
    neg_test_file = filelist[0]
    stopwords_file = filelist[1]
    stopwords = load_stopwords(stopwords_file)    # 加载停用词
    neg_test_data = get_jieba_cut_words1(neg_test_file)    # 结巴分词，并去掉空和换行符
    neg_test_data = [word for word in neg_test_data.split(" ") if word not in stopwords]    # 去掉所有停用词
    #print(neg_test_data)
    return neg_test_data


def get_dir_dataset(dir, savename):
    stopwords = load_stopwords(stopwords_file)    # 加载停用词
    [utf8_data_list, gbk_data_list] = get_utf8_data_dir(dir)
    cut_words = []
    for data in gbk_data_list:
        jieba_cut = jieba.cut(data)
        ans = ' '.join(jieba_cut).strip().replace("\n", "").replace("\r", "")
        result = [word for word in ans.split(" ") if word not in stopwords]    # 去掉所有停用词
        cut_words.append(result)
    pickle.dump(cut_words, open(savename, 'wb'))


def score(classifier):
    classifier = SklearnClassifier(classifier)
    classifier.train(train)
    pred = classifier.classify_many(dev_test_data)
    return accuracy_score(dev_test_tag, pred)


def final_score(classifier):
    classifier = SklearnClassifier(classifier)
    classifier.train(train)
    pred = classifier.classify_many(test)
    return accuracy_score(tag_test, pred)

if __name__ == '__main__':
    unpack_data('data.zip', 'data')
    stopwords_file = r'data/stopwords.txt'
    # neg file and processed data
    neg_file_dir = r'data/neg'
    neg_savename = r"data/neg_dataset"
    if not os.path.exists(neg_savename):
        get_dir_dataset(neg_file_dir, neg_savename)
    neg_test_data = pickle.load(open(neg_savename, 'rb'))

    # pos file and processed data
    pos_file_dir = r'data/pos'
    pos_savename = r"data/pos_dataset"
    if not os.path.exists(pos_savename):
        get_dir_dataset(pos_file_dir, pos_savename)
    pos_test_data = pickle.load(open(pos_savename, 'rb'))

    # all text
    pos_review = pos_test_data
    neg_review = neg_test_data
    pos_size = len(pos_review)
    neg_size = len(neg_review)
    if pos_size > neg_size:
        total_size = neg_size
        pos_review = pos_review[:neg_size]
    else:
        total_size = pos_size
        neg_review = neg_review[:pos_size]

    print(len(pos_test_data))
    print(len(neg_test_data))

    # data set size
    train_size = int(total_size * 0.8)
    dev_test_size = int(total_size * 0.9)
    test_size = total_size

    # 5 个不通的方式
    # 1
    #pos_features = get_pos_features(bag_of_word, pos_review)
    #neg_features = get_neg_features(bag_of_word, neg_review)
    # 2
    #pos_features = get_pos_features(bigram, pos_review)
    #neg_features = get_neg_features(bigram, neg_review)
    # 3
    pos_features = get_pos_features(bigram_words, pos_review)
    neg_features = get_neg_features(bigram_words, neg_review)
    # 4
    #word_score = get_word_scores(pos_review, neg_review)
    #best_words = find_best_words(word_score, 1500)
    #pos_features = get_pos_features(best_word_features, pos_review)
    #neg_features = get_neg_features(best_word_features, neg_review)
    # 5
    #word_score = get_word_bigram_scores(pos_review, neg_review)
    #best_words = find_best_words(word_score, 1500)
    #pos_features = get_pos_features(best_word_features, pos_review)
    #neg_features = get_neg_features(best_word_features, neg_review)

    # seperate data set
    #train = pos_features[0:train_size] + neg_features[0:train_size]
    #dev_test = pos_features[train_size:dev_test_size] + neg_features[train_size:dev_test_size]
    #test = pos_features[dev_test_size:] + neg_features[dev_test_size:]

    #dev_test_data, dev_test_tag = zip(*dev_test)
    ##print('BernoulliNB`s accuracy is %f' % score(BernoulliNB(), train, dev_test_data, dev#_test_tag))
    #print 'BernoulliNB`s accuracy is %f' % score(BernoulliNB())
    #print 'MultinomiaNB`s accuracy is %f' % score(MultinomialNB())
    #print 'LogisticRegression`s accuracy is %f' % score(LogisticRegression())
    #print 'SVC`s accuracy is %f' % score(SVC())
    #print 'LinearSVC`s accuracy is %f' % score(LinearSVC())
    #print 'NuSVC`s accuracy is %f' % score(NuSVC())
    #print("done")

    # 不同数量的特征值对预测结果的影响
    #dimension = [1000]
    #for d in dimension:
    #    word_score = get_word_scores(pos_review, neg_review)
    #    best_words = find_best_words(word_score, d)
    #    pos_features = get_pos_features(best_word_features, pos_review)
    #    neg_features = get_neg_features(best_word_features, neg_review)
    #    train = pos_features[0:train_size] + neg_features[0:train_size]
    #    dev_test = pos_features[train_size:dev_test_size] + neg_features[train_size:dev_test_size]
    #    test = pos_features[dev_test_size:] + neg_features[dev_test_size:]
    #    dev_test_data, dev_test_tag = zip(*dev_test)
    #    print 'BernoulliNB`s accuracy is %f' % score(BernoulliNB())
    #    print 'MultinomiaNB`s accuracy is %f' % score()
    #    print 'LogisticRegression`s accuracy is %f' % score(LogisticRegression())
    #    print 'SVC`s accuracy is %f' % score(SVC())
    #    print 'LinearSVC`s accuracy is %f' % score(LinearSVC())
    #    print 'NuSVC`s accuracy is %f' % score(NuSVC())
    #    print("done")

    #word_score = get_word_bigram_scores(pos_review, neg_review)
    #best_words = find_best_words(word_score, 2000)
    #pos_features = get_pos_features(best_word_features, pos_review)
    #neg_features = get_neg_features(best_word_features, neg_review)

    train = pos_features[0:train_size] + neg_features[0:train_size]
    dev_test = pos_features[train_size:dev_test_size] + neg_features[train_size:dev_test_size]
    test = pos_features[dev_test_size:] + neg_features[dev_test_size:]

    test, tag_test = zip(*test)
    print('BernoulliNB`s accuracy is %f' % final_score(BernoulliNB()))
    print('MultinomiaNB`s accuracy is %f' % final_score(MultinomialNB()))
    print('LogisticRegression`s accuracy is %f' % final_score(LogisticRegression()))
    print('SVC`s accuracy is %f' % final_score(SVC()))
    print('LinearSVC`s accuracy is %f' % final_score(LinearSVC()))
    print('NuSVC`s accuracy is %f' % final_score(NuSVC()))


