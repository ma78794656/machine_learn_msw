# -*- coding: utf-8 -*-
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
import itertools
# nltk words processing 可以采取的方法和步骤
# 1 特征提取
#     a 所有词
#     b 双词搭配
#     c 所有词和双词
# 2 特征选择
#     a 计算整个语料中每个词的信息量
#     b 按照信息量倒排，选择排名靠前信息量大的词
#     c 选择这些词作为训练特征


# 说明：
# 一、1,2,3不适合处理短文本（< 10个词）(增大ratio值，会使卡方统计的效果减弱，等于1时相当于没有使用)
# 二、4，5对短文本的处理效果可能会与语料质量关系比较大，语料质量不好会使短文本中的词被过滤所剩无几甚至为空的概率很大
#
# 方法一：所有词作为特征
# 词 -> 特征
def bag_of_words(words):
    return dict([(word, True) for word in words])


# 方法二：双词搭配，卡法统计，取一定比例的词，作为特征
# 词 -> 双词 -> 特征
# score_fn：BigramAssocMeasures.chi_sq - 卡方统计法；BigramAssocMeasures.pmi - 互信息方法
def bigram(words, ratio=0.3, use_fn=True, score_fn=BigramAssocMeasures.chi_sq):
    bigram_finder = BigramCollocationFinder.from_words(words, window_size=2)
    num = int(len(words) * ratio)
    bigrams = bigram_finder.nbest(score_fn, num)
    return bag_of_words(bigrams)


# 方法三：所有词+双词，卡方统计，取一定比例的词，作为特征
# 词 + 双词 -> 特征
def bigram_words(words, ratio=0.3, use_fn=True, score_fn=BigramAssocMeasures.chi_sq):
    bigram_finder = BigramCollocationFinder.from_words(words, window_size=2)
    num = int(len(words) * ratio)
    bigrams = bigram_finder.nbest(score_fn, 20)
    return bag_of_words(words + bigrams)


# 方法四：所有词作为特征，并计算其在所有语料中的信息量
# words: 所有label及其words list， 如：
# {'pos':[['aa','bb', 'cc'], ['dd','ee','ff']], 'neg':[['AA','BB', 'CC'], ['DD','EE','FF']]}
def get_word_scores(label_words, ratio=1):
    label_words_freq = {}
    label_words_count = {}
    all_words_plain = []
    for key, val in label_words.items():
        # -> ['aa','bb', 'cc','dd','ee','ff', 'AA','BB', 'CC', 'DD','EE','FF']
        plain_words = list(itertools.chain(*val))
        all_words_plain.extend(plain_words)
        # -> {'pos':{'aa':1,'bb':1, 'cc':1,'dd':1,'ee':1,'ff':1}, ...}
        freq_dist = FreqDist(plain_words)
        label_words_freq[key] = freq_dist
        count = freq_dist.N()
        # -> {'pos': 3, 'neg': 3}
        label_words_count[key] = count
    all_words_freq = FreqDist(all_words_plain)
    total_count = all_words_freq.N()
    word_scores = {}
    for word, freq in all_words_freq.items():
        score = 0
        for key in label_words.keys():
            score += BigramAssocMeasures.chi_sq(
                label_words_freq[key][word],
                (freq, label_words_count[key]),
                total_count)
        word_scores[word] = score
    return word_scores


# 方法五：所有词+双词作为特征，并计算其在所有语料中的信息量
# words: 所有label及其words list， 如：
# {'pos':[['aa','bb', 'cc'], ['dd','ee','ff']], 'neg':[['AA','BB', 'CC'], ['DD','EE','FF']]}
def get_word_bigrams_scores(label_words, ratio=0.3):
    label_words_freq = {}
    label_words_count = {}
    all_words_plain = []
    for key, val in label_words.items():
        # -> ['aa','bb', 'cc','dd','ee','ff', 'AA','BB', 'CC', 'DD','EE','FF']
        plain_words = list(itertools.chain(*val))
        bigram_finder = BigramCollocationFinder.from_words(plain_words)
        bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, int(len(plain_words) * ratio))
        words_bigrams = plain_words + bigrams
        all_words_plain.extend(words_bigrams)
        # -> {'pos':{'aa':1,'bb':1, 'cc':1,'dd':1,'ee':1,'ff':1}, ...}
        freq_dist = FreqDist(words_bigrams)
        label_words_freq[key] = freq_dist
        count = freq_dist.N()
        # -> {'pos': 3, 'neg': 3}
        label_words_count[key] = count
    all_words_freq = FreqDist(all_words_plain)
    total_count = all_words_freq.N()
    word_scores = {}
    for word, freq in all_words_freq.items():
        score = 0
        for key in label_words.keys():
            score += BigramAssocMeasures.chi_sq(
                label_words_freq[key][word],
                (freq, label_words_count[key]),
                total_count)
        word_scores[word] = score
    return word_scores


def find_best_words(word_scores, ratio=0.3):
    # 把词按信息量倒序排序。number是特征的维度，是可以不断调整直至最优的
    number = int(len(word_scores) * ratio)
    best_vals = sorted(word_scores.items(), key=lambda w_s: w_s[1], reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words


def best_word_features(words, best_words):
    return dict([(word, True) for word in words if word in best_words])


# words: {'pos':[['aa','bb', 'cc'], ['dd','ee','ff']], 'neg':[['AA','BB', 'CC'], ['DD','EE','FF']]}
def filter_with_nbest_words(words, nbest):
    features_list = []
    for key, val in words.items():
        for ws in val:
            best_features = dict([(word, True) for word in ws if word in nbest])
            if len(best_features) == 0:
                print('------ len(best_features) = 0')
                continue
            features = [best_features, key]
            features_list.append(features)
    return features_list


# words_list example: [['aa', 'bb', 'cc'], ['dd', 'ee', 'ff'], ['AA', 'BB', 'CC'], ['DD', 'EE', 'FF']]
def label_features(feature_method, words_list, label, extract_ratio=0.3):
    features_list = []
    for words in words_list:
        features = [feature_method(words, extract_ratio), label]
        features_list.append(features)
    return features_list


# 'data' 已经可以直接feed给model进行训练，这里对'data'重新删选feature，然后再恢复成与'data'相同的格式
# 由于时间和编写的原因，可以整合成为一个逻辑，不需要进行两遍操作
# data example:
# [[{'aa':True, 'bb':True, 'cc':True}, 'pos'], [{'AA':True, 'BB':True, 'CC':True}, 'neg']]


