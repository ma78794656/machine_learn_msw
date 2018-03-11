import pandas as pd
import re
import jieba

# TextProcessAdvance
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
import itertools
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# classifier
import time
import pickle as pickle
from sklearn import metrics
from nltk.classify.scikitlearn import SklearnClassifier

# set pandas display options
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 20


class TextProcess():
    def __init__(self, ignore_file):
        self.ignore_file = ignore_file
        self.ignore_words =[]
        self.load_ignore_words()

    def load_ignore_words(self):
        if not self.ignore_words:
            with open('/data/code/github/machine_learn_msw/text_classify/fasttext/scikit-learn/shangde/ignore_words.txt', encoding='utf-8') as f:
                words = f.readlines()
                self.ignore_words = [w.strip() for w in words]
        #print(ignore_words)

    def parse(self, x):
        x = re.sub(u' [\u4e00-\u9fff]{1,4}说', '', x).strip()
        x = ' '.join(re.findall(u'[\u4e00-\u9fff]+', x))
        x = re.sub('\s+', ' ', x)
        return x

    def cut_message(self, message):
        message = message.replace(' ', '')
        word_list = [w for w in jieba.lcut(message) if w not in self.ignore_words]
        return word_list

    def filter_cut_text(self, text):
        ptext = self.parse(text)
        ctext = self.cut_message(ptext)
        return ctext


class TextProcessAdvance():
    def __init__(self):
        pass
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
    def get_bag_of_words(self, data_label_list):
        data_label_list_bagwords = []
        for data_label in data_label_list:
            words = data_label[0]
            label = data_label[1]
            bag_words = self.bag_of_words(words)
            data_label_list_bagwords.append([bag_words, label])
        # [[{'致电': True, '回访': True}, '报考问题'], [{'xx': True, 'xx': True}, 'xxx'], ...]
        return data_label_list_bagwords


    def bag_of_words(self, words):
        return dict([(word, True) for word in words])

    # 方法二：双词搭配，卡法统计，取一定比例的词，作为特征
    # 词 -> 双词 -> 特征
    # score_fn：BigramAssocMeasures.chi_sq - 卡方统计法；BigramAssocMeasures.pmi - 互信息方法
    def bigram(self, words, ratio=0.3, use_fn=True, score_fn=BigramAssocMeasures.chi_sq):
        bigram_finder = BigramCollocationFinder.from_words(words, window_size=2)
        num = int(len(words) * ratio)
        bigrams = bigram_finder.nbest(score_fn, num)
        return self.bag_of_words(bigrams)

    # 方法三：所有词+双词，卡方统计，取一定比例的词，作为特征
    # 词 + 双词 -> 特征
    def get_bigram_words(self, data_label_list, ratio=0.8):
        data_label_list_bigram = []
        for data_label in data_label_list:
            words = data_label[0]
            label = data_label[1]
            bag_words = self.bigram_words(words, ratio)

            data_label_list_bigram.append([bag_words, label])
        # [[{'致电': True, '回访': True}, '报考问题'], [{'xx': True, 'xx': True}, 'xxx'], ...]
        return data_label_list_bigram

    def bigram_words(self, words, ratio=0.8, score_fn=BigramAssocMeasures.chi_sq):
        bigram_finder = BigramCollocationFinder.from_words(words, window_size=2)
        num = int(len(words) * ratio)
        bigrams = bigram_finder.nbest(score_fn, num)
        bigrams_list = []
        for line in bigrams:
            tmp = ''
            for s in line:
                for w in s:
                    tmp += w
            bigrams_list.append(tmp)
        return self.bag_of_words(words + bigrams_list)

    # 方法四：所有词作为特征，并计算其在所有语料中的信息量
    # words: 所有label及其words list， 如：
    # {'pos':[['aa','bb', 'cc'], ['dd','ee','ff']], 'neg':[['AA','BB', 'CC'], ['DD','EE','FF']]}
    def get_word_scores(self, label_words, ratio=1):
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
        #print(word_scores[:10])
        return word_scores

    def get_word_scores1(self, label_words, ratio=1):
        label_words_freq = {}
        label_words_count = {}
        all_words_plain = []
        # ConditionalFreqDist:         条件-事件
        fdist = ConditionalFreqDist((label, word) for label in label_words.keys() for word in list(itertools.chain(*(label_words[label]))))
        for key, val in label_words.items():
            # -> ['aa','bb', 'cc','dd','ee','ff', 'AA','BB', 'CC', 'DD','EE','FF']
            plain_words = list(itertools.chain(*val))
            all_words_plain.extend(plain_words)
        #    # -> {'pos':{'aa':1,'bb':1, 'cc':1,'dd':1,'ee':1,'ff':1}, ...}
        #    freq_dist = FreqDist(plain_words)
        #    label_words_freq[key] = freq_dist
        #    count = freq_dist.N()
        #    # -> {'pos': 3, 'neg': 3}
        #    label_words_count[key] = count

        all_words_freq = FreqDist(all_words_plain)
        total_count = all_words_freq.N()
        word_scores = {}
        for word, freq in all_words_freq.items():
            score = 0
            for key in label_words.keys():
                # bigram_score_fn(n_ii, (n_ix, n_xi), n_xx)
                #        w1    ~w1
                #     ------ ------
                # w2 | n_ii | n_oi | = n_xi
                #     ------ ------
                #~w2 | n_io | n_oo |
                #     ------ ------
                #     = n_ix        TOTAL = n_xx
                # w1: 单词； w2：标签
                # n_ii: 该单词在该标签下发生的概率
                # n_ix: 该单词发生的概率
                # n_xi: 该标签下所有单词的数量
                # n_xx: 所有单词的数量
                score += BigramAssocMeasures.chi_sq(
                    fdist[key][word],
                    (freq, fdist[key].N()),
                    total_count)
            word_scores[word] = score
        #print(word_scores[:10])
        return word_scores

    # 方法五：所有词+双词作为特征，并计算其在所有语料中的信息量
    # words: 所有label及其words list， 如：
    # {'pos':[['aa','bb', 'cc'], ['dd','ee','ff']], 'neg':[['AA','BB', 'CC'], ['DD','EE','FF']]}
    def get_word_bigrams_scores(self, label_words, ratio=0.3):
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

    def find_best_words(self, word_scores, ratio=0.3):
        # 把词按信息量倒序排序。number是特征的维度，是可以不断调整直至最优的
        number = int(len(word_scores) * ratio)
        best_vals = sorted(word_scores.items(), key=lambda w_s: w_s[1], reverse=True)[:number]
        best_words = set([w for w, s in best_vals])
        return best_words

    def best_word_features(self, words, best_words):
        return dict([(word, True) for word in words if word in best_words])

    # words: {'pos':[['aa','bb', 'cc'], ['dd','ee','ff']], 'neg':[['AA','BB', 'CC'], ['DD','EE','FF']]}
    def filter_with_nbest_words(self, words, nbest):
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
    def label_features(self, feature_method, words_list, label, extract_ratio=0.3):
        features_list = []
        for words in words_list:
            features = [feature_method(words, extract_ratio), label]
            features_list.append(features)
        return features_list

    # 'data' 已经可以直接feed给model进行训练，这里对'data'重新删选feature，然后再恢复成与'data'相同的格式
    # 由于时间和编写的原因，可以整合成为一个逻辑，不需要进行两遍操作
    # data example:
    # [[{'aa':True, 'bb':True, 'cc':True}, 'pos'], [{'AA':True, 'BB':True, 'CC':True}, 'neg']]

    def tfidf_process(self):
        corpus = ["我 来到 北京 清华大学",          # 第一类文本切词后的结果，词之间以空格隔开
                  "他 来到 了 网易 杭研 大厦",      # 第二类文本的切词结果
                  "小明 硕士 毕业 与 中国 科学院",   # 第三类文本的切词结果
                  "我 爱 北京 天安门"]             # 第四类文本的切词结果
        vectorizer = CountVectorizer()          # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
        transformer = TfidfTransformer()        # 该类会统计每个词语的tf-idf权值
        tfidf = transformer.fit_transform(
            vectorizer.fit_transform(corpus))   # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
        word = vectorizer.get_feature_names()   # 获取词袋模型中的所有词语
        weight = tfidf.toarray()                # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
        for i in range(len(weight)):            # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
            print("-------这里输出第", i, u"类文本的词语tf-idf权重------")
            for j in range(len(word)):
                print(word[j], weight[i][j])


class FormatData:
    def __init__(self, data_file):
        self.file_path = data_file

    def get_dataframe(self):
        return self.dataframe

    def get_text_data(self):
        self.dataframe = pd.read_excel(self.file_path, sheetname='主题明细', header=0, names=['dialog', 'cid',     'cname', 'clist'], usecols=[1,12,13,14])
        print(self.dataframe.head())

    def filterNan(self, column_name):
        self.dataframe = self.dataframe.loc[self.dataframe[column_name].isnull() == False]

    def get_filter_cut_text(self, filter_cut_func):
        self.dataframe['dialog'] = self.dataframe['dialog'].apply(filter_cut_func)
        print(self.dataframe.head())

    def get_nltk_data_label(self, label_column, data_column):
        label_list = self.dataframe[label_column].unique()
        data_label_list = []
        for index, row in self.dataframe[[data_column, label_column]].iterrows():
            data_label_list.append(row.tolist())

        label_data_map = {}
        for label in label_list:
            data = self.dataframe.loc[self.dataframe[label_column] == label][data_column].tolist()
            tmp = label_data_map.get(label, [])
            tmp.extend(data)
            label_data_map[label] = tmp
        # data_label_list : [[['a', 'b'], 'label'], ...] ; 每行数据（数据已分词）及其标签的列表
        # label_data_map: {'label':['a', 'b']} ; 所有标签及其数据的合并列表构成的map（数据已分词）
        return data_label_list, label_data_map


class TrainClassifier:
    def __init__(self, classifiers, train_data, test_data, **kwargs):
        self.classifiers = self.get_classifiers(classifiers)
        self.train_data = train_data
        self.test_data = test_data
        self.params = kwargs

    def get_classifiers(self, name_list=None):
        #test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'SVMCV', 'GBDT']
        if not name_list:
            name_list = ['NB', 'KNN', 'LR', 'RF', 'DT', 'GBDT']
        all_classifiers = {'NB': self.naive_bayes_classifier,
                           'KNN': self.knn_classifier,
                           'LR': self.logistic_regression_classifier,
                           'RF': self.random_forest_classifier,
                           'DT': self.decision_tree_classifier,
                           'SVM': self.svm_classifier,
                           'SVMCV': self.svm_cross_validation,
                           'GBDT': self.gradient_boosting_classifier
                           }
        classifiers = {}
        for name in name_list:
            try:
                classifiers[name] = all_classifiers[name]
            except Exception as e:
                print(e)
                print("classifier: " + name + " not found. ")
                continue

        return classifiers

    # Multinomial Naive Bayes Classifier
    def naive_bayes_classifier(self):
        from sklearn.naive_bayes import MultinomialNB
        clf = SklearnClassifier(MultinomialNB(alpha=0.01))
        clf.train(self.train_data)
        return clf

    # KNN Classifier
    def knn_classifier(self):
        from sklearn.neighbors import KNeighborsClassifier
        clf = SklearnClassifier(KNeighborsClassifier())
        clf.train(self.train_data)
        return clf

    # Logistic Regression Classifier
    def logistic_regression_classifier(self):
        from sklearn.linear_model import LogisticRegression
        clf = SklearnClassifier(LogisticRegression(penalty='l2'))
        clf.train(self.train_data)
        return clf

    # Random Forest Classifier
    def random_forest_classifier(self):
        from sklearn.ensemble import RandomForestClassifier
        clf = SklearnClassifier(RandomForestClassifier(n_estimators=8))
        clf.train(self.train_data)
        return clf

    # Decision Tree Classifier
    def decision_tree_classifier(self):
        from sklearn import tree
        clf = SklearnClassifier(tree.DecisionTreeClassifier())
        clf.train(self.train_data)
        return clf

    # GBDT(Gradient Boosting Decision Tree) Classifier
    def gradient_boosting_classifier(self):
        from sklearn.ensemble import GradientBoostingClassifier
        clf = SklearnClassifier(GradientBoostingClassifier(n_estimators=200))
        clf.train(self.train_data)
        return clf

    # SVM Classifier
    def svm_classifier(self):
        from sklearn.svm import SVC
        clf = SklearnClassifier(SVC(kernel='rbf', probability=True))
        clf.train(self.train_data)
        return clf

    # SVM Classifier using cross validation
    def svm_cross_validation(self):
        from sklearn.model_selection import GridSearchCV
        from sklearn.svm import SVC
        model = SVC(kernel='rbf', probability=True)
        param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
        grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
        clf = SklearnClassifier(grid_search)
        clf.train(self.train_data)
        best_parameters = grid_search.best_estimator_.get_params()
        for para, val in list(best_parameters.items()):
            print(para, val)
        clf = SklearnClassifier(SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True))
        clf.train(self.train_data)
        return clf

    # fasttext classifier
    @staticmethod
    def fasttext_classifier(train_file, model_file):
        import fasttext as ft
        #model_file = cur_dir + time.strftime('%Y%m%d', time.localtime()) + "_model"
        dim = 100
        lr = 0.1
        epoch = 5
        min_count = 1
        word_ngrams = 3
        bucket = 2000000
        thread = 4
        silent = 1
        label_prefix = '__label__'
        # Train the classifier
        classifier = ft.supervised(train_file, model_file, dim=dim, lr=lr, epoch=epoch, min_count=min_count, word_ngrams=word_ngrams, bucket=bucket,thread=thread, silent=silent, label_prefix=label_prefix)
        return classifier

    @staticmethod
    def fasttext_test(fasttext_classifier, test_file):
        result = fasttext_classifier.test(test_file)
        print('******************* fasttext ********************')
        print('P@1:', result.precision)
        print('R@1:', result.recall)
        print('Number of examples:', result.nexamples)
        return result

    def train_classifiers(self, save_classifiers=None):
        test_x, test_y = zip(*self.test_data)
        classifiers_save = {}
        for name, classifier in self.classifiers.items():
            print('******************* %s ********************' % name)
            start_time = time.time()
            model = classifier()
            print('training took %fs!' % (time.time() - start_time))
            predict = model.classify_many(test_x)
            #precision = metrics.precision_score(test_y, predict)
            #recall = metrics.recall_score(test_y, predict)
            #print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
            accuracy = metrics.accuracy_score(test_y, predict)
            print('accuracy: %.2f%%' % (100 * accuracy))
            classifiers_save[classifier] = model

        if save_classifiers:
            pickle.dump(classifiers_save, open(save_classifiers, 'wb'))

        return classifiers_save

    def classifiers_predict(self, classifiers, text_features, top_n=3):
        p1_result = {}
        p2_result = {}
        top_result = {}
        for name, model in classifiers.items():
            p1 = model.classify(text_features)  # 直接结果
            p2 = model.prob_classify(text_features)  # 包含所有预测类型和相应概率
            p1_result[name] = p1
            p2_result[name] = p2
            for key, val in p2._prob_dict.items():
                top_result[key] = top_result.get(key, 0) + val
                #print(key, ":", val)

        classifiers_num = len(classifiers)
        for key in top_result.keys():
            top_result[key] = top_result[key] / classifiers_num

        top_result = dict(sorted(top_result.items(), key=lambda d: d[1], reverse=True)[0: top_n])
        final_result = [p1_result, p2_result, top_result]
        return final_result


class TrainTestData:
    def __init__(self):
        pass

    @staticmethod
    def split_train_test(data, train_size):
        train_num = int(len(data) * train_size)
        train = data[:train_num]
        test = data[train_num:]
        print("train_size:", len(train))
        print("text_size:", len(test))
        return train, test

    def gen_train_test_file(self, data, train_file, test_file, train_size, has_bag=False):
        train_data, test_data = self.split_train_test(data, train_size)
        if len([''.join(k) for k in data[0][0].keys() if isinstance(k, tuple)]) > 0:
            has_bag = True
        with open(train_file, 'w', encoding='utf-8') as f:
            for info in train_data:
                if has_bag:
                    kl = []
                    for k in info[0].keys():
                        k = ''.join(k)
                        kl.append(k)
                    text = ' '.join(kl)
                else:
                    text = ' '.join(info[0].keys())
                label = info[1]
                label = "__label__" + str(label)
                f.write(label + "\t" + text + "\n")

        with open(test_file, 'w', encoding='utf-8') as f:
            for info in test_data:
                if has_bag:
                    kl = []
                    for k in info[0].keys():
                        k = ''.join(k)
                        kl.append(k)
                    text = ' '.join(kl)
                else:
                    text = ' '.join(info[0].keys())
                label = info[1]
                label = "__label__" + str(label)
                f.write(label + "\t" + text + "\n")


def main():
    data_file_path = "/data/code/github/machine_learn_msw/text_classify/fasttext/scikit-learn/shangde/【2018-01-22】IM聊天会话主题挖掘和词频统计结果.xlsx"
    ignore_file_path = "/data/code/github/machine_learn_msw/text_classify/fasttext/scikit-learn/shangde/ignore_words.txt"
    formatData = FormatData(data_file_path)
    processer = TextProcess(ignore_file_path)
    advancer = TextProcessAdvance()
    train_test_data = TrainTestData()
    #advance_processer = TextProcessAdvance()
    #advance_processer.tfidf_process()

    # get pandas data
    formatData.get_text_data()
    # get text parse data, action include filter unnecessary word, jieba fenci
    formatData.get_filter_cut_text(processer.filter_cut_text)
    # filter nan
    formatData.filterNan('cname')
    # 生成 data - label
    data_label_list, label_data_map = formatData.get_nltk_data_label('cname', 'dialog')
    # ----- train -----
    # --------words
    # 生成nltk模型需要的输入数据的格式
    bag_words = advancer.get_bag_of_words(data_label_list)
    # split train and test data
    train, test = train_test_data.split_train_test(bag_words, 0.8)
    # train and messured model
    models = TrainClassifier(['NB', 'KNN', 'LR', 'RF', 'DT'], train, test)
    models.train_classifiers()
    # --------bigram
    # 生成nltk模型需要的输入数据的格式
    bigram = advancer.get_bigram_words(data_label_list)
    train, test = train_test_data.split_train_test(bigram, 0.8)
    # train and messured model
    models = TrainClassifier(['NB', 'KNN', 'LR', 'RF', 'DT'], train, test)
    models.train_classifiers()




if __name__ == '__main__':
    main()
