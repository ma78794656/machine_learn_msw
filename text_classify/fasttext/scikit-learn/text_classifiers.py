# -*- coding: utf-8 -*-
import time
from sklearn import metrics
import pickle as pickle
import pandas as pd
from nltk.classify.scikitlearn import SklearnClassifier
import re


# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train):
    from sklearn.naive_bayes import MultinomialNB
    clf = SklearnClassifier(MultinomialNB(alpha=0.01))
    clf.train(train)
    return clf


# KNN Classifier
def knn_classifier(train):
    from sklearn.neighbors import KNeighborsClassifier
    clf = SklearnClassifier(KNeighborsClassifier())
    clf.train(train)
    return clf


# Logistic Regression Classifier
def logistic_regression_classifier(train):
    from sklearn.linear_model import LogisticRegression
    clf = SklearnClassifier(LogisticRegression(penalty='l2'))
    clf.train(train)
    return clf


# Random Forest Classifier
def random_forest_classifier(train):
    from sklearn.ensemble import RandomForestClassifier
    clf = SklearnClassifier(RandomForestClassifier(n_estimators=8))
    clf.train(train)
    return clf


# Decision Tree Classifier
def decision_tree_classifier(train):
    from sklearn import tree
    clf = SklearnClassifier(tree.DecisionTreeClassifier())
    clf.train(train)
    return clf


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train):
    from sklearn.ensemble import GradientBoostingClassifier
    clf = SklearnClassifier(GradientBoostingClassifier(n_estimators=200))
    clf.train(train)
    return clf


# SVM Classifier
def svm_classifier(train):
    from sklearn.svm import SVC
    clf = SklearnClassifier(SVC(kernel='rbf', probability=True))
    clf.train(train)
    return clf


# SVM Classifier using cross validation
def svm_cross_validation(train):
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    clf = SklearnClassifier(grid_search)
    clf.train(train)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    clf = SklearnClassifier(SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True))
    clf.train(train)
    clf.classify()
    clf.prob_classify()
    return clf


def gen_train_test_file(data, train_file, test_file, train_size, has_bag=False):
    train_data, test_data = split_train_test(data, train_size)
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


# fasttext classifier
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


def fasttext_test(fasttext_classifier, test_file):
    result = fasttext_classifier.test(test_file)
    print('******************* fasttext ********************')
    print('P@1:', result.precision)
    print('R@1:', result.recall)
    print('Number of examples:', result.nexamples)
    return result


def get_classifiers(name_list=None):
    #test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'SVMCV', 'GBDT']
    if not name_list:
        name_list = ['NB', 'KNN', 'LR', 'RF', 'DT', 'GBDT']
    all_classifiers = {'NB': naive_bayes_classifier,
                       'KNN': knn_classifier,
                       'LR': logistic_regression_classifier,
                       'RF': random_forest_classifier,
                       'DT': decision_tree_classifier,
                       'SVM': svm_classifier,
                       'SVMCV': svm_cross_validation,
                       'GBDT': gradient_boosting_classifier
                       }
    classifiers = {}
    for name in name_list:
        try:
            classifiers[name] = all_classifiers[name]
        except Exception as e:
            print("classifier: " + name + " not found. ")
            continue

    return classifiers


def train_classifiers(classifiers, train_data, test_data, save_classifiers=None):
    test_x, test_y = zip(*test_data)
    classifiers_save = {}
    for name, classifier in classifiers.items():
        print('******************* %s ********************' % name)
        start_time = time.time()
        model = classifier(train_data)
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


def classifiers_predict(classifiers, text_features, top_n=3):
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


def split_train_test(data, train_size):
    train_num = int(len(data) * 0.9)
    train = data[:train_num]
    test = data[train_num:]
    print("train_size:", len(train))
    print("text_size:", len(test))
    return train, test






