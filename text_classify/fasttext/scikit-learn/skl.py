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
    return clf


def read_data(data_file):
    data = pd.read_csv(data_file)
    train = data[:int(len(data) * 0.9)]
    test = data[int(len(data) * 0.9):]
    train_y = train.label
    train_x = train.drop('label', axis=1)
    test_y = test.label
    test_x = test.drop('label', axis=1)
    return train_x, train_y, test_x, test_y


def read_fasttext_data(data_file):
    data = pd.read_csv(data_file, sep='\t', header=None, names=['label', 'data'], index_col=None)
    data_nona = data.dropna(axis=0, how='any')
    values = data_nona.values
    all_format_data = []
    pat = re.compile('\s+')
    for v in values:
        vv = re.sub(pat, ' ', v[1]).split(' ')
        vl = dict([(w, True) for w in vv])
        all_format_data.append([vl, v[0]])

    train_num = int(len(all_format_data) * 0.9)
    train = all_format_data[:train_num]
    test = all_format_data[train_num:]

    return train, test


if __name__ == '__main__':
    data_file = '/data/code/github/machine_learn_msw/text_classify/fasttext/python_interface/train.csv'
    thresh = 0.5
    model_save_file = None
    model_save = {}

    #test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'SVMCV', 'GBDT']
    test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT']
    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifier,
                   'SVM': svm_classifier,
                   'SVMCV': svm_cross_validation,
                   'GBDT': gradient_boosting_classifier
                   }

    print('reading training and testing data...')
    #train_x, train_y, test_x, test_y = read_data(data_file)
    train, test = read_fasttext_data(data_file)
    test_x, test_y = zip(*test)

    for classifier in test_classifiers:
        print('******************* %s ********************' % classifier)
        start_time = time.time()
        model = classifiers[classifier](train)
        print('training took %fs!' % (time.time() - start_time))
        predict = model.classify_many(test_x)
        if model_save_file != None:
            model_save[classifier] = model
        #precision = metrics.precision_score(test_y, predict)
        #recall = metrics.recall_score(test_y, predict)
        #print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
        accuracy = metrics.accuracy_score(test_y, predict)
        print('accuracy: %.2f%%' % (100 * accuracy))

    if model_save_file != None:
        pickle.dump(model_save, open(model_save_file, 'wb'))