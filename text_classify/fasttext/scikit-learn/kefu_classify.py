import sys
from os import path
cur_dir = path.dirname(__file__)
sys.path.append(cur_dir)
from text_classifiers import *
import re
import pandas as pd


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

    return all_format_data

if __name__ == '__main__':
    data_file = cur_dir + '/train.csv'
    model_input = read_fasttext_data(data_file)
    train, test = split_train_test(model_input, 0.9)
    classifiers = get_classifiers(['NB', 'KNN', 'LR', 'RF', 'DT'])
    train_classifiers = train_classifiers(classifiers, train, test)

    # fasttext
    #train_file = "./fasttext_train.txt"
    #test_file = "./fasttext_text.txt"
    #model_file = "./fasttext_model"
    #gen_train_test_file(model_input, train_file, test_file, 0.9)
    #fasttext = fasttext_classifier(train_file, model_file)
    #fasttext_test(fasttext, test_file)

