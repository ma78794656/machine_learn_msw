import pandas as pd
import re
import jieba

import sys
from os import path
cur_dir = path.dirname(__file__)
sys.path.append(cur_dir)
from text_classifiers import *


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
    data_file = r'/data/code/github/machine_learn_msw/text_classify/fasttext/python_interface/train.csv'
    model_input = read_fasttext_data(data_file)
    train, test = split_train_test(model_input, 0.9)
    classifiers = get_classifiers(['NB', 'KNN', 'LR', 'RF', 'DT'])
    train_classifiers = train_classifiers(classifiers, train, test)

    train_file = "./fasttext_train.txt"
    test_file = "./fasttext_text.txt"
    model_file = "./fasttext_model"
    gen_train_test_file(model_input, train_file, test_file, 0.9)
    fasttext = fasttext_classifier(train_file, model_file)
    fasttext_test(fasttext, test_file)




    desc = {'5': True,
            '。': True,
            '不': True,
            '了': True,
            '元': True,
            '充值卡': True,
            '去取': True,
            '商家': True,
            '安抚': True,
            '家': True,
            '没有': True,
            '然后': True,
            '用户': True,
            '米饭': True,
            '给配': True,
            '自己': True,
            '致歉': True,
            '表示': True,
            '认可': True,
            '让': True,
            '送到': True,
            '送餐': True,
            '需要': True,
            '骑手': True,
            '，': True}

    desc1 = {'不': True,
             '之前': True,
             '了': True,
             '但是': True,
             '可以': True,
             '恢复': True,
             '支付宝': True,
             '显示': True,
             '现在': True,
             '用': True,
             '都': True,
             '，': True}

    desc2 = {' ': True,
             '-': True,
             '09': True,
             '11': True,
             '19': True,
             '19.00': True,
             '2017': True,
             '26': True,
             '58': True,
             ':': True,
             '一笔': True,
             '两次': True,
             '了': True,
             '元': True,
             '到': True,
             '原': True,
             '反馈': True,
             '告知': True,
             '在': True,
             '外卖': True,
             '已经': True,
             '您': True,
             '支付': True,
             '支付方': True,
             '查看': True,
             '款项': True,
             '用户': True,
             '的': True,
             '退回': True}

    desc3 = {'一笔': True,
             '两次': True,
             '元': True,
             '到': True,
             '原': True,
             '反馈': True,
             '告知': True,
             '在': True,
             '外卖': True,
             '已经': True,
             '您': True,
             '支付': True,
             '支付方': True,
             '查看': True,
             '款项': True,
             '用户': True,
             '的': True,
             '退回': True}

    #p1, p2, fr = classifiers_predict(train_classifiers, desc3)
    #print(p1)
    #print(p2)
    #print(fr)
