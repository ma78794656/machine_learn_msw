import pandas as pd
import re
import jieba

import sys
from os import path
cur_dir = path.dirname(__file__)
sys.path.append(cur_dir)
from text_classifiers import *


ignore_words = []


def parse_message(message):
    return ' '.join(re.findall(u'[\u4e00-\u9fff]+', message))


def load_ignore_words():
    global ignore_words
    if not ignore_words:
        with open('/data/code/github/machine_learn_msw/text_classify/fasttext/scikit-learn/stopwords.txt', encoding='utf-8') as f:
            words = f.readlines()
            ignore_words = [w.strip() for w in words]
    print(ignore_words)


def cut_message(message):
    message = message.replace(' ', '')
    word_list = [w for w in jieba.lcut(message) if w not in ignore_words]
    return word_list


def parse_data(file_name, contact_types=None, ignore_question_types=["无效会话"], quantize=False):
    id_info = {}
    dialog_data = pd.read_excel(file_name, sheetname=0, header=0, index_col=None)
    # label 数值化处理
    if quantize:
        siji_categories = pd.Categorical(dialog_data['siji'])
        dialog_data['siji'] = siji_categories.codes
        categories_list = siji_categories.categories
        ignore_categories = []
        for i in range(0, len(categories_list)):
            if categories_list[i] in ignore_question_types:
                ignore_categories.append(i)
        print(categories_list)
    else:
        ignore_categories = ignore_question_types
    print(ignore_categories)
    print("data size", dialog_data.shape[0])
    for i in range(0, dialog_data.shape[0]):
        dialog_id = dialog_data["staff_dialog_id"][i]
        message_content = dialog_data["message_content"][i]
        contact_type = dialog_data["contact_type"][i]
        siji = dialog_data["siji"][i]
        # 只处理想要的contact_type类型
        if contact_types and (contact_type not in contact_types):
            #print("contact_type not")
            continue
        # 过滤不想要的siji类型（问题类型）
        if ignore_categories and (siji in ignore_categories):
            #print("siji not")
            continue
        new_message = parse_message(message_content)
        if dialog_id not in id_info.keys():
            id_info[dialog_id] = [siji, contact_type, new_message]
        else:
            old_siji = id_info[dialog_id][0]
            old_message = id_info[dialog_id][2]
            if old_siji != siji:
                print("同一个dialog_id，但是问题类型不同")
            message = old_message + " " + new_message
            id_info[dialog_id][2] = message

    #for key, val in id_info.items():
    #    print(key, val)

    return id_info


def get_model_input(data_info):
    print('-'*20)
    label_data = []
    for key, val in data_info.items():
        label = val[0]
        message = val[2]
        features = cut_message(message)
        features = dict([(w, True) for w in features])
        label_data.append([features, label])
        #print([features, label])
    return label_data


if __name__ == '__main__':
    load_ignore_words()
    #file_name = '/Users/srt/Downloads/dialog_text.xlsx'
    file_name = cur_dir + '/dialog.xlsx'
    #parse_data(file_name, contact_types=None, ignore_question_types=["无效会话"])
    data_info = parse_data(file_name, contact_types=[1], ignore_question_types=["无效会话"])
    model_input = get_model_input(data_info)
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

