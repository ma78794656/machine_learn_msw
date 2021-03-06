# -*- coding: utf-8 -*-
import pandas as pd
import re
import jieba

import sys
from os import path
cur_dir = path.dirname(__file__)
sys.path.append(cur_dir)
from text_classifiers import *
from text_processors import *


ignore_words = []


def parse_message(message):
    return ' '.join(re.findall(u'[\u4e00-\u9fff]+', message))


def load_ignore_words():
    global ignore_words
    if not ignore_words:
        with open('/data/code/github/machine_learn_msw/text_classify/fasttext/scikit-learn/stopwords.txt', encoding='utf-8') as f:
            words = f.readlines()
            ignore_words = [w.strip() for w in words]
    #print(ignore_words)


def cut_message(message):
    message = message.replace(' ', '')
    word_list = [w for w in jieba.lcut(message) if w not in ignore_words]
    return word_list


def parse_data(file_name, contact_types=None, ignore_question_types=[u"无效会话"], quantize=True):
    id_info = {}
    dialog_data = pd.read_excel(file_name, sheetname=0, header=0, index_col=None)
    # label 数值化处理
    if quantize:
        siji_categories = pd.Categorical(dialog_data['siji'])
        dialog_data['siji'] = siji_categories.codes
        categories_list = siji_categories.categories
        ignore_categories = []
        if ignore_question_types:
            for i in range(0, len(categories_list)):
                if categories_list[i] in ignore_question_types:
                    ignore_categories.append(i)
        #print(categories_list)
    else:
        ignore_categories = ignore_question_types
    #print(ignore_categories)
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
                #print("同一个dialog_id，但是问题类型不同")
                continue
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
        if len(features) < 2:
            continue
        features = dict([(w, True) for w in features])
        label_data.append([features, label])
    return label_data


def get_model_input_nltk(data_info, words_bag):
    print('-'*20)
    label_data_map = {}
    #example: {'pos':[['aa','bb', 'cc'], ['dd','ee','ff']], 'neg':[['AA','BB', 'CC'], ['DD','EE','FF']]}
    for val in data_info.values():
        label = val[0]
        features = cut_message(val[2])
        if len(features) < 3:
            continue
        if label in label_data_map.keys():
            label_data_map[label].append(features)
        else:
            label_data_map[label] = [features]

    label_data = []
    for key, val in label_data_map.items():
        try:
            features = label_features(words_bag, val, key, extract_ratio=1)
        except Exception as e:
            continue
        label_data.extend(features)

    print(label_data[0])
    return label_data


# data_info
def get_nbest_words(data_info, ratio, nbest_fn):
    print('-'*20)
    label_data_map = {}
    # -> {'pos':[['aa','bb', 'cc'], ['dd','ee','ff']], 'neg':[['AA','BB', 'CC'], ['DD','EE','FF']]}
    for val in data_info.values():
        label = val[0]
        features = cut_message(val[2])
        if label in label_data_map.keys():
            label_data_map[label].append(features)
        else:
            label_data_map[label] = [features]

    word_scores = nbest_fn(label_data_map, ratio=1)
    best_words = find_best_words(word_scores, ratio)
    return filter_with_nbest_words(label_data_map, best_words)


if __name__ == '__main__':
    #load_ignore_words()
    #file_name = '/Users/srt/Downloads/dialog_text.xlsx'
    file_name = cur_dir + '/dialog.xlsx'
    #parse_data(file_name, contact_types=None, ignore_question_types=["无效会话"])
    contact_type_list = [None, 1, 2]
    ignore_type_list = [None, u"无效会话"]
    load_ignore = [False, True]
    #process_type = ['no bag', 'bag(two)', 'bag(one+two)', 'no bag+nbest', 'bag(one+two)+nbest']
    process_type = ['bag(one+two)']
    for load in load_ignore:
        if load:
            load_ignore_words()
            load_name = "加载停用词"
        else:
            load_name = "不加在停用词"
        for contact in contact_type_list:
            if not contact:
                contact_types = None
                contact_name = "客服+客户"
            elif contact == 1:
                contact_types = [1]
                contact_name = "用户"
            else:
                contact_types = [2]
                contact_name = "客服"
            for ignore in ignore_type_list:
                if not ignore:
                    ignore_types = None
                    ignore_name = "不过滤问题类型"
                else:
                    ignore_types = [u"无效会话"]
                    ignore_name = "过滤'无效会话'类型"
                data_info = parse_data(file_name, contact_types=contact_types, ignore_question_types=ignore_types)
                for process in process_type:
                    try:
                        if process == 'no bag':
                            process_name = "全部单词"
                            model_input = get_model_input(data_info)
                        elif process == 'bag(two)':
                            process_name = "全部双词"
                            model_input = get_model_input_nltk(data_info, words_bag=bigram)
                        elif process == 'bag(one+two)':
                            process_name = "全部单词+双词"
                            model_input = get_model_input_nltk(data_info, words_bag=bigram_words)
                        elif process == 'no bag+nbest':
                            process_name = "全部单词+取70% best"
                            model_input = get_nbest_words(data_info, 0.7, get_word_scores)
                        elif process == 'bag(one+two)+nbest':
                            process_name = "全部单词+双词+取70% best"
                            model_input = get_nbest_words(data_info, 0.7, get_word_bigrams_scores)
                        else:
                            continue
                    except Exception as e:
                        print('Error1 happen')
                        continue
                    print("#"*20+load_name+"----"+contact_name+"----"+ignore_name+"----"+process_name+"#"*20)
                    try:
                        train, test = split_train_test(model_input, 0.9)
                        classifiers = get_classifiers(['NB', 'KNN', 'LR', 'RF', 'DT'])
                        trained_classifiers = train_classifiers(classifiers, train, test)

                        # fasttext
                        train_file = "./fasttext_train.txt"
                        test_file = "./fasttext_text.txt"
                        model_file = "./fasttext_model"
                        gen_train_test_file(model_input, train_file, test_file, 0.9)
                        fasttext = fasttext_classifier(train_file, model_file)
                        fasttext_test(fasttext, test_file)
                    except Exception as e:
                        print("Error2 happen")

    ##data_info = parse_data(file_name, contact_types=[2], ignore_question_types=[u"无效会话"])
    #data_info = parse_data(file_name, contact_types=None, ignore_question_types=None)
    #model_input = get_model_input(data_info)
    ##model_input = get_model_input_nltk(data_info)
    ##model_input = get_nbest_words(data_info, 0.3)
    #train, test = split_train_test(model_input, 0.9)
    #classifiers = get_classifiers(['NB', 'KNN', 'LR', 'RF', 'DT'])
    #train_classifiers = train_classifiers(classifiers, train, test)

    ## fasttext
    #train_file = "./fasttext_train.txt"
    #test_file = "./fasttext_text.txt"
    #model_file = "./fasttext_model"
    #gen_train_test_file(model_input, train_file, test_file, 0.9)
    #fasttext = fasttext_classifier(train_file, model_file)
    #fasttext_test(fasttext, test_file)

