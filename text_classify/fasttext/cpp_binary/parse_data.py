# -*- coding: utf-8 -*-
from os import path
import pandas as pd
import jieba
#import pickle
import re

cur_dir = path.dirname(__file__)
train_file = path.join(cur_dir, 'train.csv')
test_file = path.join(cur_dir, 'test.csv')
ori_file = path.join(cur_dir, '09.18-09.24-waimai-kefu-gongdan.xlsx')

def parse_data1(ori_name, train_name, test_name, train_ratio):
    #if path.isfile(train_name) and path.isfile(test_name):
    #    return
    # 读取excel，转换成dataframe
    data=pd.read_excel(ori_name, sheetname='在线', header=0, index_col=None, parse_cols=[0, 12, 13, 19])
    # 添加label
    data.insert(1, 'label', ['__label__' + str(i) for i in data['三级问题id'].values])
    data_label=data['label'].values
    data_desc=data['用户反馈'].values
    # 数据清洗
    mobile_pat=re.compile(r'1\d{10}')
    data_desc1 = [mobile_pat.sub('', str(desc)) for desc in data_desc]
    # 分词
    data_desc2 = [' '.join(jieba.lcut(desc.replace("\t", " ").replace("\n", " "))) for desc in data_desc1]
    # 保存到csv文件
    df = pd.DataFrame({'label': data_label, 'data': data_desc2})

    #df_tmp = df.where(df.notnull(), '')
    print(df.shape)
    df.fillna('', inplace=True)
    drop_row = [i for i in range(df.shape[0]) if len(df.iloc[i][0]) < 3]
    df.drop(drop_row, inplace=True)

    print(df.shape)
    train_size = int(df.shape[0] * train_ratio)
    train_data = df.iloc[:train_size, :]
    test_data = df.iloc[train_size:, :]
    print(train_data.shape)
    print(test_data.shape)

    train_data.to_csv(train_name, header=False, index=False, columns=['label','data'], sep='\t', encoding='UTF8')
    test_data.to_csv(test_name, header=False, index=False, columns=['label','data'], sep='\t', encoding='UTF8')


if __name__ == '__main__':
    parse_data1(ori_file, train_file, test_file, 0.9)
