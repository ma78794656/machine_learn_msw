# -*- coding:utf-8 -*-
import os
import chardet
# 结巴分词

def get_utf8_data_file(fpath):
    if os.path.isfile(fpath):
        fd = open(fpath, 'rb')
        data = fd.read()
        encodeing = chardet.detect(data)['encoding']
        print(fpath, encodeing)
        gbk_content = ""
        utf8_content = ""
        success = True
        error = False
        status = error
        if encodeing and encodeing.upper().startswith("GB"):
            try:
                gbk_content = data.decode(encodeing)
                utf8_content = gbk_content.encode("utf-8")
                print(fpath, encodeing, success)
                status = success
            except Exception as e:
                print(fpath, encodeing, error)
                try:
                    gbk_content = data.decode("GBK")
                    utf8_content = gbk_content.encode("utf-8")
                    print(fpath, "GBK", success)
                    status = success
                except Exception as e:
                    print(fpath, encodeing, "GBK", error)
                    try:
                        gbk_content = data.decode("GBK", 'ignore')
                        utf8_content = gbk_content.encode("utf-8")
                        print(fpath, encodeing, "GBK ignore", success)
                        status = success
                    except Exception as e:
                        print(fpath, encodeing, "GBK ignore", error)
        elif encodeing and encodeing.upper().startswith("UTF"):
            try:
                gbk_content = data.decode(encodeing)
                utf8_content = gbk_content.encode(encodeing)
                print(fpath, encodeing, success)
                status = success
            except Exception as e:
                print(fpath, encodeing, error)
        else:
            try:
                gbk_content = data.decode("GB2312")
                utf8_content = gbk_content.encode("utf-8")
                print(fpath, encodeing, "GB2312", success)
                status = success
            except Exception as e:
                print(fpath, encodeing, "GB2312", error)
                try:
                    gbk_content = data.decode("GBK")
                    utf8_content = gbk_content.encode("utf-8")
                    print(fpath, encodeing, "GBK", success)
                    status = success
                except Exception as e:
                    print(fpath, encodeing, "GBK", error)
                    try:
                        gbk_content = data.decode("GBK", 'ignore')
                        utf8_content = gbk_content.encode("utf-8")
                        print(fpath, encodeing, "GBK ignore", success)
                        status = success
                    except Exception as e:
                        print(fpath, encodeing, "GBK ignore", error)
        fd.close()
        print("---------------------------------------------------")
        result = [status, utf8_content, gbk_content]
        return result
    else:
        raise Exception(fpath, "is not a file")

def get_utf8_data_file_lines(fpath):
    if os.path.isfile(fpath):
        fd = open(fpath, 'rb')
        data = fd.readlines()
        encodeing = chardet.detect(data[0])['encoding']
        print(fpath, encodeing)
        gbk_contents = []
        utf8_contents = []
        success = True
        error = False
        if encodeing and encodeing.upper().startswith("GB"):
            try:
                for d in data:
                    gbk_content = d.decode("GBK", 'ignore')
                    utf8_content = gbk_content.encode("utf-8")
                    gbk_contents.append(gbk_content)
                    utf8_contents.append(utf8_content)
                print(fpath, encodeing, "GBK ignore", success)
            except Exception as e:
                print(fpath, encodeing, "GBK ignore", error)
        elif encodeing and encodeing.upper().startswith("UTF"):
            try:
                for d in data:
                    gbk_content = d.decode(encodeing)
                    utf8_content = gbk_content.encode("utf-8")
                    gbk_contents.append(gbk_content)
                    utf8_contents.append(utf8_content)
                print(fpath, encodeing, success)
            except Exception as e:
                print(fpath, encodeing, error)
        else:
            try:
                for d in data:
                    gbk_content = d.decode("GBK", 'ignore')
                    utf8_content = gbk_content.encode("utf-8")
                    gbk_contents.append(gbk_content)
                    utf8_contents.append(utf8_content)
                print(fpath, encodeing, "GBK ignore", success)
            except Exception as e:
                print(fpath, encodeing, "GBK ignore", error)
        fd.close()
        print("---------------------------------------------------")
        result = [utf8_contents, gbk_contents]
        return result
    else:
        raise Exception(fpath, "is not a file")

def get_utf8_data_dir(path):
    fileList = os.listdir(path)
    utf8_data_list = []
    gbk_data_list = []
    for f in fileList:
        fpath = path + "/" + f
        [status, utf8_data, gbk_data] = get_utf8_data_file(fpath)
        if status:
            utf8_data_list.append(utf8_data)
            gbk_data_list.append(gbk_data)

    return [utf8_data_list, gbk_data_list]


if __name__ == '__main__':
    path = r'D:\codes\python\myTest\nlp\machine_learn\meituan\tousu_fankui_test'
    [status, utf8_data, gbk_data] = get_utf8_data_file_lines(path)
    path = r'D:\codes\python\myTest\file\dataset_602123\ChnSentiCorp_htl_ba_2000\neg'
    fileList = os.listdir(path)
    i = 0
    for f in fileList:
        i += 1
        fpath = path + "\\" + f
        [status, utf8_data, gbk_data] = get_utf8_data_file(fpath)
        if status:
            print(utf8_data)
            print(gbk_data)
        if i > 100:
            break

