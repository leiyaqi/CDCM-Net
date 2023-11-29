import json
import re
import clip
from tqdm import tqdm
import torch
from pytorch_transformers import BertTokenizer
import pandas as pd
import numpy as np

# image_path :/data/dataset/AVA/images
def load_name_and_text():
    max_len = 0
    with open("data_set/photonet_val.json") as t:  # ava multi model  path

        text_info = json.load(t)
        # print(text_info)
    D = {}
    L = {}
    for item in tqdm(text_info):
        Words = text_info[item]["comment"]
        S2 ="\\n"
        S1 = Words
        while S2 in S1:
            leftEnd = S1.index(S2)
            rightBegin = S1.index(S2) + len(S2)
            leftS1 = S1[0:leftEnd]
            rightS1 = S1[rightBegin:]
            S1 = leftS1 + rightS1
        Words = S1
        remove_chars = '[·’!"\#$%&\'()＃！1234567890\[\]（）*+,-./:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~]+'
        # Words = re.sub('\W*',".",Words)
        Word_list=[]


        for iitem in Words.split("\\"):
            Words_temp = re.split(r'[.?!]', iitem)
            for iiitem in range(len(Words_temp)):
                words = re.sub(remove_chars, '', Words_temp[iiitem])
                if len(words) > 3:
                    Word_list.append(words)
  
        # print(len(Word_list))
        if len(Word_list) < 1:
            print(Word_list)
            print("No enough length", text_info[item]["index"])
            # input()
            continue
        try:
            clip_token = clip.tokenize(Word_list,truncate=True)
        except:
            print(Word_list, "IS SO LONG")
            # input()
            continue
        D[text_info[item]["index"]] = clip_token
        label_list = []
        temp_lsit = text_info[item]["label"].split(" ")
        for each in temp_lsit:
            if len(each) > 0:
                label_list.append(int(each))
        L[text_info[item]["index"]] = label_list

    return D,L

# train_set = []
# test_set = []
# with open("data_set/train_all_final_EMD.json") as t:  # ava multi model  path
#     text_info = json.load(t)
#     for item in text_info:
#         train_set.append(item)
# with open("data_set/semi_test_final_EMD.json") as t:  # ava multi model  path
#     text_info = json.load(t)
#     for item in text_info:
#         test_set.append(item)

text_dict,label_dict = load_name_and_text()
torch.save(text_dict, "PhotoNet_val_text_dict.pth")
torch.save(label_dict, "PhotoNet_val_label_dict.pth")
#
# test_text_list = []
# test_name_index = []
# test_label_list = []
#
# text_dict = torch.load("AVA_text_dict.pth")
# label_dict = torch.load("AVA_label_dict.pth")
#
# D = {}
# L = {}
# for item in tqdm(test_set):
#     try:
#         D[item] = text_dict[item]
#         L[item] = label_dict[item]
#     except:
#         continue
# print(len(D))
# torch.save(D,"3AVA_test_text_dict.pth")
# torch.save(L,"3AVA_test_label_dict.pth")
# with open("/media/zxd/databases/AVA/AVA_dataset/ava_test_official.json") as t:
#     text_info = json.load(t)
#     print("", len(text_info))
#     for item in tqdm(text_info):
#         try:
#             D[item["filename"].split('.')[0]] = text_dict[item["filename"].split('.')[0]]
#         except:
#             continue
#         L[item["filename"].split('.')[0]] = item["distribution"]
#         # print(text_dict[item["filename"].split('.')[0]])
# print("test_len:",len(D))
# torch.save(D, "1AVA_test_text_dict.pth")
# torch.save(L, "1AVA_test_label_dict.pth")
#
# # D = torch.load("AVA_test_text_dict.pth")
# # L = torch.load("AVA_test_label_dict.pth")
# for item in D:
#     test_name_index.append(item)
#     test_text_list.append(D[item][0])
#     test_label_list.append(torch.Tensor(L[item]))
# D = {}
# L = {}
# for item in text_dict:
#     if item not in test_name_index:
#         D[item] = text_dict[item]
#         L[item] = label_dict[item]
#
# # torch.save(train_text_list,"AVA_train_text_dict.pth")
# # torch.save(train_label_list,"AVA_train_label_dict.pth")
# print("train_len:",len(D))
# torch.save(D,"1AVA_train_text_dict.pth")
# torch.save(L,"1AVA_train_label_dict.pth")

# import csv
# import json
#
#
# # Function to convert a CSV to JSON
# # Takes the file paths as arguments
# def make_json(csvFilePath, jsonFilePath):
#     # create a dictionary
#     data = {}
#
#     # Open a csv reader called DictReader
#     with open(csvFilePath, encoding='utf-8') as csvf:
#         csvReader = csv.DictReader(csvf)
#
#         # Convert each row into a dictionary
#         # and add it to data
#         for rows in csvReader:
#             # Assuming a column named 'No' to
#             # be the primary key
#             key = rows['index']
#             data[key] = rows
#     # Open a json writer, and use the json.dumps()
#     # function to dump data
#     with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
#         jsonf.write(json.dumps(data, indent=4))
#
#
# # Driver Code
# # Decide the two file paths according to your
# # computer system
# csvFilePath = r'data_set/photonet_val.csv'
# jsonFilePath = r'data_set/photonet_val.json'
# # Call the make_json function
# make_json(csvFilePath, jsonFilePath)
