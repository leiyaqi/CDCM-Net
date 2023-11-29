import json
import re
import clip
from tqdm import tqdm
import torch
from pytorch_transformers import BertTokenizer


# image_path :/data/dataset/AVA/images
def load_name_and_text():
    max_len = 0
    with open("/media/zxd/codes/xy/mutil_mod/ava_multi_model.json") as t:  # ava multi model  path

        text_info = json.load(t)
        # print(text_info)
    D = {}
    L = {}
    for item in tqdm(text_info):
        Words = item["sentences"]
        remove_chars = '[·’!"\#$%&\'()＃！\\1234567890（）*+,-./:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~]+'
        # Words = re.sub('\W*',".",Words)
        Word_list=[]
        for iitem in range(len(Words)):
            Words_temp = re.split(r'[.?!]',Words[iitem])

            for iiitem in range(len(Words_temp)):
                words = re.sub(remove_chars,'',Words_temp[iiitem])

                if len(words) > 0:
                    Word_list.append(words)
        if len(Word_list) > max_len:
            max_len = len(Word_list)
           
        if len(Word_list)  < 5:
            # print(Words)
            print("No enough length",item["filename"])
            continue
        try:
            clip_token= clip.tokenize(Word_list)
        except:
            print(Words,"IS SO LONG")
            continue
        D[item["filename"].split('.')[0]] = clip_token
        L[item["filename"].split('.')[0]] = item["distribution"]
    print("max_len:",max_len)
    return D,L


text_dict,label_dict = load_name_and_text()
torch.save(text_dict, "1AVA_text_dict.pth")
torch.save(label_dict, "1AVA_label_dict.pth")

test_text_list = []
test_name_index = []
test_label_list = []
#
# text_dict = torch.load("AVA_text_dict.pth")
# label_dict = torch.load("AVA_label_dict.pth")
D = {}
L = {}
with open("/media/zxd/databases/AVA/AVA_dataset/ava_test_official.json") as t:
    text_info = json.load(t)
    print("test sample：", len(text_info))
    for item in tqdm(text_info):
        try:
            D[item["filename"].split('.')[0]] = text_dict[item["filename"].split('.')[0]]
        except:
            continue
        L[item["filename"].split('.')[0]] = item["distribution"]
        # print(text_dict[item["filename"].split('.')[0]])
print("test_len:",len(D))
torch.save(D, "1AVA_test_text_dict.pth")
torch.save(L, "1AVA_test_label_dict.pth")

# D = torch.load("AVA_test_text_dict.pth")
# L = torch.load("AVA_test_label_dict.pth")
for item in D:
    test_name_index.append(item)
    test_text_list.append(D[item][0])
    test_label_list.append(torch.Tensor(L[item]))
D = {}
L = {}
for item in text_dict:
    if item not in test_name_index:
        D[item] = text_dict[item]
        L[item] = label_dict[item]

# torch.save(train_text_list,"AVA_train_text_dict.pth")
# torch.save(train_label_list,"AVA_train_label_dict.pth")
print("train_len:",len(D))
torch.save(D,"1AVA_train_text_dict.pth")
torch.save(L,"1AVA_train_label_dict.pth")