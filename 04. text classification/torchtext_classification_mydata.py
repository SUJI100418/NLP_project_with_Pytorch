
import pandas as pd
import numpy as np

import time
import torch
from torchtext import data
import torch.nn as nn

from sklearn.model_selection import train_test_split

# 1. data load
my_dir = "/Users/suji/nlp_dataset"
train = pd.read_csv(my_dir + "/train.csv")  # (7615,5)
test = pd.read_csv(my_dir + "/test.csv")  # (3263,4)

#print(train.shape)
#print(test.shape)

#print(train.head()) # columns = id, keyword, location, text, target

train.drop(columns=['id', 'keyword', 'location'], inplace=True)
'''
inplace = true : drop한 후의 데이터 프레임으로 기존 df를 대체하겠다.
'''
#print(train.head())

'''
target 설명
label = 1 : Tweet이 Disasters
label = 0 : Tweet이 not Disasters
'''

# 2. cleanning data
# r'정규표현식': 문자열을 그대로 raw string으로 처리하라.
def normalise_text(text):
    text = text.str.lower() # 소문자로 바꾸기
    text = text.str.replace(r"\#","") # 해시태그 자우기
    text = text.str.replace(r"http\S+","URL") # URL, http 지우기
    text = text.str.replace(r"@","") # @ 지우기
    text = text.str.replace(r"[^A-Za-z0-9()!?\`\'\"]"," ") # [] 지우기
    text = text.str.replace("\s{2,}","") # {2,} 지우기
    return text

train["text"]=normalise_text(train["text"])
#print(train['text'].head())

train_df, valid_df = train_test_split(train)
print("전체 갯수 :",train.shape[0],", train 갯수 :",train_df.shape[0], " , valid 갯수 :",valid_df.shape[0])

# torchtext_field 만들기

TEXT = data.Field(tokenize = 'spacy', include_lengths=True)
LABEL = data.Field(dtype= torch.float)

class DataFrameDataset(data.Dataset):
    def __init__(self, df, fields, is_test=False, **kwargs):
        examples = []
        
