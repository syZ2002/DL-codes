import d2l.torch as d2l
import os
import string
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import *
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')
#@save
def read_data_nmt():
    """载入“英语－法语”数据集"""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
             encoding='utf-8') as f:
        return f.read()
#这里使用李沐老师《动手学深度学习》的英法机器翻译数据集和下载数据集的代码
def split_data_nmt(raw_data,min_len,max_len):
    eng_list = []
    fra_list = []
    translator = str.maketrans('', '', string.punctuation)
    for raw_line in raw_data.split('\n'):
        texts_list = raw_line.split('\t')
        if len(texts_list) == 2:
            en_text,fr_text = texts_list[0].translate(translator).lower(),texts_list[1].translate(translator).lower()
            if min_len <= len(en_text.split()) <= max_len:
                eng_list.append(en_text)
                fra_list.append(fr_text)
    return eng_list, fra_list
raw_text = read_data_nmt()
eng_texts,fra_texts = split_data_nmt(raw_text,config['min_len'],config['max_len'])
with open("clean.en", "w", encoding="utf-8") as f:
    f.write("\n".join(eng_texts) + "\n")
with open("clean.fr", "w", encoding="utf-8") as f:
    f.write("\n".join(fra_texts) + "\n")