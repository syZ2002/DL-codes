from torch.utils.data import DataLoader,Dataset,random_split
import torch
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from config import *
def read_data(path):
    texts = []
    with open(path,"r",encoding="utf-8") as f:
        for line in f.readlines():
            texts.append(line.strip('\n'))
    return texts
eng_texts = read_data("./Data/clean.en")
fra_texts = read_data("./Data/clean.fr")
en_tokenizer = get_tokenizer('moses',language='en')
fr_tokenizer = get_tokenizer('moses',language='fr')
def yield_tokens(texts,tokenizer):
    for text in texts:
        yield tokenizer(text)
def text2ids(text,tokenizer,vocab):
    return vocab(tokenizer(text))
class MyDataset(Dataset):
    def __init__(self, eng_texts, fra_texts):
        self.eng_texts = eng_texts
        self.fra_texts = fra_texts
        self.len = len(eng_texts)
    def __getitem__(self, index):
        return self.eng_texts[index], self.fra_texts[index]
    def __len__(self):
        return self.len
MTDataset = MyDataset(eng_texts, fra_texts)
train_set,valid_set,test_set = random_split(MTDataset,[0.8,0.1,0.1],generator=torch.Generator().manual_seed(42))
en_vocab = build_vocab_from_iterator(yield_tokens((eng_text for (eng_text,_) in train_set),en_tokenizer),max_tokens=config['vocab_size'],specials=['<sos>','<eos>','<unk>','<pad>'],special_first=True)
fr_vocab = build_vocab_from_iterator(yield_tokens((fra_text for (_,fra_text) in train_set),fr_tokenizer),max_tokens=config['vocab_size'],specials=['<sos>','<eos>','<unk>','<pad>'],special_first=True)
en_vocab.set_default_index(en_vocab['<unk>'])
fr_vocab.set_default_index(fr_vocab['<unk>'])
def collate_fn(batch):
    en_data = []
    fr_data = []
    for eng_text,fra_text in batch:
        en_ids = (text2ids(eng_text,en_tokenizer,en_vocab))
        fr_ids = [fr_vocab['<sos>']]
        fr_temp_ids = (text2ids(fra_text,fr_tokenizer,fr_vocab))
        if len(en_ids) > config['max_len']:
            en_data.append(en_ids[:config['max_len']])
        else:
            en_data.append(en_ids + [en_vocab['<pad>']]*(config['max_len']-len(en_ids)))
        if len(fr_temp_ids) > config['max_len'] - 2:
            fr_data.append(fr_ids +  (fr_temp_ids[:config['max_len']-2]) + [fr_vocab['<eos>']])
        else:
            fr_data.append(fr_ids + fr_temp_ids + [fr_vocab['<eos>']] + [fr_vocab['<pad>']]*(config['max_len']-len(fr_temp_ids)-2))
    return torch.tensor(en_data),torch.tensor(fr_data)
train_loader = DataLoader(train_set,batch_size=config['batch_size'],shuffle=True,collate_fn=collate_fn)
valid_loader = DataLoader(valid_set,batch_size=config['batch_size'],collate_fn=collate_fn)
test_loader = DataLoader(test_set,batch_size=config['batch_size'],collate_fn=collate_fn)