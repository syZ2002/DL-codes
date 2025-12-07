import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader,Dataset,random_split
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
tokenizer = get_tokenizer("basic_english")
from config import *
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)
def text2ids(text):
    return vocab(tokenizer(text))
train_iter,test_iter = IMDB(root='./data',split=('train','test'))
vocab = build_vocab_from_iterator(yield_tokens(train_iter),max_tokens=config['vocab_size'],specials=['<unk>','<pad>'],special_first=True)
vocab.set_default_index(vocab['<unk>'])

class IMDBdataset(Dataset):
    def __init__(self,data_iter,max_len=config['max_len']):
        super(IMDBdataset,self).__init__()
        self.data_iter = data_iter
        self.label_list = []
        self.text_ids_list = []
        for label,text in data_iter:
            self.label_list.append(label-1)
            ids = text2ids(text)
            if len(ids) > max_len:
                ids = ids[:max_len]
            else:
                ids = ids + [vocab['<pad>']]*(max_len-len(ids))
            self.text_ids_list.append(ids)
        self.label_list = torch.tensor(self.label_list)
        self.text_ids_list = torch.tensor(self.text_ids_list)
    def __getitem__(self,idx):
        return self.label_list[idx],self.text_ids_list[idx]
    def __len__(self):
        return len(self.label_list)
train_set = IMDBdataset(train_iter)
test_set = IMDBdataset(test_iter)
dataset = torch.utils.data.ConcatDataset([train_set,test_set])
train_set,valid_set,test_set = random_split(dataset,[0.8,0.1,0.1],generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(dataset=train_set,shuffle=True,batch_size=config['batch_size'])
valid_loader = DataLoader(dataset=valid_set,shuffle=False,batch_size=config['batch_size'])
test_loader = DataLoader(dataset=test_set,shuffle=False,batch_size=config['batch_size'])
# for label,text_ids in train_loader:
#     print(label.sum())