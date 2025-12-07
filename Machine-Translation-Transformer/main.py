from models import Encoder,Decoder
import torch
import torch.nn as nn
from config import *
from make_dataset import train_loader,valid_loader,test_loader,fr_vocab
from train_test import trainer,tester
class Transformer(nn.Module):
    def __init__(self,d_model,n_heads,vocab_size,max_len,):
        super(Transformer,self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.encoder = Encoder(d_model,n_heads,vocab_size,max_len)
        self.decoder = Decoder(d_model,n_heads,vocab_size,max_len)
        self.FFN = nn.Sequential(
            nn.Linear(d_model,d_model),
            nn.GELU(),
            nn.Linear(d_model,vocab_size)
        )
    def forward(self,src,tgt):
        src = self.encoder(src)
        tgt = self.decoder(src,tgt)
        tgt = self.FFN(tgt)
        return tgt

if __name__=='__main__':
    model = Transformer(config['embed_size'],config['n_heads'],config['vocab_size'],config['max_len'])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    trainer(model,criterion,optimizer,train_loader,valid_loader,config,fr_vocab)
    tester(model,test_loader,config,fr_vocab)
