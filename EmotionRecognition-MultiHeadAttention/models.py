import torch.nn as nn
import torch.nn.functional as F
from PreprocessData import *
class Embeddings(nn.Module):
    #用来做位置编码和嵌入向量
    def __init__(self, vocab_size, embed_size,max_len):
        super(Embeddings, self).__init__()
        #embed_size:一个词可以有几个特征维度
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        pe = torch.zeros(max_len, embed_size)#(max_len,embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)#(max_len,1)
        d = torch.pow(10000,-torch.arange(0,embed_size,2).float() / embed_size).unsqueeze(0)#(1,embed_size//2)
        angles = position*d#(max_len,embed_size//2)
        pe[:,0::2] = torch.sin(angles)
        pe[:,1::2] = torch.cos(angles)
        self.register_buffer('position_embedding',pe)
    def forward(self, input_ids):
        return self.word_embedding(input_ids)+self.position_embedding

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model,n_heads,mask=False):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.mask = mask
        assert self.head_dim*self.n_heads == d_model
        self.W_query = nn.Linear(d_model,d_model)
        self.W_key = nn.Linear(d_model,d_model)
        self.W_value = nn.Linear(d_model,d_model)
        self.W_out = nn.Linear(d_model,d_model)
    def forward(self,query,key=None,value=None):
        #query:(batch_size,max_len,d_model)
        if key is None:
            key = query
        if value is None:
            value = query
        max_len = query.size(1)
        query = self.W_query(query).view(query.size(0),query.size(1),self.n_heads,self.head_dim).transpose(1, 2)
        key = self.W_key(key).view(key.size(0),key.size(1),self.n_heads,self.head_dim).transpose(1, 2)
        value = self.W_value(value).view(value.size(0),value.size(1),self.n_heads,self.head_dim).transpose(1, 2)
        '''
        本来是(batch_size,max_len,d_model),
        通过线性层变成(batch_size,max_len,d_model),
        view之后变成(batch_size,max_len,n_heads,head_dim),
        transpose之后变成(batch_size,n_heads,max_len,head_dim)。
        '''
        scores = torch.matmul(query,key.transpose(-1,-2))/self.head_dim**0.5
        #(batch_size, n_head, max_len, max_len)
        if self.mask:
            causal_mask = torch.tril(torch.ones(max_len,max_len,dtype=torch.bool,device=query.device))
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            '''
            一个下三角矩阵，(max_len,max_len),
            下三角全部为True，上三角全部为False
            然后用unsqueeze扩展维度，
            形状:(1,1,max_len,max_len)
            用来做交叉注意力
            '''
            scores = scores.masked_fill(~causal_mask,float('-inf'))
            '''
            mask_fill:把True位置设为对应值
            (max_len,max_len):
                行：每个词的索引
                列：每个词对其他词的分数
            这句代码把scores中的(max_len,max_len)部分的第i行前i个元素保留，
            后面设置负无穷大，方便softmax计算
            '''
        scores = F.softmax(scores,dim=-1)
        #(batch_size,n_head,max_len,max_len)
        weight = torch.matmul(scores,value).transpose(1,2).contiguous()
        '''
        score与value做矩阵乘法形状:(batch_size,n_heads,max_len,head_dim)
        transpose后形状为(batch_size,max_len,n_heads,head_dim)
        '''
        weight = weight.view(weight.size(0),weight.size(1),-1)
        #(batch_size,max_len,d_model)
        return self.W_out(weight)#(batch_size,max_len,d_model)

class Classifier(nn.Module):
    def __init__(self, d_model,n_heads,vocab_size,max_len,mask=False):
        super(Classifier, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.multihead_attention = MultiHeadAttention(d_model,n_heads,mask)
        self.embedding = Embeddings(vocab_size,d_model,max_len)
        self.classifier = nn.Linear(d_model,2)
        self.FFN = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_model,4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model,d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
    def forward(self,input_ids):
        input_ids = self.embedding(input_ids)
        input_ids = self.layer_norm1(self.multihead_attention(input_ids) +input_ids)
        input_ids = self.layer_norm2(self.FFN(input_ids) +input_ids)
        input_ids = self.classifier(input_ids.mean(dim=1))
        #用一整句的平均分数来判断
        return input_ids
