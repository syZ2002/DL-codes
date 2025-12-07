import torch.nn as nn
from torch import optim
from PreprocessData import *
from models import Classifier
from train import trainer,tester
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classifier = Classifier(d_model=config['embed_size'],n_heads=config['n_heads'],vocab_size=config['vocab_size'],max_len=config['max_len'])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(),lr=config['learning_rate'])
trainer(classifier,train_loader,valid_loader,criterion,optimizer,config,device)
test_accuracy,test_mean_loss = tester(classifier,test_loader,criterion,device)
print(f'Test accuracy:{test_accuracy:4f}')