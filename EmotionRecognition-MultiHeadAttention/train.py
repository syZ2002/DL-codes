import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
def trainer(model,train_loader,valid_loader,criterion,optimizer,config,device):
    num_epochs = config['num_epochs']
    model.to(device)
    criterion.to(device)
    model.train()
    for epoch in range(num_epochs):
        train_pbar = tqdm(train_loader)
        train_pbar.set_description_str(f'Epoch:[{epoch+1}/{num_epochs}]')
        right_train = 0
        total_train = 0
        for label,text_ids in train_pbar:
            label,text_ids = label.to(device),text_ids.to(device)
            optimizer.zero_grad()
            pred = model(text_ids)
            right_train += (pred.argmax(dim=1)==label).sum().item()
            total_train += label.size(0)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            train_pbar.set_postfix_str(f'Training Loss:{loss.item():.4f},Accuracy:{right_train / total_train:.4f}')
        with torch.no_grad():
            valid_pbar = tqdm(valid_loader)
            model.eval()
            right_valid = 0
            total_valid = 0
            valid_pbar.set_description_str(f'Epoch:[{epoch+1}/{num_epochs}]')
            for label,text_ids in valid_pbar:
                label,text_ids = label.to(device),text_ids.to(device)
                pred = model(text_ids)
                right_valid += (pred.argmax(dim=1)==label).sum().item()
                total_valid += label.size(0)
                loss = criterion(pred, label)
                valid_pbar.set_postfix_str(f'Validation Loss:{loss.item():.4f},Accuracy:{right_valid / total_valid:.4f}')
def tester(model,test_loader,criterion,device):
    model.to(device)
    criterion.to(device)
    model.eval()
    right_test = 0
    total_test = 0
    test_loss = 0
    with torch.no_grad():
        for label,text_ids in test_loader:
            label,text_ids = label.to(device),text_ids.to(device)
            pred = model(text_ids)
            loss = criterion(pred, label)
            test_loss += loss.item()
            right_test += (pred.argmax(dim=1)==label).sum().item()
            total_test += label.size(0)
        return right_test/total_test,test_loss/total_test