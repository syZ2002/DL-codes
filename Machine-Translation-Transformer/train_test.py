import torch
from tqdm import tqdm
from sacrebleu import corpus_bleu
from make_dataset import fr_vocab
def ids2tokens(ids,vocab):
    sentences = []
    vocab_tokens = vocab.get_itos()
    for seq in ids:
        sentence = []
        for token_id in seq:
            token_id = token_id.item()
            if token_id == vocab['<sos>']:
                continue
            if token_id == vocab['<eos>'] or token_id == vocab['<pad>']:
                break
            token = vocab_tokens[token_id]
            sentence.append(token)
        sentences.append(" ".join(sentence))
    return sentences


def trainer(model,criterion,optimizer,train_loader,valid_loader,config,fr_vocab):
    model.train()
    num_epochs = config['num_epochs']
    device = config['device']
    model.to(device)
    criterion.to(device)
    max_len = config['max_len']
    sos_id,eos_id = fr_vocab['<sos>'],fr_vocab['<eos>']
    for epoch in range(num_epochs):
        train_pbar = tqdm(train_loader)
        train_pbar.set_description(f'Epoch:[{epoch+1}/{num_epochs}]')
        for src,tgt in train_pbar:
            optimizer.zero_grad()
            src,tgt = src.to(device),tgt.to(device)
            #(batch_size,max_len)
            pred = model(src,tgt[:,:-1])
            #喂给模型tgt要去除最后一个token,一般是eos或pad
            loss = criterion(pred.view(-1,config['vocab_size']), tgt[:,1:].reshape(-1,))
            #用来对比的要去除第一个token即sos
            loss.backward()
            optimizer.step()
            train_pbar.set_postfix_str(f'Training Loss:{loss.item():.4f}')

        model.eval()
        with torch.no_grad():
            total_bleu = 0
            batch = 0
            valid_pbar = tqdm(valid_loader)
            valid_pbar.set_description(f'Epoch:[{epoch}/{num_epochs}]')
            for src,tgt in valid_pbar:
                batch += 1
                src,tgt = src.to(device),tgt.to(device)
                pred = torch.full((len(src),1),sos_id,dtype=torch.long,device=device)
                for _ in range(max_len-1):
                    memory = model.encoder(src)
                    logits = model.decoder(memory,pred)
                    output = model.FFN(logits)
                    #(batch_size,seq_len,vocab_size)
                    next_token_logits = output[:,-1,:]
                    #(batch_size,vocab_size)取对最后一个词的预测
                    next_token = next_token_logits.argmax(dim=-1,keepdim=True)
                    pred = torch.cat((pred,next_token),dim=1)
                    if (next_token == eos_id).all():
                        break
                pred_texts = ids2tokens(pred,fr_vocab)
                tgt_texts = ids2tokens(tgt,fr_vocab)
                score_bleu = corpus_bleu(pred_texts,[tgt_texts]).score
                total_bleu += score_bleu
                valid_pbar.set_description_str(f'Validation mean bleu:{total_bleu/batch:.2f}')
            # tqdm.write(f'Epoch:[{epoch}/{num_epochs}]:Valid  mean bleu:{total_bleu/batch}')

def tester(model,test_loader,config,fr_vocab):
    model.eval()
    sos_id = fr_vocab['<sos>']
    eos_id = fr_vocab['<eos>']
    total_bleu = 0
    device = config['device']
    model.to(device)
    batch = len(test_loader)
    max_len = config['max_len']
    with torch.no_grad():
        for src,tgt in test_loader:
            src,tgt = src.to(device),tgt.to(device)
            pred = torch.full((len(src),1),sos_id,dtype=torch.long,device=device)
            for _ in range(max_len-1):
                memory = model.encoder(src)
                logits = model.decoder(memory,pred)
                output = model.FFN(logits)
                next_token_logits = output[:,-1,:]
                next_token = next_token_logits.argmax(dim=-1,keepdim=True)
                pred = torch.cat((pred,next_token),dim=1)
                if (next_token == eos_id).all():
                    break
            pred_texts = ids2tokens(pred,fr_vocab)
            tgt_texts = ids2tokens(tgt,fr_vocab)
            total_bleu += corpus_bleu(pred_texts,[tgt_texts]).score
        print(f'Test mean bleu:{total_bleu/batch}')