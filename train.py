import math
import torch
import datasets

from torch import nn
from torch import optim

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from tqdm import tqdm

from LSTM import LSTM

def _load_dataset():
    return datasets.load_dataset('wikitext', 'wikitext-2-raw-v1')

def _tokenize_dataset(dataset):
    tokenizer = get_tokenizer('basic_english')
    tokenize_data = lambda example, tokenizer: {'tokens': tokenizer(example['text'])}  
    tokenized_dataset = dataset.map(tokenize_data, remove_columns=['text'], fn_kwargs={'tokenizer': tokenizer})
    return tokenized_dataset

def _construct_vocabulary(tokenized_dataset):
    vocab = build_vocab_from_iterator(tokenized_dataset['train']['tokens'], min_freq=3) 
    vocab.insert_token('<unk>', 0)
    vocab.insert_token('<eos>', 1)
    vocab.set_default_index(vocab['<unk>'])
    return vocab

def _get_data(dataset, vocab, batch_size):
    data = []                                                   
    for example in dataset:
        if example['tokens']:                                      
            tokens = example['tokens'].append('<eos>')             
            tokens = [vocab[token] for token in example['tokens']] 
            data.extend(tokens)                                    
    data = torch.LongTensor(data)                                 
    num_batches = data.shape[0] // batch_size 
    data = data[:num_batches * batch_size]                       
    data = data.view(batch_size, num_batches)          
    return data

def _get_batch(data, seq_len, num_batches, idx):
    src = data[:, idx:idx+seq_len]                   
    target = data[:, idx+1:idx+seq_len+1]             
    return src, target

def _init_model(
    vocab_size=0,
    embedding_dim=1024,
    hidden_dim=1024,
    num_layers=2,
    dropout_rate=0.65,
    tie_weights=True,
    lr=1e-3,
):
    model = LSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        tie_weights=tie_weights
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {num_params:,} trainable parameters')

    return model, optimizer, criterion


def train(model, data, optimizer, criterion, batch_size, seq_len, clip, device):

    epoch_loss = 0
    model.train()
    # drop all batches that are not a multiple of seq_len
    num_batches = data.shape[-1]
    data = data[:, :num_batches - (num_batches -1) % seq_len]
    num_batches = data.shape[-1]

    hidden = model.init_hidden(batch_size, device)
    
    for idx in tqdm(range(0, num_batches - 1, seq_len), desc='Training: ',leave=False):  # The last batch can't be a src
        optimizer.zero_grad()
        hidden = model.detach_hidden(hidden)

        src, target = _get_batch(data, seq_len, num_batches, idx)
        src, target = src.to(device), target.to(device)
        batch_size = src.shape[0]
        prediction, hidden = model(src, hidden)               

        prediction = prediction.reshape(batch_size * seq_len, -1)   
        target = target.reshape(-1)
        loss = criterion(prediction, target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item() * seq_len
    return epoch_loss / num_batches

def evaluate(model, data, criterion, batch_size, seq_len, device):
    epoch_loss = 0
    model.eval()
    num_batches = data.shape[-1]
    data = data[:, :num_batches - (num_batches -1) % seq_len]
    num_batches = data.shape[-1]

    hidden = model.init_hidden(batch_size, device)

    with torch.no_grad():
        for idx in range(0, num_batches - 1, seq_len):
            hidden = model.detach_hidden(hidden)
            src, target = _get_batch(data, seq_len, num_batches, idx)
            src, target = src.to(device), target.to(device)
            batch_size= src.shape[0]

            prediction, hidden = model(src, hidden)
            prediction = prediction.reshape(batch_size * seq_len, -1)
            target = target.reshape(-1)

            loss = criterion(prediction, target)
            epoch_loss += loss.item() * seq_len
    return epoch_loss / num_batches

def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)  
            prediction = torch.multinomial(probs, num_samples=1).item()    
            
            while prediction == vocab['<unk>']:
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:
                break

            indices.append(prediction)

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens

def main(
    model,
    optimizer,
    criterion,
    batch_size,
    train_data,
    test_data,
    valid_data,
    device,
    n_epochs=50,
    seq_len=50,
    clip=0.25,
    saved=False
):
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0)

    if saved:
        model.load_state_dict(torch.load('best-val-lstm_lm.pt',  map_location=device))
        test_loss = evaluate(model, test_data, criterion, batch_size, seq_len, device)
        print(f'Test Perplexity: {math.exp(test_loss):.3f}')
    else:
        best_valid_loss = float('inf')

        for epoch in range(n_epochs):
            train_loss = train(model, train_data, optimizer, criterion, 
                        batch_size, seq_len, clip, device)
            valid_loss = evaluate(model, valid_data, criterion, batch_size, 
                        seq_len, device)
            
            lr_scheduler.step(valid_loss)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'best-val-lstm_lm.pt')

            print(f'\tTrain Perplexity: {math.exp(train_loss):.3f}')
            print(f'\tValid Perplexity: {math.exp(valid_loss):.3f}')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)

    dataset = _load_dataset()
    tokenized_dataset = _tokenize_dataset(dataset=dataset)
    vocabulary = _construct_vocabulary(tokenized_dataset=tokenized_dataset)

    model, optimizer, criterion = _init_model(
        vocab_size=len(vocabulary)
    )

    batch_size = 128
    train_data = _get_data(tokenized_dataset['train'], vocabulary, batch_size)
    valid_data = _get_data(tokenized_dataset['validation'], vocabulary, batch_size)
    test_data = _get_data(tokenized_dataset['test'], vocabulary, batch_size)

    main(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        batch_size=batch_size,
        train_data=train_data,
        test_data=test_data,
        valid_data=valid_data,
        device=device
    )
