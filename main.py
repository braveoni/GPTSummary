from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import get_linear_schedule_with_warmup

import torch
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW

import os
import random
import datetime
import time

import pandas as pd
import numpy as np

from tqdm import tqdm

def add_special_tokens(name):
    tokenizer = GPT2Tokenizer.from_pretrained(name)
    special_tokens = {'pad_token': '|<pad>|', 'sep_token': '|<sep>|'}
    num_add_tokens = tokenizer.add_special_tokens(special_tokens)
    return tokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GPT22048Dataset(Dataset):
    def __init__(self, path, model_name, mode='train'):
        self.tokenizer: GPT2Tokenizer = add_special_tokens(model_name)
        self.mode = mode
        self.df = pd.read_json(path, lines=True).drop(columns=['date', 'url', 'title'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.tokenizer.encode(self.tokenizer.pad_token) * 512
        content = self.df['text'][idx] + self.tokenizer.encode(self.tokenizer.sep_token) + self.df['summary'][idx]
        text[:len(content)] = content
        text = torch.tensor(text)
        sample = {'text': text, 'sum_idx': len(self.df['text'])}

        return sample


def evaluate(model, eval_dataset, ignore_index, device, global_step=None):
    """ Returns perplexity score on validation dataset.
        Args:
            args: dict that contains all the necessary information passed by user while training
            model: finetuned gpt/gpt2 model
            eval_dataset: GPT21024Dataset object for validation data
            global_step: no. of times gradients have backpropagated
            ignore_index: token not considered in loss calculation
    """
    if not os.path.exists('output_dir'):
        os.mkdir('output_dir')
    eval_output_dir = 'output_dir'

    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=32)
    loss_fct = CrossEntropyLoss(ignore_index=ignore_index) #ignores padding token for loss calculation

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = batch['article'].to(device), batch['article'].to(device)
        
        with torch.no_grad():
            logits = model(inputs)[0]
            idx = batch['sum_idx'].item() # index of separator token
            # only consider loss on reference summary just like seq2seq models
            shift_logits = logits[..., batch['sum_idx']:-1, :].contiguous()
            shift_labels = labels[..., batch['sum_idx']+1:].contiguous()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": perplexity
    }
    print("perplexity:", perplexity.item())

    if global_step:
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as f:
            for key in sorted(result.keys()):
                f.write('\n\n')
                f.write("time = %s, %s = %s, step = %s\n" % (datetime.now().strftime("%d/%m/%Y %H:%M:%S"), key, str(result[key]), str(global_step)))
    return result


def train(model, device, train_dataset, valid_dataset, ignore_index):
    """ Trains GPT2 model and logs necessary details.
        Args:
            args: dict that contains all the necessary information passed by user while training
            model: finetuned gpt/gpt2 model
            tokenizer: GPT/GPT2 tokenizer
            train_dataset: GPT21024Dataset object for training data
            ignore_index: token not considered in loss calculation
    """
    train_sampler = RandomSampler(train_dataset)
    train_dl = DataLoader(train_dataset, sampler=train_sampler,
                          batch_size=32, num_workers=4)
    # ignores padding token for loss calculation
    loss_fct = CrossEntropyLoss(ignore_index=ignore_index)
    optimizer = AdamW(model.parameters(), lr=0.003)
    scheduler = get_linear_schedule_with_warmup(optimizer, 100, 80000)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    set_seed(15451)
    for _ in tqdm(range(50), desc="Epoch"):
        epoch_iterator = tqdm(train_dl, desc="Training")
        for step, batch in enumerate(epoch_iterator):
            inputs, labels = batch['article'].to(device), batch['article'].to(device)
            model.train()
            logits = model(inputs)[0]
            # only consider loss on reference summary just like seq2seq models
            shift_logits = logits[..., batch['sum_idx']:-1, :].contiguous()
            shift_labels = labels[..., batch['sum_idx']+1:].contiguous()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss/32
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1)
            tr_loss += loss.item()
            if (step + 1) % 32 == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                logging_loss = tr_loss
                print("loss:", loss.item(), end='\n\n')
                if (step + 1)/32 == 1.0:
                    print('After 1st update: ', end='\n\n')
            
            if (step + 1) % (10*32) == 0:
                results = evaluate(model, valid_dataset,
                                   ignore_index, device, global_step)
                print('After', global_step+1, 'updates: ', end='\n\n')



model_name_or_path = "sberbank-ai/rugpt2large"

train_data = GPT22048Dataset('gazeta_train.jsonl', model_name_or_path, mode='train')
valid_data = GPT22048Dataset('gazeta_test.jsonl', model_name_or_path, mode='valid')

tokenizer = add_special_tokens(model_name_or_path)
ignore_idx = tokenizer.pad_token_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT2LMHeadModel.from_pretrained(model_name_or_path).cuda()
model.resize_token_embeddings(len(tokenizer))
model.to(device)

start = time.time()
train(model, device, train_data, valid_data, ignore_idx)
print('total time: ', (time.time()-start)/60, " minutes", end='\n\n')