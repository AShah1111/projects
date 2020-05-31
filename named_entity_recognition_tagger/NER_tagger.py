from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
from transformers import BertTokenizer, BertForTokenClassification, BertConfig
import torch
import numpy as np
import time
import pandas as pd
from seqeval.metrics import f1_score

def load_data(data, dataset):
    vocab = set()
    sentences = []
    all_tags = []
    input_data = []
    with open(data, encoding="utf8") as f:
        sentence = []
        sentence_tags = []
        if dataset is 'Test':
            for x in f:
                if x == '\n':
                    sentences.append(sentence)
                    sentence = []
                    continue
                sentence.append(x.rstrip())
            return vocab, sentences
        elif dataset is 'Train':
            for x in f:
                if x == '\n':
                    sentences.append(sentence)
                    all_tags.append(sentence_tags)
                    sentence = []
                    sentence_tags = []
                    continue
                word = x.split()[0]
                tag = x.split()[1]
                vocab.add(word)
                sentence.append(word)
                sentence_tags.append(tag)
            return vocab, sentences, all_tags
        elif dataset is 'Dev':
            for x in f:
                if x == '\n':
                    sentences.append(sentence)
                    all_tags.append(sentence_tags)
                    input_data.append((sentence,sentence_tags))
                    sentence = []
                    sentence_tags = []
                    continue
                word = x.split()[0]
                tag = x.split()[1]
                vocab.add(word)
                sentence.append(word)
                sentence_tags.append(tag)
            return vocab, sentences, all_tags, input_data

class BertTransformData():
    def __init__(self, sentences, labels, tokenizer, tag_to_idx, Test=False):
      max_length = 128

      if Test is False:
        tokenized_texts = [tokenize_seq(sentence, label, tokenizer) for sentence, label in zip(sentences,labels)
        ]
        self.tokens = [text[0] for text in tokenized_texts]
        self.labels = [text[1] for text in tokenized_texts]


        self.padded_data = []
        self.attn_masks = []
        for tokens in self.tokens:
            tokens_tmp = ['[CLS]'] + tokens
            padded_tokens = tokens_tmp + ['[PAD]' for _ in range(max_length - len(tokens_tmp))]
            padded_tokens = tokenizer.convert_tokens_to_ids(padded_tokens)
            self.padded_data.append(padded_tokens)
        self.attn_masks = [[float(word > 0) for word in sequence] for sequence in self.padded_data]

        self.padded_labels = []
        for labels in self.labels:
            padded_labels = [tag_to_idx.get(l) for l in labels]
            padded_labels = [-100] + padded_labels + [-100]
            padded_labels = padded_labels + [-100 for _ in range(max_length - len(padded_labels))]
            self.padded_labels.append(padded_labels)

        for words, tags in zip(self.padded_data, self.padded_labels):
              if words[-1] == tokenizer.vocab["[PAD]"]:
                  continue
              else:
                  words[-1] = tokenizer.vocab["[SEP]"]
                  tags[-1] = -100
        
      if Test is True:
          tokenized_texts = [tokenize_seq(sentence, None, tokenizer, Test=True) for sentence in sentences]

          self.tokens = [text[0] for text in tokenized_texts]
          self.mask = [text[1] for text in tokenized_texts]

          self.padded_data = []
          self.attn_masks = []
          for tokens, mask in zip(self.tokens, self.mask):
              tokens_tmp = ['[CLS]'] + tokens
              padded_tokens = tokens_tmp + ['[PAD]' for _ in range(max_length - len(tokens))]
              padded_tokens = tokenizer.convert_tokens_to_ids(padded_tokens)
              self.padded_data.append(padded_tokens)
          self.attn_masks = [[float(word > 0) for word in sequence] for sequence in self.padded_data]

          for words in self.padded_data:
                if words[-1] == tokenizer.vocab["[PAD]"]:
                    continue
                else:
                    words[-1] = tokenizer.vocab["[SEP]"]

def tokenize_seq(sentence, text_labels, tokenizer, Test=False):
  tokenized_sentence = []
  labels = []
  if Test == True:
      mask = []
      for word in sentence:
        tokens = tokenizer.tokenize(word)
        num_tokens = len(tokens)
        tokenized_sentence.extend(tokens)
        if num_tokens > 1:
          mask.extend([1] + [0]*(num_tokens-2) + [0])
        if num_tokens == 1:
          mask.extend([1])
      return tokenized_sentence, mask

  else:
    for word, label in zip(sentence, text_labels):
          tokens = tokenizer.tokenize(word)
          num_tokens = len(tokens)

          tokenized_sentence.extend(tokens)
          if num_tokens > 1:
            labels.extend([label] + ['[PAD]',]*(num_tokens-2)+['[PAD]'])
          if num_tokens == 1:
            labels.extend([label])
    return tokenized_sentence, labels

def parse_data(tokenizer, tag_to_idx, batch_size):
    train_path = 'data/train/train.txt'
    vocab, sentences, labels = load_data(train_path, 'Train')
    train_data = BertTransformData(sentences, labels, tokenizer, tag_to_idx, False)

    train_data = TensorDataset(torch.tensor(train_data.padded_data), torch.tensor(train_data.padded_labels), torch.tensor(train_data.attn_masks))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler
    )
  
    dev_path = 'data/dev/dev.txt'
    dev_vocab, dev_sentences, dev_labels, dev_input_data = load_data(dev_path, 'Dev')
    dev_data = BertTransformData(dev_sentences, dev_labels, tokenizer, tag_to_idx, False)
    dev_data = TensorDataset(torch.tensor(dev_data.padded_data),  torch.tensor(dev_data.padded_labels), torch.tensor(dev_data.attn_masks) )
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(
        dev_data, batch_size=batch_size, sampler=dev_sampler
    )
    
    test_data_path = 'data/test/test.nolabels.txt'
    test_vocab, test_sentences = load_data(test_data_path, 'Test')
    test = BertTransformData(test_sentences, None, tokenizer, tag_to_idx, Test=True)
    test_inputs = torch.tensor(test.padded_data)
    test_mask = torch.tensor(test.attn_masks)

    test_data = TensorDataset(test_inputs, test_mask)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return train_dataloader, dev_dataloader, dev_sentences, test_dataloader, test_sentences

def train_model(tokenizer, tag_to_idx, model, num_epochs, train_dataloader, optimizer, device, dev_dataloader, idx_to_tag):
    all_losses = []
    start=time.time()
    for epoch in range(num_epochs):
        model.train()
        losses = 0
        for i, batch in enumerate (train_dataloader):
            batch = tuple(var.to(device) for var in batch)
            inputs = batch[0]
            mask = batch[2]
            labels = batch[1]            
            outputs = model(inputs, token_type_ids=None, attention_mask=mask, labels=labels)

            loss, probs = outputs[:2]

            losses += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=1
            )

            optimizer.step()
            model.zero_grad()

        loss = losses/len(train_dataloader)
        all_losses.append(loss)

        model.eval()
        dev_loss = 0
        dev_predictions = []
        dev_gt = []
        for i,batch in enumerate(dev_dataloader):
          batch = tuple(var.to(device) for var in batch)
         
          dev_inputs = batch[0]
          dev_mask = batch[2]
          dev_labels = batch[1]

          with torch.no_grad():
            outputs = model(dev_inputs,token_type_ids=None, attention_mask=dev_mask, labels=dev_labels)
            loss, probs = outputs[:2]
            dev_loss += loss.item()

            preds_mask = (dev_labels != -100)

            probs = probs.detach().cpu().numpy()
            filtered_probs = probs[preds_mask.cpu().squeeze()]
      
            preds = np.argmax(filtered_probs, axis=1)
            label_ids = torch.masked_select(dev_labels,(preds_mask ==1))
            label_ids = label_ids.to('cpu').numpy()

          labels = [idx_to_tag[label_id] for label_id in label_ids]
          predictions = [idx_to_tag[pred] for pred in preds]

          dev_predictions.extend(predictions)
          dev_gt.extend(labels)

        dev_acc = f1_score(dev_gt, dev_predictions)

        print('epoch: {}, loss: {}, dev accuracy {}'.format(epoch+1, loss, dev_acc))

    return model, dev_predictions


def evaluate(model, dataloader, device, idx_to_tag):

  model.eval()

  test_predictions = []
  for batch in dataloader:
    inputs = batch[0].to(device)
    mask = batch[1].to(device)

    with torch.no_grad():
      outputs = model(inputs, token_type_ids=None, attention_mask=mask, labels=None)
      probs = outputs[0]
      preds_mask = (mask != 0)
          
      probs = probs.detach().cpu().numpy()
      preds = np.argmax(probs[preds_mask.squeeze().cpu()], axis=1)

      test_predictions.extend(preds)

  test_pred_tags = [idx_to_tag[i] for i in test_predictions]

  return test_pred_tags

def save_preds(file_path, predictions, data):
    formatted_predictions = []
    count = 0
    counts = []
    for sublist in data:
      formatted_predictions.append(predictions[count:count+len(sublist)])
      count += len(sublist)
      counts.append(count)

    with open(file_path, 'w', encoding='utf8') as f:
     for sample in formatted_predictions:
          for label in sample:
              line = "{}\n".format(label)
              f.write(line)
          f.write('\n')

def main(num_epochs, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = ["B", "I", "O"]
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    tag_to_idx = {t: i for i, t in enumerate(classes)}
    tag_to_idx['[PAD]'] = -100
    idx_to_tag = {i: t for t, i in tag_to_idx.items()}

    train_dataloader, dev_dataloader, dev_sentences, test_dataloader, test_sentences = parse_data(tokenizer, tag_to_idx, batch_size=16)

    print('data loaded and tokenized')

    model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(classes))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print('model instantiated')

    model, dev_preds = train_model(tokenizer, tag_to_idx, model, num_epochs, train_dataloader, optimizer, device, dev_dataloader, idx_to_tag)
    test_preds = evaluate(model, test_dataloader, device, idx_to_tag)
    save_preds('dev_preds.txt', dev_preds, dev_sentences)
    save_preds('test_preds.txt', test_preds, test_sentences)

if __name__ == "__main__":
    main(num_epochs = 1, learning_rate = 3e-5)