import torch
from torch.utils.data import Dataset

class Tweets_Dataset(Dataset):
    def __init__(self, data, max_len, vocab, tokenizer):
        self.data = data
        self.max_len = max_len
        self.vocab = vocab
        self.tokenizer = tokenizer
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        text = self.data[idx][0]
        label = self.data[idx][1]

        text_processed = self.vocab(self.tokenizer(text))
        text_processed = text_processed[:self.max_len-2]
        text_processed = [self.vocab["<s>"]] + text_processed + [self.vocab["<s>"]]
        
        if len(text_processed) < self.max_len:
            pad_size = self.max_len - len(text_processed)
            text_processed += [self.vocab['<ad>']] * pad_size

        text_processed = torch.tensor(text_processed)
        label = torch.tensor(label)

        return text_processed, label
        
        # return {'input_ids': text_processed,
        #         'labels': label}

