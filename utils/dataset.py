import torch
from torch.utils.data import Dataset


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

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


class TranslationDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]['text']
        src_text, tgt_text = example.split('###>')

        src_encoding = self.tokenizer(
            src_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        tgt_encoding = self.tokenizer(
            tgt_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        # Remove the batch dimension
        src_encoding = {key: val.squeeze(0) for key, val in src_encoding.items()}
        tgt_encoding = {key: val.squeeze(0) for key, val in tgt_encoding.items()}

        labels = tgt_encoding['input_ids'].clone()

        # Tạo mask cho encoder và decoder
        encoder_mask = (src_encoding['input_ids'] != self.tokenizer.pad_token_id).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len)
        decoder_mask = (tgt_encoding['input_ids'] != self.tokenizer.pad_token_id).unsqueeze(0).int() & causal_mask(tgt_encoding['input_ids'].size(0))  # (1, seq_len, seq_len)

        return {
            'input_ids': src_encoding['input_ids'],
            'attention_mask': encoder_mask,  # Attention mask cho encoder
            'decoder_input': tgt_encoding['input_ids'],  # Decoder input
            'decoder_mask': decoder_mask,  # Attention mask cho decoder
            'labels': labels  # Labels cho mô hình
        }
