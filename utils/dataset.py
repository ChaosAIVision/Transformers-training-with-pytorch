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



class TranslationDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Lấy câu song ngữ từ dataset
        example = self.dataset[idx]['text']
        
        # Tách câu thành tiếng Việt và tiếng Anh
        src_text, tgt_text = example.split('###>')
        print(src_text)
        
        # Tokenize
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
        
        # Flatten tensors to remove extra batch dimension
        src_encoding = {key: val.squeeze(0) for key, val in src_encoding.items()}
        tgt_encoding = {key: val.squeeze(0) for key, val in tgt_encoding.items()}
        
        # Tạo input và label cho mô hình
        labels = tgt_encoding['input_ids'].clone()  # Truy cập input_ids như một key trong dictionary
        
        return {
            'input_ids': src_encoding['input_ids'],
            'attention_mask': src_encoding['attention_mask'],
            'labels': labels
        }
