from datasets import load_dataset
from utils.dataset import TranslationDataset

datasets = load_dataset(
    'csv', 
    data_files={
        'train': '/Users/chaos/Documents/Chaos_working/Chaos_datasets/vietnam_english_dataset/train_dataset.csv',
        'validation': '/Users/chaos/Documents/Chaos_working/Chaos_datasets/vietnam_english_dataset/validation_dataset.csv'
    }
)

train_dataset = datasets['train']
valid_dataset = datasets['validation']

# sample_train_data = train_dataset.select(range(5))

# # In ra dữ liệu
# print("Sample Train Dataset:")
# for example in sample_train_data:
#     print(example)




from transformers import AutoTokenizer

# Giả sử bạn sử dụng tokenizer của BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# Tạo dataset
train_data = TranslationDataset(train_dataset, tokenizer)
valid_data = TranslationDataset(valid_dataset, tokenizer)

# Sử dụng DataLoader để tạo batch cho training
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=4)

# Kiểm tra một batch từ train_dataloader
for batch in train_dataloader:
    print(batch)
    break