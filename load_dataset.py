from datasets import load_dataset

ds = load_dataset("kaitchup/opus-Vietnamese-to-English")

train_dataset = ds['train']
validation_dataset = ds['validation']

ds['train'].to_csv('train_dataset.csv')
ds['validation'].to_csv('validation_dataset.csv')