import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from utils.dataset import TranslationDataset
from models.transformers_seq2seq import TransformerSeq2Seq
from torch.utils.tensorboard import SummaryWriter
import os
from pathlib import Path
import torchmetrics

def get_args():
    parser = argparse.ArgumentParser(description="Train Transformer Seq2Seq from scratch")
    parser.add_argument('--data_yaml', "-d", type=str, help='Path to dataset', default='/path/to/your/data.yaml')
    parser.add_argument('--batch_size', '-b', type=int, help='input batch_size', default=16)
    parser.add_argument('--epochs', '-e', type=int, default=10)
    parser.add_argument('--learning_rate', '-l', type=float, default=2e-5)
    parser.add_argument('--resume', action='store_true', help='True if want to resume training')
    parser.add_argument('--pretrain', action='store_true', help='True if want to use pre-trained weights')

    return parser.parse_args()

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.convert_tokens_to_ids('[SOS]')
    eos_idx = tokenizer_tgt.convert_tokens_to_ids('[EOS]')

    encoder_output = model.encoder(source, source_mask)
    decoder_input = torch.tensor([[sos_idx]], dtype=torch.long, device=device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decoder(decoder_input, encoder_output, source_mask, decoder_mask)
        prob = model.projection(out[:, -1])
        _, next_word = torch.max(prob, dim=1)

        decoder_input = torch.cat(
            [decoder_input, torch.tensor([[next_word.item()]], dtype=torch.long, device=device)], dim=1
        )

        if next_word.item() == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model, validation_loader, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    with torch.no_grad():
        for batch in validation_loader:
            count += 1
            encoder_input = batch["input_ids"].to(device)
            encoder_mask = batch["attention_mask"].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = tokenizer_src.decode(encoder_input[0].tolist(), skip_special_tokens=True)
            target_text = batch["labels_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy(), skip_special_tokens=True)

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            print_msg('-' * 80)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-' * 80)
                break

    if writer:
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)

        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)

        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)

        writer.flush()

def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Load dataset
    datasets = load_dataset(
    'csv', 
    data_files={
        'train': '/Users/chaos/Documents/Chaos_working/Chaos_datasets/vietnam_english_dataset/train_dataset.csv',
        'validation': '/Users/chaos/Documents/Chaos_working/Chaos_datasets/vietnam_english_dataset/validation_dataset.csv'
    }
)

    train_dataset = datasets['train']
    valid_dataset = datasets['validation']

    # Initialize tokenizer
    tokenizer_src = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    tokenizer_tgt = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    # Create Dataset and DataLoader
    train_data = TranslationDataset(train_dataset, tokenizer_src)
    valid_data = TranslationDataset(valid_dataset, tokenizer_tgt)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=1)

    # Initialize model
    vocab_size = tokenizer_src.vocab_size
    max_length = tokenizer_src.model_max_length
    d_model = 768
    num_heads = 12
    ff_dim = 3072
    dropout = 0.1

    model = TransformerSeq2Seq(
        vocab_size=vocab_size,
        max_length=max_length,
        d_model=d_model,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
        device=device
    ).to(device)

    # Initialize optimizer and loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter(f'runs/{args.data_yaml}')

    best_loss = float('inf')

    for epoch in range(args.epochs):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in batch_iterator:
            encoder_input = batch['input_ids'].to(device)
            decoder_input = batch['input_ids'].to(device)
            encoder_mask = batch['attention_mask'].to(device)
            decoder_mask = causal_mask(decoder_input.size(1)).to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                encoder_output = model.encoder(encoder_input, encoder_mask)
                decoder_output = model.decoder(decoder_input, encoder_output, encoder_mask, decoder_mask)
                proj_output = model.projection(decoder_output)

                label = batch['labels'].to(device)
                loss = loss_fn(proj_output.view(-1, vocab_size), label.view(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            batch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})
            writer.add_scalar("Train Loss", loss.item(), epoch * len(train_loader) + batch_iterator.n)

        run_validation(model, valid_loader, tokenizer_src, tokenizer_tgt, max_length, device, lambda msg: batch_iterator.write(msg), epoch, writer)

        model_filename = f"model_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), model_filename)

if __name__ == "__main__":
    args = get_args()
    train_model(args)
