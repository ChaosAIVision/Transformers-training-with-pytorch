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
    parser.add_argument('--batch_size', '-b', type=int, help='input batch_size', default=2)
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

def train_fn(train_loader, model, optimizer, loss_fn, epoch, total_epochs, writer, device):
    model.train()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs}", leave=True, colour='green')
    mean_loss = []

    for batch_idx, batch in enumerate(progress_bar):
        encoder_input = batch['input_ids'].to(device)
        decoder_input = batch['decoder_input'].to(device)
        encoder_mask = batch['attention_mask'].to(device)
        decoder_mask = batch['decoder_mask'].to(device)

        optimizer.zero_grad()
        # Remove mixed precision support
        encoder_output = model.encoder(encoder_input, encoder_mask)

        decoder_output = model.decoder(decoder_input, encoder_output, encoder_mask, decoder_mask)

        proj_output = model.projection(decoder_output)

        label = batch['labels'].to(device)
        loss = loss_fn(proj_output.view(-1, proj_output.size(-1)), label.view(-1))
        loss.backward()
        optimizer.step()

        mean_loss.append(loss.item())
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = sum(mean_loss) / len(mean_loss)
    writer.add_scalar("Train Loss", avg_loss, epoch)
    return avg_loss
def val_fn(valid_loader, model, tokenizer_src, tokenizer_tgt, max_len, device, writer, epoch):
    model.eval()
    progress_bar = tqdm(valid_loader, desc=f"Validation Epoch {epoch + 1}", leave=True, colour='yellow')
    source_texts, expected, predicted = [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
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

            progress_bar.set_postfix({'prediction': model_out_text})

    metric_cer = torchmetrics.CharErrorRate()
    cer = metric_cer(predicted, expected)
    writer.add_scalar('Validation CER', cer, epoch)

    metric_wer = torchmetrics.WordErrorRate()
    wer = metric_wer(predicted, expected)
    writer.add_scalar('Validation WER', wer, epoch)

    metric_bleu = torchmetrics.BLEUScore()
    bleu = metric_bleu(predicted, expected)
    writer.add_scalar('Validation BLEU', bleu, epoch)

    return cer, wer, bleu

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    datasets = load_dataset(
    'csv', 
    data_files={
        'train': '/home/chaos/Documents/ChaosAIVision/dataset/viet2eng/train_dataset.csv',
        'validation': '/home/chaos/Documents/ChaosAIVision/dataset/viet2eng/validation_dataset.csv'
    }
)

    train_dataset = datasets['train']
    valid_dataset = datasets['validation']

    tokenizer_src = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    tokenizer_tgt = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    train_data = TranslationDataset(train_dataset, tokenizer_src)
    valid_data = TranslationDataset(valid_dataset, tokenizer_tgt)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=1)

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

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    writer = SummaryWriter(f'runs/{args.data_yaml}')

    for epoch in range(args.epochs):
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, epoch, args.epochs, writer, device)
        cer, wer, bleu = val_fn(valid_loader, model, tokenizer_src, tokenizer_tgt, max_length, device, writer, epoch)

        print(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss:.4f}, CER: {cer:.4f}, WER: {wer:.4f}, BLEU: {bleu:.4f}")

        model_filename = f"model_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), model_filename)

    writer.close()

if __name__ == "__main__":
    args = get_args()
    train(args)
