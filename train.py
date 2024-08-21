from utils.general import load_csv, yiel_token, separate_train_valid, ManagerDataYaml, ManageSaveDir, save_plots_from_tensorboard
from torchtext.data.utils import get_tokenizer
from utils.dataset import Tweets_Dataset
import torchtext
from torchtext.vocab import build_vocab_from_iterator
import warnings
import torch
from torch.utils.tensorboard import SummaryWriter
# warnings.filterwarnings("ignore")
# torchtext.disable_torchtext_deprecation_warning()
warnings.warn(torchtext._TORCHTEXT_DEPRECATION_MSG)
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from termcolor import colored
import argparse
from models.transformers import TransformerTextCls
import torch.nn as nn
import os


def get_args():
    parser = argparse.ArgumentParser(description="Train YOLOv1 from scratch")
    parser.add_argument('--data_yaml', "-d", type= str, help= 'Path to dataset', default= '/Users/chaos/Documents/Chaos_working/Chaos_projects/VGG16-from-scratch-Pytorch/data.yaml')
    parser.add_argument('--batch_size', '-b', type = int, help = 'input batch_size')
    parser.add_argument('--epochs', '-e', type= int, default= 100)
    parser.add_argument('--learning_rate', '-l', type= float, default= 2e-5)
    parser.add_argument('--resume', action='store_true', help='True if want to resume training')
    parser.add_argument('--pretrain', action='store_true', help='True if want to use pre-trained weights')

    return parser.parse_args()  # Cần trả về kết quả từ parser.parse_args()



def train_fn(train_loader, model, optimizer, loss_fn, epoch, total_epochs, scaler = None):
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    mean_loss = []
    running_correct = 0
    total = 0

    # Dùng `leave=True` để đảm bảo tqdm hoàn thành thanh tiến trình đúng cách
    progress_bar = tqdm(train_loader, colour='green', desc=f"Epochs: {epoch + 1}/{total_epochs}", leave=True)

    for batch_idx, (x, y) in enumerate(progress_bar):
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)) or \
            torch.any(torch.isnan(y)) or torch.any(torch.isinf(y)):
                continue
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out = model(x)
            loss = loss_fn(out, y)
            mean_loss.append(loss.item())

        progress_bar.set_postfix({'loss': f'{loss.item():0.4f}'})

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        _, predicted = torch.max(out,1 )

        running_correct += (predicted == y).sum().item()
        total += y.size(0)

    avg_loss = sum(mean_loss) / len(mean_loss)
    avg_accuracy =   running_correct / total
    print(colored(f"\nTrain \t loss: {avg_loss:3.10f} \t accuracy: {avg_accuracy:3.10f}\n", 'green'))

    return avg_accuracy, avg_loss


def test_fn(test_loader, model, loss_fn, epoch, total_epochs, scaler = None):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model.eval()
    running_correct = 0
    total = 0
    mean_loss = []

    progress_bar = tqdm(test_loader, colour='yellow', desc=f"Epochs: {epoch + 1}/{total_epochs}", leave=True)

    for batch_idx, (x, y) in enumerate(progress_bar):
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)) or \
            torch.any(torch.isnan(y)) or torch.any(torch.isinf(y)):
                continue
        x, y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast():
            out = model(x)
            loss = loss_fn(out, y)


        progress_bar.set_postfix({'loss': f'{loss.item():0.4f}'})
        mean_loss.append(loss.item())
        _, predicted = torch.max(out,1 )

        running_correct += (predicted == y).sum().item()
        total += y.size(0)



    avg_loss = sum(mean_loss) / len(mean_loss)
    avg_accuracy =   running_correct / total
    print(colored(f"\nTest \t loss: {avg_loss:3.10f} \t accuracy: {avg_accuracy:3.10f}\n", 'yellow'))
    return avg_accuracy, avg_loss

def train(args):
    seed = 123
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    data_yaml_manage = ManagerDataYaml(args.data_yaml)
    data_yaml_manage.load_yaml()
    pretrain_weight = data_yaml_manage.get_properties(key='pretrain_weight')
    train_path = data_yaml_manage.get_properties(key='train')
    df = load_csv(train_path)

    english_tokenizer = get_tokenizer('basic_english')
    train_df, valid_df = separate_train_valid(df)
    vocab_size = 2000
    vocab = build_vocab_from_iterator(yiel_token(train_df,english_tokenizer),
                                    min_freq= 3,
                                    max_tokens= vocab_size,
                                    specials= ["<pad>", "<s>", "</s>", '<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    max_length = 200
    train_data = Tweets_Dataset(train_df, max_length,  vocab, english_tokenizer)
    test_data = Tweets_Dataset(valid_df, max_length,  vocab, english_tokenizer)

    train_loader = DataLoader(
        dataset= train_data,
        batch_size= args.batch_size,
        shuffle= True,
        pin_memory= True,
        drop_last= True

    )

    valid_loader = DataLoader(dataset= test_data, 
                            batch_size= args.batch_size,
                            shuffle= False,
                            pin_memory= True,
                            drop_last= True)
    vocab_size = 30522  # Kích thước từ vựng, ví dụ như kích thước từ vựng của DistilBERT
    max_length = 128    # Độ dài tối đa của chuỗi đầu vào
    embed_dim = 768     # Kích thước của các embedding vectors, ví dụ như kích thước embedding của DistilBERT
    num_heads = 12      # Số lượng đầu của multi-head attention
    ff_dim = 3072       # Kích thước của feed-forward network, tương tự như DistilBERT
    dropout = 0.1       # Tỷ lệ dropout

    # Chọn thiết bị (CPU hoặc GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Khởi tạo mô hình
    model = TransformerTextCls(
        vocab_size=vocab_size,
        max_length=max_length,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
        device=device
    ).to(device)
    ##############################################################################################

    # pretrain_weight = '/home/chaos/Documents/ChaosAIVision/temp_folder/backbone448/weights/last.pt'
    # checkpoint = torch.load(pretrain_weight)
    # backbone_state_dict = rename_keys(checkpoint['model_state_dict'])


   

    # model = torch.compile(model)
  
    # model.darknet.load_state_dict(backbone_state_dict, strict=False)
    print('Loading backbone pretrain successfully !')
    # for param in model.darknet.parameters():
    #     param.requires_grad = False
    #################################################################################################
    # # Kiểm tra xem các tham số của backbone đã được đóng băng chưa
    # for name, param in model.named_parameters():
    #     print(f"{name}: requires_grad={param.requires_grad}")

    
    # Use DataParallel if more than 1 GPU is available

    
    loss_fn = nn.CrossEntropyLoss(reduce= 'sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate, momentum= 0.9)

    # if args.pretrain:
    #     state_dict = torch.load(pretrain_weight)
    #     model.load_state_dict(state_dict, strict=False)
        # print('Loaded pretrain weights successfully!')
    
  
  

    model.to(device)
    scaler = torch.cuda.amp.GradScaler()
    locate_save_dir = ManageSaveDir(args.data_yaml)
    weights_folder, tensorboard_folder = locate_save_dir.create_save_dir()  # Get save directories for weights and logs
    save_dir = locate_save_dir.get_save_dir_path()
    writer = SummaryWriter(tensorboard_folder)

    #TRAIN
    print(f'Results will be saved at {save_dir}')

    best_accuracy_train = 0
    best_accuracy_test = 0


    for epoch in range(args.epochs):
        train_mAP, train_avg_loss = train_fn(train_loader, model, optimizer, loss_fn, epoch, args.epochs,scaler)
        valid_mAP, valid_avg_loss = test_fn(valid_loader, model, loss_fn, epoch, args.epochs,scaler)
        # Write mAP and meanLoss to plot
        writer.add_scalar("Train/mAP50", train_mAP, epoch)
        writer.add_scalar("Train/mean_loss", train_avg_loss, epoch)
        writer.add_scalar("Valid/mAP50", valid_mAP, epoch)
        writer.add_scalar("Valid/mean_loss", valid_avg_loss, epoch)

        checkpoint = {
            'model_state_dict': model.state_dict()}
        torch.save(checkpoint, os.path.join(weights_folder, 'last.pt'))


        # Update the best mAP for train and test
        if train_mAP > best_accuracy_train:
            best_accuracy_train = train_mAP

        if valid_mAP > best_accuracy_test:
            best_accuracy_test = valid_mAP
            torch.save(checkpoint, os.path.join(weights_folder, 'best.pt'))


    print(colored(f"Best Train accuracy: {best_accuracy_train:3.10f}", 'green'))
    print(colored(f"Best Test accuracy: {best_accuracy_test:3.10f}", 'yellow'))
 
    save_plots_from_tensorboard(tensorboard_folder, save_dir)


        


if __name__ == "__main__":
    args = get_args()
    data_yaml = args.data_yaml
    train(args)