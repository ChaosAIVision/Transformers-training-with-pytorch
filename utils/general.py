import pandas as pd
import numpy as np
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import io
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import numpy as np
import pickle
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import yaml
from PIL import Image

def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state= 42)
    df.drop(['id', 'keyword', 'location' ], axis= 1, inplace= True)
    df.reset_index(inplace= True, drop= True)

    return df

def separate_train_valid(df):
    test_df = df.sample(frac= 0.1, random_state =42)
    train_df = df.drop(test_df.index)
    print(f'Using {len(train_df)} samples for training and {len(test_df)} for validation')
    train_df = train_df.to_numpy()
    test_df = test_df.to_numpy()

    return train_df, test_df

def yiel_token(data_inter, english_tokenizer):
    for data in data_inter:
        yield english_tokenizer(data[0])


class ManagerDataYaml:
    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.data = None

    def load_yaml(self) -> dict:
        """
        Load data from YAML file and return its properties as a dictionary.
        """
        try:
            with open(self.yaml_path, 'r') as file:
                self.data = yaml.safe_load(file)
                return self.data
        except Exception as e:
            return f"Error loading YAML file: {self.yaml_path}. Exception: {e}"

    def get_properties(self, key: str) :
        """
        Get the value of a specific property from the loaded YAML data.
        """
        if isinstance(self.data, dict):
            if key in self.data:
                value = self.data[key]
                return (value)
            else:
                return f"Key '{key}' not found in the data."
        else:
            return "Data has not been loaded or is not a dictionary."
        

class ManageSaveDir():
    def __init__(self, data_yaml):
        data_yaml_manage = ManagerDataYaml(data_yaml)
        data_yaml_manage.load_yaml()
        self.save_dir_locations = data_yaml_manage.get_properties('save_dirs')
        self.train_dataset = data_yaml_manage.get_properties('train')
        self.valid_dataset = data_yaml_manage.get_properties('valid')
        self.test_dataset = data_yaml_manage.get_properties('test')
        self.categories = data_yaml_manage.get_properties('categories')

    def create_save_dir(self):
        if not os.path.exists(self.save_dir_locations):
            return f'Folder path {self.save_dir_locations} is not exists'
        else:
            self.result_dir = os.path.join(self.save_dir_locations, 'result')
            weight_dir = os.path.join(self.result_dir, 'weights')
            tensorboard_dir = os.path.join(self.result_dir, 'tensorboard')

            if not os.path.exists(self.result_dir):
                os.makedirs(self.result_dir)
                os.makedirs(weight_dir)
                os.makedirs(tensorboard_dir)
                return weight_dir, tensorboard_dir # Cần return tensorbard_dir để lấy location  ghi log và weight
            else:
                counter = 1
                while True:
                    self.result_dir = os.path.join(self.save_dir_locations, f'result{counter}')
                    weight_dir = os.path.join(self.result_dir, 'weights')
                    tensorboard_dir = os.path.join(self.result_dir, 'tensorboard')
                    if not os.path.exists(self.result_dir):
                        os.makedirs(self.result_dir)
                        os.makedirs(weight_dir)
                        os.makedirs(tensorboard_dir)
                        return weight_dir, tensorboard_dir
                    counter += 1
    def get_save_dir_path(self):
        return self.result_dir
    def count_items_in_folder(self, folder_path):
        try:
            items = os.listdir(folder_path)
            num_items = len(items)
            return num_items
        except FileNotFoundError:
            return f'The folder {folder_path} does not exist'
        except PermissionError:
            return f'Permission denied to access the folder {folder_path}'
        
    def count_distribution_labels(self, mode):
        if mode == 'train':
            data_path = self.train_dataset
        elif mode == 'valid':
            data_path = self.valid_dataset
        else:
            data_path = self.test_dataset

        num_categories = []
        for category in self.categories:
            categories_path = os.path.join(data_path, category)
            num_labels = self.count_items_in_folder(categories_path)
            num_categories.append(num_labels)
        return num_categories

def save_plots_from_tensorboard(tensorboard_folder, output_image_folder):
    # Khởi tạo EventAccumulator để đọc các tệp sự kiện trong thư mục
    event_accumulator = EventAccumulator(tensorboard_folder)
    event_accumulator.Reload()

    # Lấy tất cả các tags từ TensorBoard
    scalar_tags = event_accumulator.Tags()['scalars']
    
    # Lấy dữ liệu cho các scalar tags
    def get_scalar_data(tag):
        events = event_accumulator.Scalars(tag)
        steps = [event.step for event in events]
        values = [event.value for event in events]
        return steps, values

    # Tạo tấm ảnh đầu tiên với 8 biểu đồ
    plt.figure(figsize=(16, 8))
    
    # Các tag cho huấn luyện
    train_tags = ['Train/mean_loss', 'Train/mAP50']
    # Các tag cho kiểm tra
    valid_tags = ['Valid/mean_loss', 'Valid/mAP50']

    # Vẽ các biểu đồ cho tập huấn luyện
    for i, tag in enumerate(train_tags):
        plt.subplot(2, 4, i + 1)
        steps, values = get_scalar_data(tag)
        plt.plot(steps, values)
        plt.title(tag)
        plt.xlabel('Steps')
        plt.ylabel('Value')

    # Vẽ các biểu đồ cho tập kiểm tra
    for i, tag in enumerate(valid_tags):
        plt.subplot(2, 4, i + 5)
        steps, values = get_scalar_data(tag)
        plt.plot(steps, values)
        plt.title(tag)
        plt.xlabel('Steps')
        plt.ylabel('Value')

    plt.tight_layout()
    plt.savefig(os.path.join(output_image_folder, 'training_and_validation_plots.png'))
    plt.close()

    # Lấy dữ liệu ma trận nhầm lẫn
    confusion_matrix_tag = 'confusion_matrix'
    image_events = event_accumulator.Images(confusion_matrix_tag)

    if image_events:
        # Lấy bước cuối cùng từ các sự kiện hình ảnh
        last_step = max(event.step for event in image_events)
        
        # Lọc sự kiện hình ảnh với bước cuối cùng
        for event in reversed(image_events):
            if event.step == last_step:
                image_string = event.encoded_image_string
                image = Image.open(io.BytesIO(image_string))
                
                plt.figure(figsize=(8, 8))
                plt.imshow(image)
                plt.title('Confusion Matrix')
                plt.axis('off')
                plt.savefig(os.path.join(output_image_folder, 'confusion_matrix.png'))
                plt.close()
                break
    else:
        print("No confusion matrix found in TensorBoard logs.")
