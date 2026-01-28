import numpy as np
import pickle

import os, pickle, numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from time import time
import tqdm


class COCO_Feeder(Dataset):
    def __init__(self, data_path, label_path, gait_path=None, ignore_empty_sample=True, mode='train'):
        self.data_path = data_path
        self.label_path = label_path
        self.gait_path = gait_path
        self.ignore_empty_sample = ignore_empty_sample
        self.mode = mode

        self.load_data()

    def load_data(self):
        # Load labels
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)
        self.label = np.array(self.label)

        # Load data
        data_file = os.path.join(self.data_path, f'{self.mode}_data.npy')
        self.data = np.load(data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_numpy = self.data[index]

        # Load gait parameters if available
        gait_params = None
        if self.gait_path:
            gait_file = os.path.join(self.gait_path, self.sample_name[index].replace('_data.npy', '_gait.npy'))
            if os.path.exists(gait_file):
                gait_params = np.load(gait_file).astype(np.float32)

        label = self.label[index]

        if gait_params is not None:
            return data_numpy, label, gait_params
        else:
            return data_numpy, label

# Load train data using COCO_Feeder
train_feeder = COCO_Feeder(data_path='outputs/coco', label_path='outputs/coco/train_label.pkl', mode='train')

print("Train data length:", len(train_feeder))
data, label = train_feeder[0]
print("First train sample shape:", data.shape)
print("First train label:", label)
print("First train name:", train_feeder.sample_name[0])

# Load eval data using COCO_Feeder
eval_feeder = COCO_Feeder(data_path='outputs/coco', label_path='outputs/coco/eval_label.pkl', mode='eval')

print("Eval data length:", len(eval_feeder))
data, label = eval_feeder[0]
print("First eval sample shape:", data)
print("First eval label:", label)
print("First eval name:", eval_feeder.sample_name[0])

# Create DataLoader for testing
train_loader = DataLoader(train_feeder, batch_size=4, shuffle=False, num_workers=0)

# Test the data loading with a simple loop
print("\nTesting data loading with DataLoader:")
start_time = time()
num_samples = 0
for num, batch in enumerate(tqdm.tqdm(train_loader, dynamic_ncols=True)):
    # Unpack batch
    if len(batch) == 3:
        x, y, gait_params = batch
        print(batch)
        if gait_params is not None:
            if isinstance(gait_params, torch.Tensor):
                gait_params = gait_params.float()
            elif isinstance(gait_params, tuple):
                gait_params = tuple(torch.tensor(t).float() for t in gait_params)
            else:
                gait_params = torch.tensor(gait_params).float()
        else:
            gait_params = None
    else:
        x, y = batch
        gait_params = None
    
    num_samples += len(x)
    
    if num == 0:  # Print first batch details
        print(f"First batch x shape: {x.shape}")
        print(f"First batch y: {y}")
        if gait_params is not None:
            print(f"First batch gait_params shape: {gait_params.shape}")
        else:
            print("No gait_params in first batch")
    
    if num >= 4:  # Test only first 5 batches
        break

end_time = time()
print(f"Processed {num_samples} samples in {end_time - start_time:.2f} seconds")