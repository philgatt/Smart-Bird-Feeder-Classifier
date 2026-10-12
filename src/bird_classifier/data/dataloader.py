from torch.utils.data import Dataset, Dataloader
from PIL import Image
import numpy as np
import torch

class BirdDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        # self.mean = 
        # self.std = 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = self.labels[idx]

        image = image.convert("RGB")
        image = image.resize((224, 224))
        image = np.asarray(image)
        image = image / 255.0 

        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute(2, 0, 1) # HWC to CHW
        # image = (image - self.mean) / self.std 

        return image, label

def create_dataloader(dataset, batch_size, shuffle=True):
    return Dataloader(dataset,
                      batch_size = batch_size, 
                      shuffle = shuffle
    )





    