import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms.functional as TF

class GroupRandomTransform:
    def __call__(self, images):
        transformed_images = []
        for img in images:
            img = TF.to_tensor(img)
            transformed_images.append(img)
        return transformed_images

class CustomDataset(Dataset):
    def __init__(self, label_folder, data_folder, file, mode, transform=None):
        self.file = file
        self.label_folder = label_folder
        self.data_folder = data_folder
        self.transform = transform
        self.data = []

        with open(os.path.join(label_folder, file), "r+") as reader:
            for line in reader:
                self.data.append(line.strip("\n"))
        print(self.file)
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        people, side, label2, gender = line.split(";")

        label2 = int(label2)
        gender = int(gender)

        people_folder = os.path.join(self.data_folder, people, people + "_" + side)
        closed_pd_images = [Image.open(os.path.join(people_folder, f"Closed-PD{i}.jpg")).convert('RGB') for i in range(1, 4)]
        closed_t2w_images = [Image.open(os.path.join(people_folder, f"Closed-T2W{i}.jpg")).convert('RGB') for i in range(1, 4)]
        open_pd_images = [Image.open(os.path.join(people_folder, f"Open-PD{i}.jpg")).convert('RGB') for i in range(1, 4)]

        if self.transform:
            closed_pd_images = [self.transform(img) for img in closed_pd_images]
            closed_t2w_images = [self.transform(img) for img in closed_t2w_images]
            open_pd_images = [self.transform(img) for img in open_pd_images]

        return closed_pd_images, closed_t2w_images, open_pd_images, label2, label3, gender