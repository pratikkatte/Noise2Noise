
import numpy as np
import os
import cv2
import string
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from skimage import io

class ImageNetDataset(Dataset):
    
    def __init__(self, dataset_path, transform=None):
        self.data_path = dataset_path

        self.data = os.listdir(self.data_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name = self.data[index]
        img_path = os.path.join(self.data_path,img_name)
        image = io.imread(img_path)
        if self.transform:
            image = self.transform(image)
        
        return image


class addNoise(object):
    
    def __init__(self,token, std_range=None):
        self.token = token
        self.std_range = std_range
        self.min_occupancy = 1
        self.max_occupancy = 20
        
    def __call__(self, img):
        
        if self.token == "gaussian":
            if len(np.shape(img))==2:
                img = np.stack((img,)*3, axis=-1)

            minval,maxval = self.std_range
            noise_img = img.astype(np.float)
            stddev = np.random.uniform(minval, maxval)
            noise = np.random.randn(*img.shape) * stddev
            noise_img += noise
            noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)

            return noise_img

        elif self.token == "text":

            if len(np.shape(img))==2:
                img = np.stack((img,)*3, axis=-1)

            img = img.copy()
            h, w, _ = img.shape
            font = cv2.FONT_HERSHEY_SIMPLEX
            img_for_cnt = np.zeros((h, w), np.uint8)
            occupancy = np.random.uniform(self.min_occupancy, self.max_occupancy)

            while True:
                n = random.randint(5, 10)
                random_str = ''.join([random.choice(string.ascii_letters + string.digits) for i in range(n)])
                font_scale = np.random.uniform(0.5, 1)
                thickness = random.randint(1, 3)
                (fw, fh), baseline = cv2.getTextSize(random_str, font, font_scale, thickness)
                x = random.randint(0, max(0, w - 1 - fw))
                y = random.randint(fh, h - 1 - baseline)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.putText(img, random_str, (x, y), font, font_scale, color, thickness)
                cv2.putText(img_for_cnt, random_str, (x, y), font, font_scale, 255, thickness)

                if (img_for_cnt > 0).sum() > h * w * occupancy / 100:
                    break
            return img

        elif self.token == "poisson":
            if len(np.shape(img))==2:
                img = np.stack((img,)*3, axis=-1)

            image = img.astype(float)

            noise_mask = np.random.poisson(image)

            noisy_img = img + noise_mask
            noisy_img = noisy_img.astype(np.uint8)

            return noisy_img

        else:
            return img



def getTrainLoader(dataset_path, noise_model, batch_size):

    train_dataset = ImageNetDataset(dataset_path,
                                    transform=transforms.Compose([
                                        addNoise(noise_model, (0,50)),
                                        transforms.ToPILImage(),
                                        transforms.Resize((256,256)),
                                        transforms.ToTensor(),
                                    ]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    return(train_loader)


def getLabledLoader(dataset_path, noise_model, batch_size):
    label_dataset = ImageNetDataset(dataset_path,   
                                    transform=transforms.Compose([
                                        addNoise(noise_model, (0,50)),
                                        transforms.ToPILImage(),
                                        transforms.Resize((256,256)),
                                        transforms.ToTensor()
                                    ]))

    label_loader = DataLoader(label_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    return(label_loader)


                    