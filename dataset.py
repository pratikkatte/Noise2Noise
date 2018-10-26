
import numpy as np
import os
from skimage import io
import argparse
import string
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader


class ImageNetDataset(Dataset):
    
    def __init__(self, root_dir, data_folder, transform=None):
        self.data_path = os.path.join(root_dir,data_folder)
        self.data = os.listdir(self.data_path)
        self.root = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name = self.data[index]
        img_path = os.path.join(self.data_path,img_name)
        image = io.imread(img_path)
        if self.transform:
            image = self.transform(image)
        
        return image,img_name



class addNoise:
    
    def __init__(self,token, std_range=None):
        self.token = token
        self.std_range = std_range
        
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

            image = img.astype(float)

            noise_mask = numpy.random.poisson(image)

            noisy_img = img + noise_mask

            return noisy_img


class Rescale(object):

    def __call__(self, image):

        if len(image.shape) == 2:
            image = np.stack((image,)*3, axis=-1)
        resize = transforms.Resize((256,256))

        tens = transforms.ToTensor()
        
        img = tens(resize(image))
        print(type(img))
        return img



def get_args():
    parser = argparse.ArgumentParser(description="Data set Prep for image restoration",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_size", type=int, default=256,
                        help="training patch size")
    parser.add_argument("--noise_model", type=str, default="gaussian,0,50",
                        help="noise model to be trained")
    args = parser.parse_args()
    
    return args

