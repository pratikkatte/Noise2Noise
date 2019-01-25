import os
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from skimage import io, transform

    
class addGaussian:
    
    def __init__(self, std_range):
        self.std_range = std_range

    def __call__(self,img):
        
        if len(np.shape(img))==2:
            img = np.stack((img,)*3, axis=-1)
    
        minval,maxval = self.std_range
        noise_img = img.astype(np.float)
        stddev = np.random.uniform(minval, maxval)
        noise = np.random.randn(*img.shape) * stddev
        noise_img += noise
        noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
        return noise_img

class GaussianDataset:

    def init(self, config):
        
        self.data_path = os.path.join(config['root_dir'],config['data_folder'])
        self.data = sorted(os.listdir(self.data_path))[0:100]

        transform_list = [addGaussian((0,50)),
                     transforms.ToPILImage(),
                     transforms.Resize((256,256)),
                     transforms.ToTensor()]
        
        self.transform = transforms.Compose(transform_list)
        
    def __getitem__(self, index):
        img_name = self.data[index]
        
        img_path = os.path.join(self.data_path,img_name)
        
        image = io.imread(img_path)
        source_image = self.transform(image)
        target_image = self.transform(image)

        return source_image, target_image
    
    def __len__(self):
        return len(self.data)



class TextDataset:
    
    def init(self, config):
        pass
    
    def __get_item__(self, index):
        pass

class addText:
    
    def __init__(self):
        pass
    
    def __call__(self, img):
        pass


def createDataset(config):
    dataset = None
    if config['dataset'] == 'gaussian':
        dataset = GaussianDataset()
    if config['dataset'] == 'Text':
        dataset = addText()
    
    dataset.init(config)
    print('%s dataset was created.' % config['dataset'])
 
    return dataset


class datasetLoader:
    
    def __init__(self, config):
        self.dataset = createDataset(config)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size = config['batch_size'],
            shuffle = True,
            num_workers = int(config['num_workers']))

    def load_data(self):
        return self.dataloader
    
    def __len__(self):
        return len(self.dataloader)