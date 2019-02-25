import torch
import numpy as np
from datetime import datetime
from utils import *
from network import *
from dataset import *
import argparse
from summary import viewSummary

def true_image_dataset(config, image_name):
    import os
    from skimage import io
    import torchvision.transforms as transforms
    
    path = os.path.join(config['root_dir'], config['data_folder'])
    image_path = path + image_name[0]
    image = io.imread(image_path)
    try:
        if len(np.shape(image))==2:
            image = np.stack((image)*3, axis=-1)
    
        transform_list = [transforms.ToPILImage(),
                          transforms.Resize((256,256)),
                          transforms.ToTensor()]

        transform = transforms.Compose(transform_list)
        img = transform(image)
        return img
    except:
        print(image_path, image.shape)
    

class Noise2Noise(object):
    
    def __init__(self, config, trainloader, validloader, trainable):
        
        self.model  = UNet().to(config['device'])
        self.trainable = trainable

        if self.trainable:
            self.train_loader = trainloader.load_data()
            self.valid_loader = validloader.load_data()
            
            self.loss, self.optimiser, self.schedular = lossOptimiser(self.model, config)
            self.loss_size = 0
            self.epoch = 0

            self.loss_meter = AvgMeter()
            self.psnr_meter = AvgMeter()
                                           
        self.summary = viewSummary(config['log_dir'])

    def eval(self, config, epoch, stats):

        self.model.train(False)
        
        valid_start = datetime.now()

    
        for valid_steps, data in enumerate(self.valid_loader):
        
            inputs, labels, valid_name = data
            
            
            inputs = inputs.float()
            labels = inputs.float()
        
            inputs = inputs.to(config['device'])
            labels = labels.to(config['device'])
        
            outputs = self.model(inputs)
            
            self.loss_size = self.loss(outputs, labels)
            self.loss_meter.update(self.loss_size.item())
            
            if valid_steps%10==0:
                print("validated", valid_steps)

        for i in range(config['batch_size_valid']):
            labels = labels.cpu()
            outputs = outputs.cpu()
            self.psnr_meter.update(psnr(outputs[i], labels[i]).item())
        
        valid_loss = self.loss_meter.avg
        valid_time = time_elapsed_since(valid_start)[0]
        psnr_avg = self.psnr_meter.avg
        
        true_image = true_image_dataset(config, valid_name)
        self.summary.view_image({'true_images':true_image.view(-1,3,256,256)[:1].cpu().numpy(),
                                     'output_images':outputs.view(-1,3,256,256)[:1].detach().cpu().numpy()}, epoch)
        return valid_loss, valid_time, psnr_avg


    def train(self, config):

        self.model.train(True)

        stats = {
        'train_loss': [],
        'valid_loss': [],
        'valid_psnr': []
        }

        for epoch in range(self.epoch + 1, config['num_epochs']):

            epoch_start_time = datetime.now()

            running_loss = 0.0

            total_train_loss = 0

            epoch_start = datetime.now()

            train_loss_meter = AvgMeter()
            loss_meter = AvgMeter()
            time_meter = AvgMeter()


            # TODO 
            for total_steps, data in enumerate(self.train_loader):

                batch_start = datetime.now()
                iter_time_start = datetime.now()

    #             print(inputs[0].shape)
                inputs, labels, name = data
                inputs = inputs.float()
                labels = labels.float()
                inputs = inputs.to(config['device'])
                labels = labels.to(config['device'])

                outputs = self.model(inputs)

                self.loss_size = self.loss(outputs, labels)
                loss_meter.update(self.loss_size.item())

                self.optimiser.zero_grad()
                self.loss_size.backward()
                self.optimiser.step()

                time_meter.update(time_elapsed_since(batch_start)[1])

                if total_steps % 20 == 0:
                    running_loss += self.loss_size.item()

                    t = (datetime.now() - iter_time_start) / config['batch_size_train']
                    print("Epoch: {}, total_steps: {:d}, train_loss: {}, time: {}".format(epoch, total_steps, running_loss, t))

                    train_loss_meter.update(loss_meter.avg)

                    loss_meter.reset()
                    time_meter.reset()


            print('validating and saving the model at the end of epoch %d \t Time Taken: %d sec' %
                  (epoch, (datetime.now() - epoch_start_time).total_seconds()))

            valid_loss, valid_time, valid_psnr = self.eval(config, epoch, stats)
            
            self.schedular.step(valid_loss)

            stats['train_loss'].append(train_loss_meter.avg)
            stats['valid_loss'].append(valid_loss)
            stats['valid_psnr'].append(valid_psnr)
            self.summary.view_stats(stats, epoch)
            save_network(self.model,'latest.pt', epoch, self.optimiser, loss_size, config, stats)

        print("finished Training")
        
    def load_model(self, config):
        model_path = os.path.join(config['checkpoint_dir'],config['checkpoint_filename'])

        modl = torch.load(model_path)

        self.model.load_state_dict(modl['model_state_dict'])
        self.optimiser.load_state_dict(modl['optimiser'])

        self.epoch = modl['epoch']

        self.loss_size = modl['loss']


            
        


