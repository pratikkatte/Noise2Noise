import torch
import numpy as np
from datetime import datetime
from utils import *
from network import *
from dataset import *
import argparse
from summary import viewSummary

class Noise2Noise(object):
    
    def __init__(self, config, trainloader, validloader, trainable):
        
        self.model  = UNet().to(config['device'])
        
        self.trainable = trainable

        if self.trainable:
            self.train_loader = trainloader.load_data()
            self.valid_loader = validloader.load_data()
            self.loss, self.optimiser, self.schedular = lossOptimiser(self.model, config)
            
        self.summary = viewSummary(config['log_dir'])

    def eval(self, config):

        self.model.train(False)
        
        valid_start = datetime.now()
        loss_meter = AvgMeter()
        psnr_meter = AvgMeter()
    
        for _, data in enumerate(self.valid_loader):
        
            inputs, labels = data
        
            inputs = inputs.float()
            labels = inputs.float()
        
            inputs = inputs.to(config['device'])
            labels = labels.to(config['device'])
        
            outputs = self.model(inputs)
        
            loss_size = self.loss(outputs, labels)
            loss_meter.update(loss_size.item())
        
        for i in range(config['batch_size']):
            labels = labels.cpu()
            outputs = outputs.cpu()
            psnr_meter.update(psnr(outputs[i], labels[i]).item())
        
        valid_loss = loss_meter.avg
        valid_time = time_elapsed_since(valid_start)[0]
        psnr_avg = psnr_meter.avg
        
        return valid_loss, valid_time, psnr_avg


    def train(self, config):

        self.model.train(True)

        stats = {
        'train_loss': [],
        'valid_loss': [],
        'valid_psnr': []
        }

        for epoch in range(config['num_epochs']):

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
                inputs, labels = data

                inputs = inputs.float()
                labels = labels.float()
                inputs = inputs.to(config['device'])
                labels = labels.to(config['device'])

                outputs = self.model(inputs)

                loss_size = self.loss(outputs, labels)
                loss_meter.update(loss_size.item())

                self.optimiser.zero_grad()
                loss_size.backward()
                self.optimiser.step()

                time_meter.update(time_elapsed_since(batch_start)[1])

                if total_steps % 10 == 0:
                    running_loss += loss_size.data[0]

                    t = (datetime.now() - iter_time_start) / config['batch_size']
                    print("Epoch: {}, total_steps: {:d}, train_loss: {}, time: {}".format(epoch, total_steps, running_loss, t))

                    train_loss_meter.update(loss_meter.avg)

                    loss_meter.reset()
                    time_meter.reset()


            print('validating and saving the model at the end of epoch %d \t Time Taken: %d sec' %
                  (epoch, (datetime.now() - epoch_start_time).total_seconds()))

            valid_loss, valid_time, valid_psnr = self.eval(config)
            
            self.schedular.step(valid_loss)

            stats['train_loss'].append(train_loss_meter.avg)
            stats['valid_loss'].append(valid_loss)
            stats['valid_psnr'].append(valid_psnr)
            
            self.summary.view_stats(stats, epoch)
            
            self.summary.view_image({'source_images':inputs.view(-1,3,256,256)[:2].cpu().numpy(),
                                     'target_images':labels.view(-1,3,256,256)[:2].cpu().numpy() }, epoch)
            
            
            save_network(self.model,'latest.pt', epoch, self.optimiser, loss_size, config, stats)

        print("finished Training")


