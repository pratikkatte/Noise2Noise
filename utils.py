from datetime import datetime
import numpy as np
import torch
import os
import json


class AvgMeter(object):
    """Computes and stores the average and current value.
    Useful for tracking averages such as elapsed times, minibatch losses, etc.
    """

    def __init__(self):
        self.reset()


    def reset(self):
        self.val = 0
        self.avg = 0.
        self.sum = 0
        self.count = 0


    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def time_elapsed_since(start):
    """Computes elapsed time since start."""

    timedelta = datetime.now() - start
    string = str(timedelta)[:-7]
    ms = int(timedelta.total_seconds() * 1000)

    return string, ms

def show_on_report(batch_idx, num_batches, loss, elapsed):
    """Formats training stats."""

    clear_line()
    dec = int(np.ceil(np.log10(num_batches)))
    print('Batch {:>{dec}d} / {:d} | Avg loss: {:>1.5f} | Avg train time / batch: {:d} ms'.format(batch_idx + 1, num_batches, loss, int(elapsed), dec=dec))
    
def psnr(source, target):
    return 10 * np.log10(1 / ((spurce - target)**2).mean())


def save_network(net, label, epoch,optimizer,loss_size, config, stats):
    filename = label
    save_path = os.path.join(config['checkpoint_dir'], filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimiser': optimizer.state_dict(),
        'loss':loss_size
    }, save_path)

    json_filename = '{}/n2nstate.json'.format(config['checkpoint_dir'])
    with open(json_filename, 'w') as fp:
        json.dump(stats, fp, indent=2)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('number of parameters: %d'% (num_params))
    

                
    