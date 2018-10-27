import torch.nn as nn
import torch.nn.functional as F
import torch

class AutoEncoder(nn.Module):
    
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.conv1 = nn.Conv2d(3,48,3,padding=1)

        self.conv2 = nn.Conv2d(48,48,3,padding=1)

        self.pool = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(48,48,3,padding=1)

        self.conv4 = nn.Conv2d(48,48,3,padding=1)


        self.conv5 = nn.Conv2d(48,48,3,padding=1)

        self.conv6 = nn.Conv2d(48,48,3,padding=1)


        self.upscale = nn.Upsample(scale_factor=2)

        self.decov1 = nn.Conv2d(96,96,3,padding=1)


        self.decov2 = nn.Conv2d(96,96,3,padding=1)

        self.decov3 = nn.Conv2d(144,96,3,padding=1)


        self.decov4 = nn.Conv2d(96,96,3,padding=1)


        self.decov5 = nn.Conv2d(144,96,3,padding=1)

        self.decov6 = nn.Conv2d(96,96,3,padding=1)


        self.decov7 = nn.Conv2d(144,96,3,padding=1)

        self.decov8 = nn.Conv2d(96,96,3,padding=1)


        self.decov9 = nn.Conv2d(99,64,3,padding=1)

        self.decov10 = nn.Conv2d(64,32,3,padding=1)

        self.conv7 = nn.Conv2d(32,3,3,padding=1)
        
    def forward(self, x):
        
        skips = [x]
        
        # Enc. Conv0
        x = F.leaky_relu(self.conv1(x))
        
        # Enc. Conv1
        x = F.leaky_relu(self.conv2(x))
        
        #pool1
        x = self.pool(x)
        skips.append(x)
        
        # Enc. Conv2 and p00l2
        x = self.pool(F.leaky_relu(self.conv3(x)))
        
        skips.append(x)
        
        # Enc. Conv3 and p00l3
        x = self.pool(F.leaky_relu(self.conv4(x)))
        
        skips.append(x)
        
        # Enc. Conv4 and pool4
        x = self.pool(F.leaky_relu(self.conv5(x)))
        
        skips.append(x)
        
        # Enc. Conv5 and pool5
        x = self.pool(F.leaky_relu(self.conv6(x)))
        
        # Enc. Conv6
        x = F.leaky_relu(self.conv6(x))
        
        #---------#
        #upsample 5
        x = self.upscale(x)
        
        # concat 5
        x = torch.cat([x, skips.pop()],1)        
        
        x = F.leaky_relu(self.decov1(x))
        
        x = self.upscale(F.leaky_relu_(self.decov2(x)))

        x = torch.cat([x, skips.pop()],1)
        
        x = F.leaky_relu(self.decov3(x))
        x = self.upscale(F.leaky_relu(self.decov4(x)))
        
        # concat 3
        x = torch.cat([x, skips.pop()],1)
        x = F.leaky_relu(self.decov5(x))
        
        x = self.upscale(F.leaky_relu(self.decov6(x)))
        # concat2
        x = torch.cat([x, skips.pop()],1)
        x = F.leaky_relu(self.decov7(x))
        x = self.upscale(F.leaky_relu(self.decov8(x)))
        
        #concat 1
        x = torch.cat([x, skips.pop()],1)
        x = F.leaky_relu(self.decov9(x))
        x = F.leaky_relu(self.decov10(x))
        
        
        x = self.conv7(x)
        
        return x