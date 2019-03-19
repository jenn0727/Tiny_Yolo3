import torch
import torch.nn as nn
from yolo_layer import YoloLayer
from util import *
import torch.nn.functional as F
from collections import OrderedDict
from collections import defaultdict


class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x

class MaxPoolStride1(nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x

class tiny_yolo(nn.Module):
    def __init__(self, config):
        super(tiny_yolo, self).__init__()

        self.config = config

        self.loss_names = ["x", "y", "w", "h", "conf", "cls", "recall", "precision"]
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0])

        self.conv_bn = [0, 4, 8, 12, 16, 20, 24, 27, 30, 36, 41]
        self.conv = [33,44]

        self.cnn = nn.Sequential(OrderedDict([
            # 0 conv 0-2
            ('conv0', nn.Conv2d(3, 16, 3, 1, 1, bias=False)),
            ('bn0', nn.BatchNorm2d(16)),
            ('leaky0', nn.LeakyReLU(0.1, inplace=True)),

            # 1 max 3
            ('max1', nn.MaxPool2d(2, 2)),

            # 2 conv 4-6
            ('conv2', nn.Conv2d(16, 32, 3, 1, 1, bias=False)),
            ('bn2', nn.BatchNorm2d(32)),
            ('leaky2', nn.LeakyReLU(0.1, inplace=True)),

            # 3 max 7
            ('pool3', nn.MaxPool2d(2, 2)),

            # 4 conv 8-10
            ('conv4', nn.Conv2d(32, 64, 3, 1, 1, bias=False)),
            ('bn4', nn.BatchNorm2d(64)),
            ('leaky4', nn.LeakyReLU(0.1, inplace=True)),

            # 5 max 11
            ('pool5', nn.MaxPool2d(2, 2)),

            # 6 conv 12-14
            ('conv6', nn.Conv2d(64, 128, 3, 1, 1, bias=False)),
            ('bn6', nn.BatchNorm2d(128)),
            ('leaky6', nn.LeakyReLU(0.1, inplace=True)),

            # 7 max 15
            ('pool7', nn.MaxPool2d(2, 2)),

            # 8 conv 16-18
            ('conv8', nn.Conv2d(128, 256, 3, 1, 1, bias=False)),
            ('bn8', nn.BatchNorm2d(256)),
            ('leaky8', nn.LeakyReLU(0.1, inplace=True)),

            # 9 max 19
            ('pool9', nn.MaxPool2d(2, 2)),

            # 10 conv 20-22
            ('conv10', nn.Conv2d(256, 512, 3, 1, 1, bias=False)),
            ('bn10', nn.BatchNorm2d(512)),
            ('leaky10', nn.LeakyReLU(0.1, inplace=True)),

            # 11 max 23
            ('pool11', MaxPoolStride1()),

            # 12 conv 24-26
            ('conv12', nn.Conv2d(512, 1024, 3, 1, 1, bias=False)),
            ('bn12', nn.BatchNorm2d(1024)),
            ('leaky12', nn.LeakyReLU(0.1, inplace=True)),

            # 13 conv 27-29
            ('conv13', nn.Conv2d(1024, 256, 1, 1, 0, bias=False)),
            ('bn13', nn.BatchNorm2d(256)),
            ('leaky13', nn.LeakyReLU(0.1, inplace=True)),

            # 14 conv 30-32
            ('conv14', nn.Conv2d(256, 512, 3, 1, 1, bias=False)),
            ('bn14', nn.BatchNorm2d(512)),
            ('leaky14', nn.LeakyReLU(0.1, inplace=True)),

            # 15 conv 33
            ('conv15', nn.Conv2d(512, 255, kernel_size=1, stride=1, padding=0)),

            # 16 yolo 34
            ('yolo16', YoloLayer([3, 4, 5], self.config)),

            # 17 route 35
            ('route17', EmptyModule()),

            # 18 conv 36-38
            ('conv18', nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)),
            ('bn18', nn.BatchNorm2d(128)),
            ('leaky18', nn.LeakyReLU(0.1, inplace=True)),

            # 19 upsample 39
            ('upsample', nn.Upsample(scale_factor=2)),

            # 20 route 40
            ('route20', EmptyModule()),

            # 21 conv  41-43
            ('conv21', nn.Conv2d(384, 256, 3, 1, 1, bias=False)),
            ('bn21', nn.BatchNorm2d(256)),
            ('leaky21', nn.LeakyReLU(0.1, inplace=True)),

            # 22 conv 44
            ('conv22', nn.Conv2d(256, 255, kernel_size=1, stride=1, padding=0)),

            # 23 yolo 45
            ('yolo23', YoloLayer([0, 1, 2], self.config)),

        ]))


    """
    def Conv_BN_Leaky(self, in_channel, out_channel, kernel_size, padding, bias=False):
        conv_bn_leaky = nn.Sequential(
                          nn.Conv2d(in_channel, out_channel, kernel_size, padding, bias),
                          nn.BatchNorm2d(out_channel),
                          nn.LeakyReLU(0.1, inplace=True)
                        )
        return conv_bn_leaky
        """

    def forward(self, x, targets =None):

        self.losses = defaultdict(float)
        out_boxes = []
        output= []
        for i in range(19):
            x = self.cnn[i](x)
        x1 = x
        # x1:26*26*256

        for i in range(19,30):
            x= self.cnn[i](x)
        x2 = x

        # x2:13*13*256

        for i in range(30,34):
            x = self.cnn[i](x)

        y1 = x

        for i in range(36,40):
            x2 = self.cnn[i](x2)

        # x2:26*26*128

        #20 route 40th
        x = torch.cat((x2,x1), 1)
        # x:26*26*384


        for i in range(41,45):
            x = self.cnn[i](x)

        y2 = x


        if self.config.is_train:
            x, *losses = self.cnn[34](y1, targets)
            for name, loss in zip(self.loss_names, losses):
                self.losses[name] += loss
            output.append(x)

            x, *losses = self.cnn[45](y2, targets)
            for name, loss in zip(self.loss_names, losses):
                self.losses[name] += loss
            output.append(x)


        else:
            boxes = self.yolo_layer1(y1,targets)
            out_boxes.append(boxes)
            boxes = self.yolo_layer1(y2,targets)
            out_boxes.append(boxes)



        self.losses["recall"] /= 3
        self.losses["precision"] /= 3

        return sum(output) if self.config.is_train else torch.cat(out_boxes,1)

    def load_weights(self, weightfile):

        # Open the weights file
        fp = open(weightfile, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

        # Needed to write header when saving weights
        self.header_info = header

        self.seen = header[3]
        buf = np.fromfile(fp, dtype=np.float32)  # The rest are weights
        fp.close()
        start = 0
        """
        for i in self.conv_bn[0:-2]:

            start = load_conv_bn(buf, start, self.cnn[i], self.cnn[i+1])
            print(i)

        """
        start = load_conv_bn(buf, start, self.cnn[0], self.cnn[1])
        start = load_conv_bn(buf, start, self.cnn[4], self.cnn[5])
        start = load_conv_bn(buf, start, self.cnn[8], self.cnn[9])
        start = load_conv_bn(buf, start, self.cnn[12], self.cnn[13])
        start = load_conv_bn(buf, start, self.cnn[16], self.cnn[17])
        start = load_conv_bn(buf, start, self.cnn[20], self.cnn[21])

        start = load_conv_bn(buf, start, self.cnn[24], self.cnn[25])
        start = load_conv_bn(buf, start, self.cnn[27], self.cnn[28])
        start = load_conv_bn(buf, start, self.cnn[30], self.cnn[31])


        start = load_conv(buf, start, self.cnn[33])

        start = load_conv_bn(buf, start, self.cnn[36], self.cnn[37])
        start = load_conv_bn(buf, start, self.cnn[41], self.cnn[42])

        start = load_conv(buf, start, self.cnn[44])


    def save_weights(self, outfile):

        fp = open(outfile, 'wb')
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        for i in range(len(self.cnn)):
            if i in self.conv_bn:
                save_conv_bn(fp, self.cnn[i], self.cnn[i+1])
            if i in self.conv:
                save_conv(fp, self.cnn[i])

        fp.close()

