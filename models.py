from torch import nn
import torch
from torch.nn import functional as F


class CrossConvNet(nn.Module):
    def __init__(self, n_classes, depth_model, ir_model, rgb_model, mode='depth_ir_rgb', cross_mode=None):
        super(CrossConvNet, self).__init__()

        self.n_classes = n_classes


        self.depth_model = depth_model
        self.ir_model = ir_model
        self.rgb_model = rgb_model
        self.mode = mode
        self.cross_mode = cross_mode



        #modifiche ultimop layer reti
        # self.model_depth_ir.classifier = nn.Linear(in_features=2208, out_features=128)
        # self.model_rgb.classifier = nn.Linear(in_features=2208, out_features=128)
        # self.model_lstm.fc = nn.Linear(in_features=512, out_features=128)
        self.conv = nn.Conv2d(in_channels=2208 * 3 if self.mode == 'depth_ir_rgb' else 2208 * 2
                              , out_channels=512, kernel_size=3)
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3)
        self.fc = nn.Linear(in_features=1152, out_features=256)
        self.fc1 = nn.Linear(in_features=256, out_features=self.n_classes)

    def forward(self, *input):


        out_depth, out_ir, out_rgb, x_depth, x_ir, x_rgb = None, None, None, None, None, None
        xs = [value for value in input]

        if self.mode == 'depth_ir_rgb':
            x_depth = xs[0]
            x_ir = xs[1]
            x_rgb = xs[2]
        elif self.mode == 'depth_ir':
            x_depth = xs[0]
            x_ir = xs[1]
        elif self.mode == 'depth_rgb':
            x_depth = xs[0]
            x_rgb = xs[1]
        elif self.mode == 'ir_rgb':
            x_ir = xs[0]
            x_rgb = xs[1]

        with torch.no_grad():
            if x_depth is not None:
                out_depth = self.depth_model(x_depth)

            if x_ir is not None:
                out_ir = self.ir_model(x_ir)

            if x_rgb is not None:
                out_rgb = self.rgb_model(x_rgb)

        # concateno le  3 uscite
        if out_depth is not None and out_ir is not None and out_rgb is not None:
            if self.cross_mode == 'avg':
                x = (out_depth + out_ir + out_rgb)/3
            else:
                x = torch.cat((out_depth, out_ir, out_rgb), 1)
        # concateno gli out
        elif out_depth is not None and out_ir is not None:
            if self.cross_mode == 'avg':
                x = (out_depth + out_ir)/2
            else:
                x = torch.cat((out_depth, out_ir), 1)
        elif out_depth is not None and out_rgb is not None:
            if self.cross_mode == 'avg':
                x = (out_depth + out_rgb)/2
            else:
                x = torch.cat((out_depth, out_rgb), 1)
        elif out_ir is not None and out_rgb is not None:
            if self.cross_mode == 'avg':
                x = (out_ir + out_rgb)/2
            else:
                x = torch.cat((out_ir, out_rgb), 1)

        if self.cross_mode == 'avg':
            out = x
        else:
            x = F.relu(self.conv(x))
            x = F.dropout2d(x, p=0.2)
            x = F.relu(self.conv1(x))
            x = F.dropout2d(x, p=0.2)

            x = x.view(-1, self.num_flat_features(x))

            x = F.dropout(self.fc(x), p=0.2)
            out = self.fc1(x)

        return out

    def num_flat_features(self, x):
        return torch.prod(torch.tensor(x.size()[1:]))
