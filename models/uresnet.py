import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class UResNet(nn.Module):

    def __init__(self, args, data_stats):
        super(UResNet, self).__init__()
        
        self.args = args
        input_channels = data_stats["input_channels"]
        output_channels = data_stats["output_channels"]
        
        self.model = smp.Unet('resnet50', classes=output_channels,in_channels=input_channels, activation=None,encoder_weights="imagenet")
        
    """
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    """

    def forward(self, input):

        if type(input) is dict:
            x = input['x']
        else:
            x = input
        
        y = self.model(x)

        return {"y_pred": y,"y_pred_sigmoid": torch.sigmoid(y)}


    def get_loss(self, sample, output):


        if self.args.loss == "CE":
            criterion = torch.nn.BCEWithLogitsLoss()
        elif self.args.loss == "wCE":
            criterion = torch.nn.BCEWithLogitsLoss(weight=weight_cross_entropy_loss(sample["y"]))
        elif self.args.loss == "focal":
            criterion = smp.losses.FocalLoss("binary", alpha=1/15, gamma=2.0, reduction='mean')
        else:
            raise NotImplementedError
    
        loss = criterion(output["y_pred"], sample["y"])
        
        criterion_std = torch.nn.BCEWithLogitsLoss()
        std_loss = criterion_std(output["y_pred"].detach(), sample["y"]).detach()

        return loss, { "optimization_loss": loss.detach().item(),
                        "std_loss": std_loss.detach().item()}

def weight_cross_entropy_loss(target):
    # Calculate sum of weighted cross entropy loss.
    # Reference:
    # https://github.com/xwjabc/hed/blob/master/hed.py
    mask = (target > 0.5).float()
    b, c, h, w = mask.shape
    num_pos = torch.sum(mask)
    num_neg = b*c*h*w - num_pos
    weight = torch.zeros_like(target)
    neg_weight = num_pos / (num_pos + num_neg)
    pos_weight = num_neg / (num_pos + num_neg)
    weight[target > 0.5]  = pos_weight
    weight[target <= 0.5] = neg_weight
    return weight

