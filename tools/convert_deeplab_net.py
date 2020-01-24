# Convert a backbone trained with pytorch-deeplab-xception to detectron2 format
import torch
import re

def convert_resnet(filepath):
    backbone = {k:v for k, v in torch.load(filepath)['state_dict'].items() if k.startswith('backbone')}
    p = re.compile('backbone\.layer([0-9])\.([0-9]).\w+([1-3]).(\w+)')

    new_backbone = {}
    for k, v in backbone.items():

        #First couple layers don't fit rules
        if k == "backbone.conv1.weight":
            new_backbone['conv1_w'] = v
            continue
        elif k == "backbone.bn1.running_mean":
            new_backbone['res_conv1_bn_running_mean']= v
            continue
        elif k == "backbone.bn1.running_var":
            new_backbone['res_conv1_bn_running_var']= v
            continue
        elif k == "backbone.bn1.weights":
            new_backbone['res_conv1_bn_gamma']= v
            continue
        elif k == "backbone.bn1.bias":
            new_backbone['res_conv1_bn_beta']= v
            continue

        #Rules start here
        match = re.match(p, k)
        if match:
            index1 = match.group(0)
            index2 = match.group(1)
            layer = match.group(2)
            branch = match.group(3)
            variable = match.group(4)

            if layer == 'downsample':
                b = '2c'
            elif branch == 1:
                b = 1
            elif branch == 2:
                b = '2a'
            elif branch == 2:
                b = '2b'
            else:
                raise ValueError(k)

            if layer == 'conv':
                variable = 'w'
            elif layer == 'downsample' and branch == 0:
                variable = 'w'
            elif layer in ['bn', 'downsample'] and variable == 'weight':
                variable = 'gamma'
            elif layer in ['bn', 'downsample'] and variable == 'bias':
                variable = 'beta'
            elif variable == 'num_batches_tracked':
                continue

            new_k = 'res_{}_{}_branch{}_{}'.format(index1+1, index2, b, variable)

            new_backbone[new_k] = v
        else:
            raise ValueError(k)

    return new_backbone