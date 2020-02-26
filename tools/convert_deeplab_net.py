# Convert a backbone trained with pytorch-deeplab-xception to detectron2 format
import torch
import re
import pickle

# The method wraps a deeplab save file with a few extra variables so that it can be loaded by the detectron2 checkpointer
def wrap_pytorch_resnet(filepath):
    if torch.cuda.is_available():
        network = torch.load(filepath, map_location=torch.device('cuda'))
    else:
        network = torch.load(filepath, map_location=torch.device('cpu'))

    new_resnet = {k.replace('backbone', 'backbone.model'):v for k,v in network['state_dict'].items()}
    return {"__author__": "deeplab",
            "model": new_resnet}

# This method changes the names of the deeplab layers to match the format of a saved detectron2 resnet with dilation
# For best results use the DC5_3x config file
# TODO Issues: The weights match between the two models but stride length differs leading to different results
def convert_pytorch_resnet(filepath):
    if torch.cuda.is_available():
        network = torch.load(filepath, map_location=torch.device('cuda'))
    else:
        network = torch.load(filepath, map_location=torch.device('cpu'))

    resnet = {k[9:]:v for k, v in network['state_dict'].items() if k.startswith('backbone')}

    p = re.compile('layer([0-9])\.([0-9]+)\.(\w+)([\.0-3]+)\.(\w+)')

    new_resnet = {}
    for k, v in resnet.items():

        # First couple layers don't fit rules
        if k == "conv1.weight":
            new_resnet['conv1_w'] = v
            continue
        elif k == "bn1.running_mean":
            new_resnet['res_conv1_bn_running_mean'] = v
            continue
        elif k == "bn1.running_var":
            new_resnet['res_conv1_bn_running_var'] = v
            continue
        elif k == "bn1.weight":
            new_resnet['res_conv1_bn_gamma'] = v
            continue
        elif k == "bn1.bias":
            new_resnet['res_conv1_bn_beta'] = v
            continue
        elif k == "bn1.num_batches_tracked":
            continue
        elif k in ['fc.weight', 'fc.bias']:
            continue

        # Rules start here
        match = re.match(p, k)
        if match:
            index1 = match.group(1)
            index2 = match.group(2)
            layer = match.group(3)
            branch = match.group(4)
            variable = match.group(5)

            if layer == 'downsample':
                b = '1'
            elif branch == '1':
                b = '2a'
            elif branch == '2':
                b = '2b'
            elif branch == '3':
                b = '2c'
            else:
                raise ValueError(k)

            if layer == 'conv':
                variable = 'w'
            elif layer == 'downsample' and branch == '.0':
                variable = 'w'
            elif layer in ['bn', 'downsample'] and variable == 'weight':
                variable = 'bn_gamma'
            elif layer in ['bn', 'downsample'] and variable == 'bias':
                variable = 'bn_beta'
            elif variable == 'num_batches_tracked':
                continue
            else:
                variable = 'bn_{}'.format(variable)

            new_k = 'res{}_{}_branch{}_{}'.format(int(index1) + 1, index2, b, variable)

            new_resnet[new_k] = v.cpu().numpy()
        else:
            raise ValueError(k)

    return new_resnet

# Test loads both the original deeplab model and the converted detectron model and checks both structural equivalence and output
def test_equivalence(cfg, infile, outfile):
    from deeplab3.modeling.backbone.resnet import ResNet101
    from detectron2.checkpoint.detection_checkpoint import  DetectionCheckpointer
    from detectron2.engine import DefaultTrainer
    import torch.nn as nn

    #Deeplab
    model = ResNet101(cfg.MODEL.DEEPLAB, BatchNorm=nn.BatchNorm2d)
    model.eval()

    if torch.cuda.is_available():
        checkpoint = torch.load(infile, map_location=torch.device('cuda'))
    else:
        checkpoint = torch.load(infile, map_location=torch.device('cpu'))

    state_dict = {k.replace('backbone.', ''):v for k, v in checkpoint['state_dict'].items() if 'backbone' in k}
    model.load_state_dict(state_dict)

    #Detectron2
    model2 = DefaultTrainer.build_model(cfg)
    model2.eval()
    DetectionCheckpointer(model2, save_dir=outfile).resume_or_load(
        cfg.MODEL.WEIGHTS)

    #Check that the layers are identical
    convs = []
    downsample = []
    bn = []
    downsample_bn = []
    for name, param in model.state_dict().items():
        param_dict = {'name': name, 'param': param, 'size': param.size()}
        if 'conv' in name:
            convs.append(param_dict)
        elif 'downsample.0' in name:
            downsample.append(param_dict)
        elif 'bn' in name:
            bn.append(param_dict)
        elif 'downsample.1' in name:
            downsample_bn.append(param_dict)
        else:
            print(name)

    convs2 = []
    shortcut = []
    bn2 = []
    downsample_bn2 = []
    for name, param in model2.backbone.state_dict().items():
        param_dict = {'name': name, 'param': param, 'size': param.size()}
        if 'conv' in name:
            convs.append(param_dict)
        elif 'downsample.0' in name:
            downsample.append(param_dict)
        elif 'bn' in name:
            bn.append(param_dict)
        elif 'downsample.1' in name:
            downsample_bn.append(param_dict)
        else:
            print(name)

    for p1, p2 in zip(convs, convs2):
        print(p1['name'], p2['name'])
        if(p1['size'] != p2['size']):
            print(p1['size'], p2['size'])
            raise ValueError('Sizes not equal')
        assert(torch.all(torch.eq(p1['param'], p2['param'])))

    for p1, p2 in zip(shortcut, downsample):
        print(p1['name'], p2['name'])
        if(p1['size'] != p2['size']):
            print(p1['size'], p2['size'])
            raise ValueError('Sizes not equal')
        assert(torch.all(torch.eq(p1['param'], p2['param'])))

    for p1, p2 in zip(bn, bn2):
        print(p1['name'], p2['name'])
        if(p1['size'] != p2['size']):
            print(p1['size'], p2['size'])
            raise ValueError('Sizes not equal')
        assert(torch.all(torch.eq(p1['param'], p2['param'])))

    for p1, p2 in zip(downsample_bn2, downsample_bn):
        print(p1['name'], p2['name'])
        if (p1['size'] != p2['size']):
            print(p1['size'], p2['size'])
            raise ValueError('Sizes not equal')
        assert (torch.all(torch.eq(p1['param'], p2['param'])))

    #Check that output is identical
    input = torch.ones((1, 4, 640, 640), dtype=torch.float32)

    output = model(input)
    output2 = model2.backbone(input)

    assert(torch.all(torch.eq(output['res5'], output2['res5'])))
    assert (torch.all(torch.eq(output['res2'], output2['res2'])))

    print("Passed all tests")


if __name__ == "__main__":
    # Arguments: infile : where the deeplab network is saved (with torch.save)
    #            outfile : where the detectron network should be saved (with pickle.dump)

    import sys
    from detectron2.config import get_cfg

    if(len(sys.argv)!=4):
        print("Program requires three arguments: infile outfile config_file")
        exit(1)

    infile = sys.argv[1]
    outfile = sys.argv[2]
    config_file = sys.argv[3]

    backbone = wrap_pytorch_resnet(infile)

    with open(outfile, 'wb') as f:
        pickle.dump(backbone, f)

    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(['MODEL.DEVICE', 'cpu', 'MODEL.WEIGHTS', outfile])
    cfg.freeze()

    test_equivalence(cfg, infile, outfile)
