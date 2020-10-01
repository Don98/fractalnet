from torch import from_numpy
import torch
import pickle as pkl
def load_dict():
    with open('weights1.pkl', 'rb') as wp:
        name_weights = pkl.load(wp)
    state_dict = {}
    state_dict['convH_0.weight'] = torch.from_numpy(name_weights['convH_0']['weight'])

    state_dict['bn1.running_var'] = from_numpy(name_weights['batch_convH_0']['running_var'])
    state_dict['bn1.running_mean'] = from_numpy(name_weights['batch_convH_0']['running_mean'])
    state_dict['bn1.weight'] = torch.ones_like(state_dict['bn1' + '.running_var'])
    state_dict['bn1.bias'] = torch.zeros_like(state_dict['bn1' + '.running_var'])

    state_dict['the_block1.0.conv0_0.conv1.weight'] = torch.from_numpy(name_weights['conv0_0']['weight'])

    state_dict['the_block1.0.conv0_0.conv1.bias'] = torch.from_numpy(name_weights['conv0_0']['bias'])
    state_dict['the_block1.0.conv0_0.bn1.running_var'] = from_numpy(name_weights['batch_conv0_0']['running_var'])
    state_dict['the_block1.0.conv0_0.bn1.running_mean'] = from_numpy(name_weights['batch_conv0_0']['running_mean'])
    state_dict['the_block1.0.conv0_0.bn1.weight'] = torch.ones_like(state_dict['the_block1.0.conv0_0.bn1.running_var'])
    state_dict['the_block1.0.conv0_0.bn1.bias'] = torch.ones_like(state_dict['the_block1.0.conv0_0.bn1.running_var'])

    state_dict['the_block1.0.conv2_0.conv1.weight'] = torch.from_numpy(name_weights['conv2_0']['weight'])

    state_dict['the_block1.0.conv2_0.conv1.bias'] = torch.from_numpy(name_weights['conv2_0']['bias'])
    state_dict['the_block1.0.conv2_0.bn1.running_var'] = from_numpy(name_weights['batch_conv2_0']['running_var'])
    state_dict['the_block1.0.conv2_0.bn1.running_mean'] = from_numpy(name_weights['batch_conv2_0']['running_mean'])
    state_dict['the_block1.0.conv2_0.bn1.weight'] = torch.ones_like(state_dict['the_block1.0.conv2_0.bn1.running_var'])
    state_dict['the_block1.0.conv2_0.bn1.bias'] = torch.ones_like(state_dict['the_block1.0.conv2_0.bn1.running_var'])

    state_dict['the_block1.0.conv3_0.conv1.weight'] = torch.from_numpy(name_weights['conv3_0']['weight'])

    state_dict['the_block1.0.conv3_0.conv1.bias'] = torch.from_numpy(name_weights['conv3_0']['bias'])
    state_dict['the_block1.0.conv3_0.bn1.running_var'] = from_numpy(name_weights['batch_conv3_0']['running_var'])
    state_dict['the_block1.0.conv3_0.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_0']['running_mean'])
    state_dict['the_block1.0.conv3_0.bn1.weight'] = torch.ones_like(state_dict['the_block1.0.conv3_0.bn1.running_var'])
    state_dict['the_block1.0.conv3_0.bn1.bias'] = torch.ones_like(state_dict['the_block1.0.conv3_0.bn1.running_var'])

    state_dict['the_block1.0.conv3_1.conv1.weight'] = torch.from_numpy(name_weights['conv3_1']['weight'])

    state_dict['the_block1.0.conv3_1.conv1.bias'] = torch.from_numpy(name_weights['conv3_1']['bias'])
    state_dict['the_block1.0.conv3_1.bn1.running_var'] = from_numpy(name_weights['batch_conv3_1']['running_var'])
    state_dict['the_block1.0.conv3_1.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_1']['running_mean'])
    state_dict['the_block1.0.conv3_1.bn1.weight'] = torch.ones_like(state_dict['the_block1.0.conv3_1.bn1.running_var'])
    state_dict['the_block1.0.conv3_1.bn1.bias'] = torch.ones_like(state_dict['the_block1.0.conv3_1.bn1.running_var'])

    state_dict['the_block1.0.conv2_1.conv1.weight'] = torch.from_numpy(name_weights['conv2_1']['weight'])

    state_dict['the_block1.0.conv2_1.conv1.bias'] = torch.from_numpy(name_weights['conv2_1']['bias'])
    state_dict['the_block1.0.conv2_1.bn1.running_var'] = from_numpy(name_weights['batch_conv2_1']['running_var'])
    state_dict['the_block1.0.conv2_1.bn1.running_mean'] = from_numpy(name_weights['batch_conv2_1']['running_mean'])
    state_dict['the_block1.0.conv2_1.bn1.weight'] = torch.ones_like(state_dict['the_block1.0.conv2_1.bn1.running_var'])
    state_dict['the_block1.0.conv2_1.bn1.bias'] = torch.ones_like(state_dict['the_block1.0.conv2_1.bn1.running_var'])

    state_dict['the_block1.0.conv3_2.conv1.weight'] = torch.from_numpy(name_weights['conv3_2']['weight'])

    state_dict['the_block1.0.conv3_2.conv1.bias'] = torch.from_numpy(name_weights['conv3_2']['bias'])
    state_dict['the_block1.0.conv3_2.bn1.running_var'] = from_numpy(name_weights['batch_conv3_2']['running_var'])
    state_dict['the_block1.0.conv3_2.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_2']['running_mean'])
    state_dict['the_block1.0.conv3_2.bn1.weight'] = torch.ones_like(state_dict['the_block1.0.conv3_2.bn1.running_var'])
    state_dict['the_block1.0.conv3_2.bn1.bias'] = torch.ones_like(state_dict['the_block1.0.conv3_2.bn1.running_var'])

    state_dict['the_block1.0.conv3_3.conv1.weight'] = torch.from_numpy(name_weights['conv3_3']['weight'])

    state_dict['the_block1.0.conv3_3.conv1.bias'] = torch.from_numpy(name_weights['conv3_3']['bias'])
    state_dict['the_block1.0.conv3_3.bn1.running_var'] = from_numpy(name_weights['batch_conv3_3']['running_var'])
    state_dict['the_block1.0.conv3_3.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_3']['running_mean'])
    state_dict['the_block1.0.conv3_3.bn1.weight'] = torch.ones_like(state_dict['the_block1.0.conv3_3.bn1.running_var'])
    state_dict['the_block1.0.conv3_3.bn1.bias'] = torch.ones_like(state_dict['the_block1.0.conv3_3.bn1.running_var'])

    state_dict['the_block1.0.conv1_0.conv1.weight'] = torch.from_numpy(name_weights['conv1_0']['weight'])

    state_dict['the_block1.0.conv1_0.conv1.bias'] = torch.from_numpy(name_weights['conv1_0']['bias'])
    state_dict['the_block1.0.conv1_0.bn1.running_var'] = from_numpy(name_weights['batch_conv1_0']['running_var'])
    state_dict['the_block1.0.conv1_0.bn1.running_mean'] = from_numpy(name_weights['batch_conv1_0']['running_mean'])
    state_dict['the_block1.0.conv1_0.bn1.weight'] = torch.ones_like(state_dict['the_block1.0.conv1_0.bn1.running_var'])
    state_dict['the_block1.0.conv1_0.bn1.bias'] = torch.ones_like(state_dict['the_block1.0.conv1_0.bn1.running_var'])

    state_dict['the_block1.0.conv2_2.conv1.weight'] = torch.from_numpy(name_weights['conv2_2']['weight'])

    state_dict['the_block1.0.conv2_2.conv1.bias'] = torch.from_numpy(name_weights['conv2_2']['bias'])
    state_dict['the_block1.0.conv2_2.bn1.running_var'] = from_numpy(name_weights['batch_conv2_2']['running_var'])
    state_dict['the_block1.0.conv2_2.bn1.running_mean'] = from_numpy(name_weights['batch_conv2_2']['running_mean'])
    state_dict['the_block1.0.conv2_2.bn1.weight'] = torch.ones_like(state_dict['the_block1.0.conv2_2.bn1.running_var'])
    state_dict['the_block1.0.conv2_2.bn1.bias'] = torch.ones_like(state_dict['the_block1.0.conv2_2.bn1.running_var'])

    state_dict['the_block1.0.conv3_4.conv1.weight'] = torch.from_numpy(name_weights['conv3_4']['weight'])

    state_dict['the_block1.0.conv3_4.conv1.bias'] = torch.from_numpy(name_weights['conv3_4']['bias'])
    state_dict['the_block1.0.conv3_4.bn1.running_var'] = from_numpy(name_weights['batch_conv3_4']['running_var'])
    state_dict['the_block1.0.conv3_4.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_4']['running_mean'])
    state_dict['the_block1.0.conv3_4.bn1.weight'] = torch.ones_like(state_dict['the_block1.0.conv3_4.bn1.running_var'])
    state_dict['the_block1.0.conv3_4.bn1.bias'] = torch.ones_like(state_dict['the_block1.0.conv3_4.bn1.running_var'])

    state_dict['the_block1.0.conv3_5.conv1.weight'] = torch.from_numpy(name_weights['conv3_5']['weight'])

    state_dict['the_block1.0.conv3_5.conv1.bias'] = torch.from_numpy(name_weights['conv3_5']['bias'])
    state_dict['the_block1.0.conv3_5.bn1.running_var'] = from_numpy(name_weights['batch_conv3_5']['running_var'])
    state_dict['the_block1.0.conv3_5.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_5']['running_mean'])
    state_dict['the_block1.0.conv3_5.bn1.weight'] = torch.ones_like(state_dict['the_block1.0.conv3_5.bn1.running_var'])
    state_dict['the_block1.0.conv3_5.bn1.bias'] = torch.ones_like(state_dict['the_block1.0.conv3_5.bn1.running_var'])

    state_dict['the_block1.0.conv1_1.conv1.weight'] = torch.from_numpy(name_weights['conv1_1']['weight'])

    state_dict['the_block1.0.conv1_1.conv1.bias'] = torch.from_numpy(name_weights['conv1_1']['bias'])
    state_dict['the_block1.0.conv1_1.bn1.running_var'] = from_numpy(name_weights['batch_conv1_1']['running_var'])
    state_dict['the_block1.0.conv1_1.bn1.running_mean'] = from_numpy(name_weights['batch_conv1_1']['running_mean'])
    state_dict['the_block1.0.conv1_1.bn1.weight'] = torch.ones_like(state_dict['the_block1.0.conv1_1.bn1.running_var'])
    state_dict['the_block1.0.conv1_1.bn1.bias'] = torch.ones_like(state_dict['the_block1.0.conv1_1.bn1.running_var'])

    state_dict['the_block1.0.conv2_3.conv1.weight'] = torch.from_numpy(name_weights['conv2_3']['weight'])

    state_dict['the_block1.0.conv2_3.conv1.bias'] = torch.from_numpy(name_weights['conv2_3']['bias'])
    state_dict['the_block1.0.conv2_3.bn1.running_var'] = from_numpy(name_weights['batch_conv2_3']['running_var'])
    state_dict['the_block1.0.conv2_3.bn1.running_mean'] = from_numpy(name_weights['batch_conv2_3']['running_mean'])
    state_dict['the_block1.0.conv2_3.bn1.weight'] = torch.ones_like(state_dict['the_block1.0.conv2_3.bn1.running_var'])
    state_dict['the_block1.0.conv2_3.bn1.bias'] = torch.ones_like(state_dict['the_block1.0.conv2_3.bn1.running_var'])

    state_dict['the_block1.0.conv3_6.conv1.weight'] = torch.from_numpy(name_weights['conv3_6']['weight'])

    state_dict['the_block1.0.conv3_6.conv1.bias'] = torch.from_numpy(name_weights['conv3_6']['bias'])
    state_dict['the_block1.0.conv3_6.bn1.running_var'] = from_numpy(name_weights['batch_conv3_6']['running_var'])
    state_dict['the_block1.0.conv3_6.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_6']['running_mean'])
    state_dict['the_block1.0.conv3_6.bn1.weight'] = torch.ones_like(state_dict['the_block1.0.conv3_6.bn1.running_var'])
    state_dict['the_block1.0.conv3_6.bn1.bias'] = torch.ones_like(state_dict['the_block1.0.conv3_6.bn1.running_var'])

    state_dict['the_block1.0.conv3_7.conv1.weight'] = torch.from_numpy(name_weights['conv3_7']['weight'])

    state_dict['the_block1.0.conv3_7.conv1.bias'] = torch.from_numpy(name_weights['conv3_7']['bias'])
    state_dict['the_block1.0.conv3_7.bn1.running_var'] = from_numpy(name_weights['batch_conv3_7']['running_var'])
    state_dict['the_block1.0.conv3_7.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_7']['running_mean'])
    state_dict['the_block1.0.conv3_7.bn1.weight'] = torch.ones_like(state_dict['the_block1.0.conv3_7.bn1.running_var'])
    state_dict['the_block1.0.conv3_7.bn1.bias'] = torch.ones_like(state_dict['the_block1.0.conv3_7.bn1.running_var'])

    state_dict['the_block2.0.conv0_0.conv1.weight'] = torch.from_numpy(name_weights['conv0_1']['weight'])

    state_dict['the_block2.0.conv0_0.conv1.bias'] = torch.from_numpy(name_weights['conv0_1']['bias'])
    state_dict['the_block2.0.conv0_0.bn1.running_var'] = from_numpy(name_weights['batch_conv0_1']['running_var'])
    state_dict['the_block2.0.conv0_0.bn1.running_mean'] = from_numpy(name_weights['batch_conv0_1']['running_mean'])
    state_dict['the_block2.0.conv0_0.bn1.weight'] = torch.ones_like(state_dict['the_block2.0.conv0_0.bn1.running_var'])
    state_dict['the_block2.0.conv0_0.bn1.bias'] = torch.ones_like(state_dict['the_block2.0.conv0_0.bn1.running_var'])

    state_dict['the_block2.0.conv2_0.conv1.weight'] = torch.from_numpy(name_weights['conv2_4']['weight'])

    state_dict['the_block2.0.conv2_0.conv1.bias'] = torch.from_numpy(name_weights['conv2_4']['bias'])
    state_dict['the_block2.0.conv2_0.bn1.running_var'] = from_numpy(name_weights['batch_conv2_4']['running_var'])
    state_dict['the_block2.0.conv2_0.bn1.running_mean'] = from_numpy(name_weights['batch_conv2_4']['running_mean'])
    state_dict['the_block2.0.conv2_0.bn1.weight'] = torch.ones_like(state_dict['the_block2.0.conv2_0.bn1.running_var'])
    state_dict['the_block2.0.conv2_0.bn1.bias'] = torch.ones_like(state_dict['the_block2.0.conv2_0.bn1.running_var'])

    state_dict['the_block2.0.conv3_0.conv1.weight'] = torch.from_numpy(name_weights['conv3_8']['weight'])

    state_dict['the_block2.0.conv3_0.conv1.bias'] = torch.from_numpy(name_weights['conv3_8']['bias'])
    state_dict['the_block2.0.conv3_0.bn1.running_var'] = from_numpy(name_weights['batch_conv3_8']['running_var'])
    state_dict['the_block2.0.conv3_0.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_8']['running_mean'])
    state_dict['the_block2.0.conv3_0.bn1.weight'] = torch.ones_like(state_dict['the_block2.0.conv3_0.bn1.running_var'])
    state_dict['the_block2.0.conv3_0.bn1.bias'] = torch.ones_like(state_dict['the_block2.0.conv3_0.bn1.running_var'])

    state_dict['the_block2.0.conv3_1.conv1.weight'] = torch.from_numpy(name_weights['conv3_9']['weight'])

    state_dict['the_block2.0.conv3_1.conv1.bias'] = torch.from_numpy(name_weights['conv3_9']['bias'])
    state_dict['the_block2.0.conv3_1.bn1.running_var'] = from_numpy(name_weights['batch_conv3_9']['running_var'])
    state_dict['the_block2.0.conv3_1.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_9']['running_mean'])
    state_dict['the_block2.0.conv3_1.bn1.weight'] = torch.ones_like(state_dict['the_block2.0.conv3_1.bn1.running_var'])
    state_dict['the_block2.0.conv3_1.bn1.bias'] = torch.ones_like(state_dict['the_block2.0.conv3_1.bn1.running_var'])

    state_dict['the_block2.0.conv2_1.conv1.weight'] = torch.from_numpy(name_weights['conv2_5']['weight'])

    state_dict['the_block2.0.conv2_1.conv1.bias'] = torch.from_numpy(name_weights['conv2_5']['bias'])
    state_dict['the_block2.0.conv2_1.bn1.running_var'] = from_numpy(name_weights['batch_conv2_5']['running_var'])
    state_dict['the_block2.0.conv2_1.bn1.running_mean'] = from_numpy(name_weights['batch_conv2_5']['running_mean'])
    state_dict['the_block2.0.conv2_1.bn1.weight'] = torch.ones_like(state_dict['the_block2.0.conv2_1.bn1.running_var'])
    state_dict['the_block2.0.conv2_1.bn1.bias'] = torch.ones_like(state_dict['the_block2.0.conv2_1.bn1.running_var'])

    state_dict['the_block2.0.conv3_2.conv1.weight'] = torch.from_numpy(name_weights['conv3_10']['weight'])

    state_dict['the_block2.0.conv3_2.conv1.bias'] = torch.from_numpy(name_weights['conv3_10']['bias'])
    state_dict['the_block2.0.conv3_2.bn1.running_var'] = from_numpy(name_weights['batch_conv3_10']['running_var'])
    state_dict['the_block2.0.conv3_2.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_10']['running_mean'])
    state_dict['the_block2.0.conv3_2.bn1.weight'] = torch.ones_like(state_dict['the_block2.0.conv3_2.bn1.running_var'])
    state_dict['the_block2.0.conv3_2.bn1.bias'] = torch.ones_like(state_dict['the_block2.0.conv3_2.bn1.running_var'])

    state_dict['the_block2.0.conv3_3.conv1.weight'] = torch.from_numpy(name_weights['conv3_11']['weight'])

    state_dict['the_block2.0.conv3_3.conv1.bias'] = torch.from_numpy(name_weights['conv3_11']['bias'])
    state_dict['the_block2.0.conv3_3.bn1.running_var'] = from_numpy(name_weights['batch_conv3_11']['running_var'])
    state_dict['the_block2.0.conv3_3.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_11']['running_mean'])
    state_dict['the_block2.0.conv3_3.bn1.weight'] = torch.ones_like(state_dict['the_block2.0.conv3_3.bn1.running_var'])
    state_dict['the_block2.0.conv3_3.bn1.bias'] = torch.ones_like(state_dict['the_block2.0.conv3_3.bn1.running_var'])

    state_dict['the_block2.0.conv1_0.conv1.weight'] = torch.from_numpy(name_weights['conv1_2']['weight'])

    state_dict['the_block2.0.conv1_0.conv1.bias'] = torch.from_numpy(name_weights['conv1_2']['bias'])
    state_dict['the_block2.0.conv1_0.bn1.running_var'] = from_numpy(name_weights['batch_conv1_2']['running_var'])
    state_dict['the_block2.0.conv1_0.bn1.running_mean'] = from_numpy(name_weights['batch_conv1_2']['running_mean'])
    state_dict['the_block2.0.conv1_0.bn1.weight'] = torch.ones_like(state_dict['the_block2.0.conv1_0.bn1.running_var'])
    state_dict['the_block2.0.conv1_0.bn1.bias'] = torch.ones_like(state_dict['the_block2.0.conv1_0.bn1.running_var'])

    state_dict['the_block2.0.conv2_2.conv1.weight'] = torch.from_numpy(name_weights['conv2_6']['weight'])

    state_dict['the_block2.0.conv2_2.conv1.bias'] = torch.from_numpy(name_weights['conv2_6']['bias'])
    state_dict['the_block2.0.conv2_2.bn1.running_var'] = from_numpy(name_weights['batch_conv2_6']['running_var'])
    state_dict['the_block2.0.conv2_2.bn1.running_mean'] = from_numpy(name_weights['batch_conv2_6']['running_mean'])
    state_dict['the_block2.0.conv2_2.bn1.weight'] = torch.ones_like(state_dict['the_block2.0.conv2_2.bn1.running_var'])
    state_dict['the_block2.0.conv2_2.bn1.bias'] = torch.ones_like(state_dict['the_block2.0.conv2_2.bn1.running_var'])

    state_dict['the_block2.0.conv3_4.conv1.weight'] = torch.from_numpy(name_weights['conv3_12']['weight'])

    state_dict['the_block2.0.conv3_4.conv1.bias'] = torch.from_numpy(name_weights['conv3_12']['bias'])
    state_dict['the_block2.0.conv3_4.bn1.running_var'] = from_numpy(name_weights['batch_conv3_12']['running_var'])
    state_dict['the_block2.0.conv3_4.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_12']['running_mean'])
    state_dict['the_block2.0.conv3_4.bn1.weight'] = torch.ones_like(state_dict['the_block2.0.conv3_4.bn1.running_var'])
    state_dict['the_block2.0.conv3_4.bn1.bias'] = torch.ones_like(state_dict['the_block2.0.conv3_4.bn1.running_var'])

    state_dict['the_block2.0.conv3_5.conv1.weight'] = torch.from_numpy(name_weights['conv3_13']['weight'])

    state_dict['the_block2.0.conv3_5.conv1.bias'] = torch.from_numpy(name_weights['conv3_13']['bias'])
    state_dict['the_block2.0.conv3_5.bn1.running_var'] = from_numpy(name_weights['batch_conv3_13']['running_var'])
    state_dict['the_block2.0.conv3_5.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_13']['running_mean'])
    state_dict['the_block2.0.conv3_5.bn1.weight'] = torch.ones_like(state_dict['the_block2.0.conv3_5.bn1.running_var'])
    state_dict['the_block2.0.conv3_5.bn1.bias'] = torch.ones_like(state_dict['the_block2.0.conv3_5.bn1.running_var'])

    state_dict['the_block2.0.conv1_1.conv1.weight'] = torch.from_numpy(name_weights['conv1_3']['weight'])

    state_dict['the_block2.0.conv1_1.conv1.bias'] = torch.from_numpy(name_weights['conv1_3']['bias'])
    state_dict['the_block2.0.conv1_1.bn1.running_var'] = from_numpy(name_weights['batch_conv1_3']['running_var'])
    state_dict['the_block2.0.conv1_1.bn1.running_mean'] = from_numpy(name_weights['batch_conv1_3']['running_mean'])
    state_dict['the_block2.0.conv1_1.bn1.weight'] = torch.ones_like(state_dict['the_block2.0.conv1_1.bn1.running_var'])
    state_dict['the_block2.0.conv1_1.bn1.bias'] = torch.ones_like(state_dict['the_block2.0.conv1_1.bn1.running_var'])

    state_dict['the_block2.0.conv2_3.conv1.weight'] = torch.from_numpy(name_weights['conv2_7']['weight'])

    state_dict['the_block2.0.conv2_3.conv1.bias'] = torch.from_numpy(name_weights['conv2_7']['bias'])
    state_dict['the_block2.0.conv2_3.bn1.running_var'] = from_numpy(name_weights['batch_conv2_7']['running_var'])
    state_dict['the_block2.0.conv2_3.bn1.running_mean'] = from_numpy(name_weights['batch_conv2_7']['running_mean'])
    state_dict['the_block2.0.conv2_3.bn1.weight'] = torch.ones_like(state_dict['the_block2.0.conv2_3.bn1.running_var'])
    state_dict['the_block2.0.conv2_3.bn1.bias'] = torch.ones_like(state_dict['the_block2.0.conv2_3.bn1.running_var'])

    state_dict['the_block2.0.conv3_6.conv1.weight'] = torch.from_numpy(name_weights['conv3_14']['weight'])

    state_dict['the_block2.0.conv3_6.conv1.bias'] = torch.from_numpy(name_weights['conv3_14']['bias'])
    state_dict['the_block2.0.conv3_6.bn1.running_var'] = from_numpy(name_weights['batch_conv3_14']['running_var'])
    state_dict['the_block2.0.conv3_6.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_14']['running_mean'])
    state_dict['the_block2.0.conv3_6.bn1.weight'] = torch.ones_like(state_dict['the_block2.0.conv3_6.bn1.running_var'])
    state_dict['the_block2.0.conv3_6.bn1.bias'] = torch.ones_like(state_dict['the_block2.0.conv3_6.bn1.running_var'])

    state_dict['the_block2.0.conv3_7.conv1.weight'] = torch.from_numpy(name_weights['conv3_15']['weight'])

    state_dict['the_block2.0.conv3_7.conv1.bias'] = torch.from_numpy(name_weights['conv3_15']['bias'])
    state_dict['the_block2.0.conv3_7.bn1.running_var'] = from_numpy(name_weights['batch_conv3_15']['running_var'])
    state_dict['the_block2.0.conv3_7.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_15']['running_mean'])
    state_dict['the_block2.0.conv3_7.bn1.weight'] = torch.ones_like(state_dict['the_block2.0.conv3_7.bn1.running_var'])
    state_dict['the_block2.0.conv3_7.bn1.bias'] = torch.ones_like(state_dict['the_block2.0.conv3_7.bn1.running_var'])

    state_dict['the_block3.0.conv0_0.conv1.weight'] = torch.from_numpy(name_weights['conv0_2']['weight'])

    state_dict['the_block3.0.conv0_0.conv1.bias'] = torch.from_numpy(name_weights['conv0_2']['bias'])
    state_dict['the_block3.0.conv0_0.bn1.running_var'] = from_numpy(name_weights['batch_conv0_2']['running_var'])
    state_dict['the_block3.0.conv0_0.bn1.running_mean'] = from_numpy(name_weights['batch_conv0_2']['running_mean'])
    state_dict['the_block3.0.conv0_0.bn1.weight'] = torch.ones_like(state_dict['the_block3.0.conv0_0.bn1.running_var'])
    state_dict['the_block3.0.conv0_0.bn1.bias'] = torch.ones_like(state_dict['the_block3.0.conv0_0.bn1.running_var'])

    state_dict['the_block3.0.conv2_0.conv1.weight'] = torch.from_numpy(name_weights['conv2_8']['weight'])

    state_dict['the_block3.0.conv2_0.conv1.bias'] = torch.from_numpy(name_weights['conv2_8']['bias'])
    state_dict['the_block3.0.conv2_0.bn1.running_var'] = from_numpy(name_weights['batch_conv2_8']['running_var'])
    state_dict['the_block3.0.conv2_0.bn1.running_mean'] = from_numpy(name_weights['batch_conv2_8']['running_mean'])
    state_dict['the_block3.0.conv2_0.bn1.weight'] = torch.ones_like(state_dict['the_block3.0.conv2_0.bn1.running_var'])
    state_dict['the_block3.0.conv2_0.bn1.bias'] = torch.ones_like(state_dict['the_block3.0.conv2_0.bn1.running_var'])

    state_dict['the_block3.0.conv3_0.conv1.weight'] = torch.from_numpy(name_weights['conv3_16']['weight'])

    state_dict['the_block3.0.conv3_0.conv1.bias'] = torch.from_numpy(name_weights['conv3_16']['bias'])
    state_dict['the_block3.0.conv3_0.bn1.running_var'] = from_numpy(name_weights['batch_conv3_16']['running_var'])
    state_dict['the_block3.0.conv3_0.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_16']['running_mean'])
    state_dict['the_block3.0.conv3_0.bn1.weight'] = torch.ones_like(state_dict['the_block3.0.conv3_0.bn1.running_var'])
    state_dict['the_block3.0.conv3_0.bn1.bias'] = torch.ones_like(state_dict['the_block3.0.conv3_0.bn1.running_var'])

    state_dict['the_block3.0.conv3_1.conv1.weight'] = torch.from_numpy(name_weights['conv3_17']['weight'])

    state_dict['the_block3.0.conv3_1.conv1.bias'] = torch.from_numpy(name_weights['conv3_17']['bias'])
    state_dict['the_block3.0.conv3_1.bn1.running_var'] = from_numpy(name_weights['batch_conv3_17']['running_var'])
    state_dict['the_block3.0.conv3_1.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_17']['running_mean'])
    state_dict['the_block3.0.conv3_1.bn1.weight'] = torch.ones_like(state_dict['the_block3.0.conv3_1.bn1.running_var'])
    state_dict['the_block3.0.conv3_1.bn1.bias'] = torch.ones_like(state_dict['the_block3.0.conv3_1.bn1.running_var'])

    state_dict['the_block3.0.conv2_1.conv1.weight'] = torch.from_numpy(name_weights['conv2_9']['weight'])

    state_dict['the_block3.0.conv2_1.conv1.bias'] = torch.from_numpy(name_weights['conv2_9']['bias'])
    state_dict['the_block3.0.conv2_1.bn1.running_var'] = from_numpy(name_weights['batch_conv2_9']['running_var'])
    state_dict['the_block3.0.conv2_1.bn1.running_mean'] = from_numpy(name_weights['batch_conv2_9']['running_mean'])
    state_dict['the_block3.0.conv2_1.bn1.weight'] = torch.ones_like(state_dict['the_block3.0.conv2_1.bn1.running_var'])
    state_dict['the_block3.0.conv2_1.bn1.bias'] = torch.ones_like(state_dict['the_block3.0.conv2_1.bn1.running_var'])

    state_dict['the_block3.0.conv3_2.conv1.weight'] = torch.from_numpy(name_weights['conv3_18']['weight'])

    state_dict['the_block3.0.conv3_2.conv1.bias'] = torch.from_numpy(name_weights['conv3_18']['bias'])
    state_dict['the_block3.0.conv3_2.bn1.running_var'] = from_numpy(name_weights['batch_conv3_18']['running_var'])
    state_dict['the_block3.0.conv3_2.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_18']['running_mean'])
    state_dict['the_block3.0.conv3_2.bn1.weight'] = torch.ones_like(state_dict['the_block3.0.conv3_2.bn1.running_var'])
    state_dict['the_block3.0.conv3_2.bn1.bias'] = torch.ones_like(state_dict['the_block3.0.conv3_2.bn1.running_var'])

    state_dict['the_block3.0.conv3_3.conv1.weight'] = torch.from_numpy(name_weights['conv3_19']['weight'])

    state_dict['the_block3.0.conv3_3.conv1.bias'] = torch.from_numpy(name_weights['conv3_19']['bias'])
    state_dict['the_block3.0.conv3_3.bn1.running_var'] = from_numpy(name_weights['batch_conv3_19']['running_var'])
    state_dict['the_block3.0.conv3_3.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_19']['running_mean'])
    state_dict['the_block3.0.conv3_3.bn1.weight'] = torch.ones_like(state_dict['the_block3.0.conv3_3.bn1.running_var'])
    state_dict['the_block3.0.conv3_3.bn1.bias'] = torch.ones_like(state_dict['the_block3.0.conv3_3.bn1.running_var'])

    state_dict['the_block3.0.conv1_0.conv1.weight'] = torch.from_numpy(name_weights['conv1_4']['weight'])

    state_dict['the_block3.0.conv1_0.conv1.bias'] = torch.from_numpy(name_weights['conv1_4']['bias'])
    state_dict['the_block3.0.conv1_0.bn1.running_var'] = from_numpy(name_weights['batch_conv1_4']['running_var'])
    state_dict['the_block3.0.conv1_0.bn1.running_mean'] = from_numpy(name_weights['batch_conv1_4']['running_mean'])
    state_dict['the_block3.0.conv1_0.bn1.weight'] = torch.ones_like(state_dict['the_block3.0.conv1_0.bn1.running_var'])
    state_dict['the_block3.0.conv1_0.bn1.bias'] = torch.ones_like(state_dict['the_block3.0.conv1_0.bn1.running_var'])

    state_dict['the_block3.0.conv2_2.conv1.weight'] = torch.from_numpy(name_weights['conv2_10']['weight'])

    state_dict['the_block3.0.conv2_2.conv1.bias'] = torch.from_numpy(name_weights['conv2_10']['bias'])
    state_dict['the_block3.0.conv2_2.bn1.running_var'] = from_numpy(name_weights['batch_conv2_10']['running_var'])
    state_dict['the_block3.0.conv2_2.bn1.running_mean'] = from_numpy(name_weights['batch_conv2_10']['running_mean'])
    state_dict['the_block3.0.conv2_2.bn1.weight'] = torch.ones_like(state_dict['the_block3.0.conv2_2.bn1.running_var'])
    state_dict['the_block3.0.conv2_2.bn1.bias'] = torch.ones_like(state_dict['the_block3.0.conv2_2.bn1.running_var'])

    state_dict['the_block3.0.conv3_4.conv1.weight'] = torch.from_numpy(name_weights['conv3_20']['weight'])

    state_dict['the_block3.0.conv3_4.conv1.bias'] = torch.from_numpy(name_weights['conv3_20']['bias'])
    state_dict['the_block3.0.conv3_4.bn1.running_var'] = from_numpy(name_weights['batch_conv3_20']['running_var'])
    state_dict['the_block3.0.conv3_4.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_20']['running_mean'])
    state_dict['the_block3.0.conv3_4.bn1.weight'] = torch.ones_like(state_dict['the_block3.0.conv3_4.bn1.running_var'])
    state_dict['the_block3.0.conv3_4.bn1.bias'] = torch.ones_like(state_dict['the_block3.0.conv3_4.bn1.running_var'])

    state_dict['the_block3.0.conv3_5.conv1.weight'] = torch.from_numpy(name_weights['conv3_21']['weight'])

    state_dict['the_block3.0.conv3_5.conv1.bias'] = torch.from_numpy(name_weights['conv3_21']['bias'])
    state_dict['the_block3.0.conv3_5.bn1.running_var'] = from_numpy(name_weights['batch_conv3_21']['running_var'])
    state_dict['the_block3.0.conv3_5.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_21']['running_mean'])
    state_dict['the_block3.0.conv3_5.bn1.weight'] = torch.ones_like(state_dict['the_block3.0.conv3_5.bn1.running_var'])
    state_dict['the_block3.0.conv3_5.bn1.bias'] = torch.ones_like(state_dict['the_block3.0.conv3_5.bn1.running_var'])

    state_dict['the_block3.0.conv1_1.conv1.weight'] = torch.from_numpy(name_weights['conv1_5']['weight'])

    state_dict['the_block3.0.conv1_1.conv1.bias'] = torch.from_numpy(name_weights['conv1_5']['bias'])
    state_dict['the_block3.0.conv1_1.bn1.running_var'] = from_numpy(name_weights['batch_conv1_5']['running_var'])
    state_dict['the_block3.0.conv1_1.bn1.running_mean'] = from_numpy(name_weights['batch_conv1_5']['running_mean'])
    state_dict['the_block3.0.conv1_1.bn1.weight'] = torch.ones_like(state_dict['the_block3.0.conv1_1.bn1.running_var'])
    state_dict['the_block3.0.conv1_1.bn1.bias'] = torch.ones_like(state_dict['the_block3.0.conv1_1.bn1.running_var'])

    state_dict['the_block3.0.conv2_3.conv1.weight'] = torch.from_numpy(name_weights['conv2_11']['weight'])

    state_dict['the_block3.0.conv2_3.conv1.bias'] = torch.from_numpy(name_weights['conv2_11']['bias'])
    state_dict['the_block3.0.conv2_3.bn1.running_var'] = from_numpy(name_weights['batch_conv2_11']['running_var'])
    state_dict['the_block3.0.conv2_3.bn1.running_mean'] = from_numpy(name_weights['batch_conv2_11']['running_mean'])
    state_dict['the_block3.0.conv2_3.bn1.weight'] = torch.ones_like(state_dict['the_block3.0.conv2_3.bn1.running_var'])
    state_dict['the_block3.0.conv2_3.bn1.bias'] = torch.ones_like(state_dict['the_block3.0.conv2_3.bn1.running_var'])

    state_dict['the_block3.0.conv3_6.conv1.weight'] = torch.from_numpy(name_weights['conv3_22']['weight'])

    state_dict['the_block3.0.conv3_6.conv1.bias'] = torch.from_numpy(name_weights['conv3_22']['bias'])
    state_dict['the_block3.0.conv3_6.bn1.running_var'] = from_numpy(name_weights['batch_conv3_22']['running_var'])
    state_dict['the_block3.0.conv3_6.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_22']['running_mean'])
    state_dict['the_block3.0.conv3_6.bn1.weight'] = torch.ones_like(state_dict['the_block3.0.conv3_6.bn1.running_var'])
    state_dict['the_block3.0.conv3_6.bn1.bias'] = torch.ones_like(state_dict['the_block3.0.conv3_6.bn1.running_var'])

    state_dict['the_block3.0.conv3_7.conv1.weight'] = torch.from_numpy(name_weights['conv3_23']['weight'])

    state_dict['the_block3.0.conv3_7.conv1.bias'] = torch.from_numpy(name_weights['conv3_23']['bias'])
    state_dict['the_block3.0.conv3_7.bn1.running_var'] = from_numpy(name_weights['batch_conv3_23']['running_var'])
    state_dict['the_block3.0.conv3_7.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_23']['running_mean'])
    state_dict['the_block3.0.conv3_7.bn1.weight'] = torch.ones_like(state_dict['the_block3.0.conv3_7.bn1.running_var'])
    state_dict['the_block3.0.conv3_7.bn1.bias'] = torch.ones_like(state_dict['the_block3.0.conv3_7.bn1.running_var'])

    state_dict['the_block4.0.conv0_0.conv1.weight'] = torch.from_numpy(name_weights['conv0_3']['weight'])

    state_dict['the_block4.0.conv0_0.conv1.bias'] = torch.from_numpy(name_weights['conv0_3']['bias'])
    state_dict['the_block4.0.conv0_0.bn1.running_var'] = from_numpy(name_weights['batch_conv0_3']['running_var'])
    state_dict['the_block4.0.conv0_0.bn1.running_mean'] = from_numpy(name_weights['batch_conv0_3']['running_mean'])
    state_dict['the_block4.0.conv0_0.bn1.weight'] = torch.ones_like(state_dict['the_block4.0.conv0_0.bn1.running_var'])
    state_dict['the_block4.0.conv0_0.bn1.bias'] = torch.ones_like(state_dict['the_block4.0.conv0_0.bn1.running_var'])

    state_dict['the_block4.0.conv2_0.conv1.weight'] = torch.from_numpy(name_weights['conv2_12']['weight'])

    state_dict['the_block4.0.conv2_0.conv1.bias'] = torch.from_numpy(name_weights['conv2_12']['bias'])
    state_dict['the_block4.0.conv2_0.bn1.running_var'] = from_numpy(name_weights['batch_conv2_12']['running_var'])
    state_dict['the_block4.0.conv2_0.bn1.running_mean'] = from_numpy(name_weights['batch_conv2_12']['running_mean'])
    state_dict['the_block4.0.conv2_0.bn1.weight'] = torch.ones_like(state_dict['the_block4.0.conv2_0.bn1.running_var'])
    state_dict['the_block4.0.conv2_0.bn1.bias'] = torch.ones_like(state_dict['the_block4.0.conv2_0.bn1.running_var'])

    state_dict['the_block4.0.conv3_0.conv1.weight'] = torch.from_numpy(name_weights['conv3_24']['weight'])

    state_dict['the_block4.0.conv3_0.conv1.bias'] = torch.from_numpy(name_weights['conv3_24']['bias'])
    state_dict['the_block4.0.conv3_0.bn1.running_var'] = from_numpy(name_weights['batch_conv3_24']['running_var'])
    state_dict['the_block4.0.conv3_0.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_24']['running_mean'])
    state_dict['the_block4.0.conv3_0.bn1.weight'] = torch.ones_like(state_dict['the_block4.0.conv3_0.bn1.running_var'])
    state_dict['the_block4.0.conv3_0.bn1.bias'] = torch.ones_like(state_dict['the_block4.0.conv3_0.bn1.running_var'])

    state_dict['the_block4.0.conv3_1.conv1.weight'] = torch.from_numpy(name_weights['conv3_25']['weight'])

    state_dict['the_block4.0.conv3_1.conv1.bias'] = torch.from_numpy(name_weights['conv3_25']['bias'])
    state_dict['the_block4.0.conv3_1.bn1.running_var'] = from_numpy(name_weights['batch_conv3_25']['running_var'])
    state_dict['the_block4.0.conv3_1.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_25']['running_mean'])
    state_dict['the_block4.0.conv3_1.bn1.weight'] = torch.ones_like(state_dict['the_block4.0.conv3_1.bn1.running_var'])
    state_dict['the_block4.0.conv3_1.bn1.bias'] = torch.ones_like(state_dict['the_block4.0.conv3_1.bn1.running_var'])

    state_dict['the_block4.0.conv2_1.conv1.weight'] = torch.from_numpy(name_weights['conv2_13']['weight'])

    state_dict['the_block4.0.conv2_1.conv1.bias'] = torch.from_numpy(name_weights['conv2_13']['bias'])
    state_dict['the_block4.0.conv2_1.bn1.running_var'] = from_numpy(name_weights['batch_conv2_13']['running_var'])
    state_dict['the_block4.0.conv2_1.bn1.running_mean'] = from_numpy(name_weights['batch_conv2_13']['running_mean'])
    state_dict['the_block4.0.conv2_1.bn1.weight'] = torch.ones_like(state_dict['the_block4.0.conv2_1.bn1.running_var'])
    state_dict['the_block4.0.conv2_1.bn1.bias'] = torch.ones_like(state_dict['the_block4.0.conv2_1.bn1.running_var'])

    state_dict['the_block4.0.conv3_2.conv1.weight'] = torch.from_numpy(name_weights['conv3_26']['weight'])

    state_dict['the_block4.0.conv3_2.conv1.bias'] = torch.from_numpy(name_weights['conv3_26']['bias'])
    state_dict['the_block4.0.conv3_2.bn1.running_var'] = from_numpy(name_weights['batch_conv3_26']['running_var'])
    state_dict['the_block4.0.conv3_2.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_26']['running_mean'])
    state_dict['the_block4.0.conv3_2.bn1.weight'] = torch.ones_like(state_dict['the_block4.0.conv3_2.bn1.running_var'])
    state_dict['the_block4.0.conv3_2.bn1.bias'] = torch.ones_like(state_dict['the_block4.0.conv3_2.bn1.running_var'])

    state_dict['the_block4.0.conv3_3.conv1.weight'] = torch.from_numpy(name_weights['conv3_27']['weight'])

    state_dict['the_block4.0.conv3_3.conv1.bias'] = torch.from_numpy(name_weights['conv3_27']['bias'])
    state_dict['the_block4.0.conv3_3.bn1.running_var'] = from_numpy(name_weights['batch_conv3_27']['running_var'])
    state_dict['the_block4.0.conv3_3.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_27']['running_mean'])
    state_dict['the_block4.0.conv3_3.bn1.weight'] = torch.ones_like(state_dict['the_block4.0.conv3_3.bn1.running_var'])
    state_dict['the_block4.0.conv3_3.bn1.bias'] = torch.ones_like(state_dict['the_block4.0.conv3_3.bn1.running_var'])

    state_dict['the_block4.0.conv1_0.conv1.weight'] = torch.from_numpy(name_weights['conv1_6']['weight'])

    state_dict['the_block4.0.conv1_0.conv1.bias'] = torch.from_numpy(name_weights['conv1_6']['bias'])
    state_dict['the_block4.0.conv1_0.bn1.running_var'] = from_numpy(name_weights['batch_conv1_6']['running_var'])
    state_dict['the_block4.0.conv1_0.bn1.running_mean'] = from_numpy(name_weights['batch_conv1_6']['running_mean'])
    state_dict['the_block4.0.conv1_0.bn1.weight'] = torch.ones_like(state_dict['the_block4.0.conv1_0.bn1.running_var'])
    state_dict['the_block4.0.conv1_0.bn1.bias'] = torch.ones_like(state_dict['the_block4.0.conv1_0.bn1.running_var'])

    state_dict['the_block4.0.conv2_2.conv1.weight'] = torch.from_numpy(name_weights['conv2_14']['weight'])

    state_dict['the_block4.0.conv2_2.conv1.bias'] = torch.from_numpy(name_weights['conv2_14']['bias'])
    state_dict['the_block4.0.conv2_2.bn1.running_var'] = from_numpy(name_weights['batch_conv2_14']['running_var'])
    state_dict['the_block4.0.conv2_2.bn1.running_mean'] = from_numpy(name_weights['batch_conv2_14']['running_mean'])
    state_dict['the_block4.0.conv2_2.bn1.weight'] = torch.ones_like(state_dict['the_block4.0.conv2_2.bn1.running_var'])
    state_dict['the_block4.0.conv2_2.bn1.bias'] = torch.ones_like(state_dict['the_block4.0.conv2_2.bn1.running_var'])

    state_dict['the_block4.0.conv3_4.conv1.weight'] = torch.from_numpy(name_weights['conv3_28']['weight'])

    state_dict['the_block4.0.conv3_4.conv1.bias'] = torch.from_numpy(name_weights['conv3_28']['bias'])
    state_dict['the_block4.0.conv3_4.bn1.running_var'] = from_numpy(name_weights['batch_conv3_28']['running_var'])
    state_dict['the_block4.0.conv3_4.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_28']['running_mean'])
    state_dict['the_block4.0.conv3_4.bn1.weight'] = torch.ones_like(state_dict['the_block4.0.conv3_4.bn1.running_var'])
    state_dict['the_block4.0.conv3_4.bn1.bias'] = torch.ones_like(state_dict['the_block4.0.conv3_4.bn1.running_var'])

    state_dict['the_block4.0.conv3_5.conv1.weight'] = torch.from_numpy(name_weights['conv3_29']['weight'])

    state_dict['the_block4.0.conv3_5.conv1.bias'] = torch.from_numpy(name_weights['conv3_29']['bias'])
    state_dict['the_block4.0.conv3_5.bn1.running_var'] = from_numpy(name_weights['batch_conv3_29']['running_var'])
    state_dict['the_block4.0.conv3_5.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_29']['running_mean'])
    state_dict['the_block4.0.conv3_5.bn1.weight'] = torch.ones_like(state_dict['the_block4.0.conv3_5.bn1.running_var'])
    state_dict['the_block4.0.conv3_5.bn1.bias'] = torch.ones_like(state_dict['the_block4.0.conv3_5.bn1.running_var'])

    state_dict['the_block4.0.conv1_1.conv1.weight'] = torch.from_numpy(name_weights['conv1_7']['weight'])

    state_dict['the_block4.0.conv1_1.conv1.bias'] = torch.from_numpy(name_weights['conv1_7']['bias'])
    state_dict['the_block4.0.conv1_1.bn1.running_var'] = from_numpy(name_weights['batch_conv1_7']['running_var'])
    state_dict['the_block4.0.conv1_1.bn1.running_mean'] = from_numpy(name_weights['batch_conv1_7']['running_mean'])
    state_dict['the_block4.0.conv1_1.bn1.weight'] = torch.ones_like(state_dict['the_block4.0.conv1_1.bn1.running_var'])
    state_dict['the_block4.0.conv1_1.bn1.bias'] = torch.ones_like(state_dict['the_block4.0.conv1_1.bn1.running_var'])

    state_dict['the_block4.0.conv2_3.conv1.weight'] = torch.from_numpy(name_weights['conv2_15']['weight'])

    state_dict['the_block4.0.conv2_3.conv1.bias'] = torch.from_numpy(name_weights['conv2_15']['bias'])
    state_dict['the_block4.0.conv2_3.bn1.running_var'] = from_numpy(name_weights['batch_conv2_15']['running_var'])
    state_dict['the_block4.0.conv2_3.bn1.running_mean'] = from_numpy(name_weights['batch_conv2_15']['running_mean'])
    state_dict['the_block4.0.conv2_3.bn1.weight'] = torch.ones_like(state_dict['the_block4.0.conv2_3.bn1.running_var'])
    state_dict['the_block4.0.conv2_3.bn1.bias'] = torch.ones_like(state_dict['the_block4.0.conv2_3.bn1.running_var'])

    state_dict['the_block4.0.conv3_6.conv1.weight'] = torch.from_numpy(name_weights['conv3_30']['weight'])

    state_dict['the_block4.0.conv3_6.conv1.bias'] = torch.from_numpy(name_weights['conv3_30']['bias'])
    state_dict['the_block4.0.conv3_6.bn1.running_var'] = from_numpy(name_weights['batch_conv3_30']['running_var'])
    state_dict['the_block4.0.conv3_6.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_30']['running_mean'])
    state_dict['the_block4.0.conv3_6.bn1.weight'] = torch.ones_like(state_dict['the_block4.0.conv3_6.bn1.running_var'])
    state_dict['the_block4.0.conv3_6.bn1.bias'] = torch.ones_like(state_dict['the_block4.0.conv3_6.bn1.running_var'])

    state_dict['the_block4.0.conv3_7.conv1.weight'] = torch.from_numpy(name_weights['conv3_31']['weight'])

    state_dict['the_block4.0.conv3_7.conv1.bias'] = torch.from_numpy(name_weights['conv3_31']['bias'])
    state_dict['the_block4.0.conv3_7.bn1.running_var'] = from_numpy(name_weights['batch_conv3_31']['running_var'])
    state_dict['the_block4.0.conv3_7.bn1.running_mean'] = from_numpy(name_weights['batch_conv3_31']['running_mean'])
    state_dict['the_block4.0.conv3_7.bn1.weight'] = torch.ones_like(state_dict['the_block4.0.conv3_7.bn1.running_var'])
    state_dict['the_block4.0.conv3_7.bn1.bias'] = torch.ones_like(state_dict['the_block4.0.conv3_7.bn1.running_var'])
    return state_dict
