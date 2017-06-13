import numpy as np
import torch

def load_weights(model,weights_file,dtype):

    # Print parameter names for our and their model for debugging

    # for name, param in model.named_parameters():
    #     print(name+'\n')


    # for op_name in data_dict:
    #    print(op_name)
    #    for param_name, data in iter(data_dict[op_name].items()):
    #         print(param_name)

    model_params = model.state_dict()
    data_dict = np.load(weights_file, encoding='latin1').item()

    ####

    model_params['conv1.weight'] = torch.from_numpy(data_dict['conv1']['weights']).type(dtype).permute(3,2,0,1)
    model_params['conv1.bias'] = torch.from_numpy(data_dict['conv1']['biases']).type(dtype)
    model_params['bn1.weight'] = torch.from_numpy(data_dict['bn_conv1']['scale']).type(dtype)
    model_params['bn1.bias'] = torch.from_numpy(data_dict['bn_conv1']['offset']).type(dtype)

    ####

    model_params['proj_layer1.conv4.weight'] = torch.from_numpy(data_dict['res2a_branch1']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['proj_layer1.conv4.bias'] = torch.FloatTensor(model_params['proj_layer1.conv4.weight'].size()[0]).zero_()
    model_params['proj_layer1.bn4.weight'] = torch.from_numpy(data_dict['bn2a_branch1']['scale']).type(dtype)
    model_params['proj_layer1.bn4.bias'] = torch.from_numpy(data_dict['bn2a_branch1']['offset']).type(dtype)

    model_params['proj_layer1.conv1.weight'] = torch.from_numpy(data_dict['res2a_branch2a']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['proj_layer1.conv1.bias'] = torch.FloatTensor(model_params['proj_layer1.conv1.weight'].size()[0]).zero_()
    model_params['proj_layer1.bn1.weight'] = torch.from_numpy(data_dict['bn2a_branch2a']['scale']).type(dtype)
    model_params['proj_layer1.bn1.bias'] = torch.from_numpy(data_dict['bn2a_branch2a']['offset']).type(dtype)

    model_params['proj_layer1.conv2.weight'] = torch.from_numpy(data_dict['res2a_branch2b']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['proj_layer1.conv2.bias'] =  torch.FloatTensor(model_params['proj_layer1.conv2.weight'].size()[0]).zero_()
    model_params['proj_layer1.bn2.weight'] = torch.from_numpy(data_dict['bn2a_branch2b']['scale']).type(dtype)
    model_params['proj_layer1.bn2.bias'] = torch.from_numpy(data_dict['bn2a_branch2b']['offset']).type(dtype)

    model_params['proj_layer1.conv3.weight'] = torch.from_numpy(data_dict['res2a_branch2c']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['proj_layer1.conv3.bias'] =  torch.FloatTensor(model_params['proj_layer1.conv3.weight'].size()[0]).zero_()
    model_params['proj_layer1.bn3.weight'] = torch.from_numpy(data_dict['bn2a_branch2c']['scale']).type(dtype)
    model_params['proj_layer1.bn3.bias'] =  torch.from_numpy(data_dict['bn2a_branch2c']['offset']).type(dtype)

    model_params['skip_layer1_1.conv1.weight'] = torch.from_numpy(data_dict['res2b_branch2a']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer1_1.conv1.bias'] =  torch.FloatTensor(model_params['skip_layer1_1.conv1.weight'].size()[0]).zero_()
    model_params['skip_layer1_1.bn1.weight'] = torch.from_numpy(data_dict['bn2b_branch2a']['scale']).type(dtype)
    model_params['skip_layer1_1.bn1.bias'] =  torch.from_numpy(data_dict['bn2b_branch2a']['offset']).type(dtype)

    model_params['skip_layer1_1.conv2.weight'] = torch.from_numpy(data_dict['res2b_branch2b']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer1_1.conv2.bias'] =  torch.FloatTensor(model_params['skip_layer1_1.conv2.weight'].size()[0]).zero_()
    model_params['skip_layer1_1.bn2.weight'] = torch.from_numpy(data_dict['bn2b_branch2b']['scale']).type(dtype)
    model_params['skip_layer1_1.bn2.bias'] =  torch.from_numpy(data_dict['bn2b_branch2b']['offset']).type(dtype)

    model_params['skip_layer1_1.conv3.weight'] = torch.from_numpy(data_dict['res2b_branch2c']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer1_1.conv3.bias'] = torch.FloatTensor(model_params['skip_layer1_1.conv3.weight'].size()[0]).zero_()
    model_params['skip_layer1_1.bn3.weight'] = torch.from_numpy(data_dict['bn2b_branch2c']['scale']).type(dtype)
    model_params['skip_layer1_1.bn3.bias'] =  torch.from_numpy(data_dict['bn2b_branch2c']['offset']).type(dtype)

    model_params['skip_layer1_2.conv1.weight'] = torch.from_numpy(data_dict['res2c_branch2a']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer1_2.conv1.bias'] =  torch.FloatTensor(model_params['skip_layer1_2.conv1.weight'].size()[0]).zero_()
    model_params['skip_layer1_2.bn1.weight'] = torch.from_numpy(data_dict['bn2c_branch2a']['scale']).type(dtype)
    model_params['skip_layer1_2.bn1.bias'] =  torch.from_numpy(data_dict['bn2c_branch2a']['offset']).type(dtype)

    model_params['skip_layer1_2.conv2.weight'] = torch.from_numpy(data_dict['res2c_branch2b']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer1_2.conv2.bias'] =  torch.FloatTensor(model_params['skip_layer1_2.conv2.weight'].size()[0]).zero_()
    model_params['skip_layer1_2.bn2.weight'] = torch.from_numpy(data_dict['bn2c_branch2b']['scale']).type(dtype)
    model_params['skip_layer1_2.bn2.bias'] =  torch.from_numpy(data_dict['bn2c_branch2b']['offset']).type(dtype)

    model_params['skip_layer1_2.conv3.weight'] = torch.from_numpy(data_dict['res2c_branch2c']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer1_2.conv3.bias'] =  torch.FloatTensor(model_params['skip_layer1_2.conv3.weight'].size()[0]).zero_()
    model_params['skip_layer1_2.bn3.weight'] = torch.from_numpy(data_dict['bn2c_branch2c']['scale']).type(dtype)
    model_params['skip_layer1_2.bn3.bias'] =  torch.from_numpy(data_dict['bn2c_branch2c']['offset']).type(dtype)

    ###

    model_params['proj_layer2.conv4.weight'] = torch.from_numpy(data_dict['res3a_branch1']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['proj_layer2.conv4.bias'] = torch.FloatTensor(model_params['proj_layer2.conv4.weight'].size()[0]).zero_()
    model_params['proj_layer2.bn4.weight'] = torch.from_numpy(data_dict['bn3a_branch1']['scale']).type(dtype)
    model_params['proj_layer2.bn4.bias'] = torch.from_numpy(data_dict['bn3a_branch1']['offset']).type(dtype)

    model_params['proj_layer2.conv1.weight'] = torch.from_numpy(data_dict['res3a_branch2a']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['proj_layer2.conv1.bias'] = torch.FloatTensor(model_params['proj_layer2.conv1.weight'].size()[0]).zero_()
    model_params['proj_layer2.bn1.weight'] = torch.from_numpy(data_dict['bn3a_branch2a']['scale']).type(dtype)
    model_params['proj_layer2.bn1.bias'] = torch.from_numpy(data_dict['bn3a_branch2a']['offset']).type(dtype)

    model_params['proj_layer2.conv2.weight'] = torch.from_numpy(data_dict['res3a_branch2b']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['proj_layer2.conv2.bias'] =  torch.FloatTensor(model_params['proj_layer2.conv2.weight'].size()[0]).zero_()
    model_params['proj_layer2.bn2.weight'] = torch.from_numpy(data_dict['bn3a_branch2b']['scale']).type(dtype)
    model_params['proj_layer2.bn2.bias'] = torch.from_numpy(data_dict['bn3a_branch2b']['offset']).type(dtype)

    model_params['proj_layer2.conv3.weight'] = torch.from_numpy(data_dict['res3a_branch2c']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['proj_layer2.conv3.bias'] =  torch.FloatTensor(model_params['proj_layer2.conv3.weight'].size()[0]).zero_()
    model_params['proj_layer2.bn3.weight'] = torch.from_numpy(data_dict['bn3a_branch2c']['scale']).type(dtype)
    model_params['proj_layer2.bn3.bias'] =  torch.from_numpy(data_dict['bn3a_branch2c']['offset']).type(dtype)

    model_params['skip_layer2_1.conv1.weight'] = torch.from_numpy(data_dict['res3b_branch2a']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer2_1.conv1.bias'] =  torch.FloatTensor(model_params['skip_layer2_1.conv1.weight'].size()[0]).zero_()
    model_params['skip_layer2_1.bn1.weight'] = torch.from_numpy(data_dict['bn3b_branch2a']['scale']).type(dtype)
    model_params['skip_layer2_1.bn1.bias'] =  torch.from_numpy(data_dict['bn3b_branch2a']['offset']).type(dtype)

    model_params['skip_layer2_1.conv2.weight'] = torch.from_numpy(data_dict['res3b_branch2b']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer2_1.conv2.bias'] =  torch.FloatTensor(model_params['skip_layer2_1.conv2.weight'].size()[0]).zero_()
    model_params['skip_layer2_1.bn2.weight'] = torch.from_numpy(data_dict['bn3b_branch2b']['scale']).type(dtype)
    model_params['skip_layer2_1.bn2.bias'] =  torch.from_numpy(data_dict['bn3b_branch2b']['offset']).type(dtype)

    model_params['skip_layer2_1.conv3.weight'] = torch.from_numpy(data_dict['res3b_branch2c']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer2_1.conv3.bias'] =  torch.FloatTensor(model_params['skip_layer2_1.conv3.weight'].size()[0]).zero_()
    model_params['skip_layer2_1.bn3.weight'] = torch.from_numpy(data_dict['bn3b_branch2c']['scale']).type(dtype)
    model_params['skip_layer2_1.bn3.bias'] =  torch.from_numpy(data_dict['bn3b_branch2c']['offset']).type(dtype)

    model_params['skip_layer2_2.conv1.weight'] = torch.from_numpy(data_dict['res3c_branch2a']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer2_2.conv1.bias'] =  torch.FloatTensor(model_params['skip_layer2_2.conv1.weight'].size()[0]).zero_()
    model_params['skip_layer2_2.bn1.weight'] = torch.from_numpy(data_dict['bn3c_branch2a']['scale']).type(dtype)
    model_params['skip_layer2_2.bn1.bias'] =  torch.from_numpy(data_dict['bn3c_branch2a']['offset']).type(dtype)

    model_params['skip_layer2_2.conv2.weight'] = torch.from_numpy(data_dict['res3c_branch2b']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer2_2.conv2.bias'] =  torch.FloatTensor(model_params['skip_layer2_2.conv2.weight'].size()[0]).zero_()
    model_params['skip_layer2_2.bn2.weight'] = torch.from_numpy(data_dict['bn3c_branch2b']['scale']).type(dtype)
    model_params['skip_layer2_2.bn2.bias'] =  torch.from_numpy(data_dict['bn3c_branch2b']['offset']).type(dtype)

    model_params['skip_layer2_2.conv3.weight'] = torch.from_numpy(data_dict['res3c_branch2c']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer2_2.conv3.bias'] =  torch.FloatTensor(model_params['skip_layer2_2.conv3.weight'].size()[0]).zero_()
    model_params['skip_layer2_2.bn3.weight'] = torch.from_numpy(data_dict['bn3c_branch2c']['scale']).type(dtype)
    model_params['skip_layer2_2.bn3.bias'] =  torch.from_numpy(data_dict['bn3c_branch2c']['offset']).type(dtype)

    model_params['skip_layer2_3.conv1.weight'] = torch.from_numpy(data_dict['res3d_branch2a']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer2_3.conv1.bias'] =  torch.FloatTensor(model_params['skip_layer2_3.conv1.weight'].size()[0]).zero_()
    model_params['skip_layer2_3.bn1.weight'] = torch.from_numpy(data_dict['bn3d_branch2a']['scale']).type(dtype)
    model_params['skip_layer2_3.bn1.bias'] =  torch.from_numpy(data_dict['bn3d_branch2a']['offset']).type(dtype)

    model_params['skip_layer2_3.conv2.weight'] = torch.from_numpy(data_dict['res3d_branch2b']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer2_3.conv2.bias'] =  torch.FloatTensor(model_params['skip_layer2_3.conv2.weight'].size()[0]).zero_()
    model_params['skip_layer2_3.bn2.weight'] = torch.from_numpy(data_dict['bn3d_branch2b']['scale']).type(dtype)
    model_params['skip_layer2_3.bn2.bias'] =  torch.from_numpy(data_dict['bn3d_branch2b']['offset']).type(dtype)

    model_params['skip_layer2_3.conv3.weight'] = torch.from_numpy(data_dict['res3d_branch2c']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer2_3.conv3.bias'] =  torch.FloatTensor(model_params['skip_layer2_3.conv3.weight'].size()[0]).zero_()
    model_params['skip_layer2_3.bn3.weight'] = torch.from_numpy(data_dict['bn3d_branch2c']['scale']).type(dtype)
    model_params['skip_layer2_3.bn3.bias'] =  torch.from_numpy(data_dict['bn3d_branch2c']['offset']).type(dtype)

    #####

    model_params['proj_layer3.conv4.weight'] = torch.from_numpy(data_dict['res4a_branch1']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['proj_layer3.conv4.bias'] = torch.FloatTensor(model_params['proj_layer3.conv4.weight'].size()[0]).zero_()
    model_params['proj_layer3.bn4.weight'] = torch.from_numpy(data_dict['bn4a_branch1']['scale']).type(dtype)
    model_params['proj_layer3.bn4.bias'] = torch.from_numpy(data_dict['bn4a_branch1']['offset']).type(dtype)

    model_params['proj_layer3.conv1.weight'] = torch.from_numpy(data_dict['res4a_branch2a']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['proj_layer3.conv1.bias'] = torch.FloatTensor(model_params['proj_layer3.conv1.weight'].size()[0]).zero_()
    model_params['proj_layer3.bn1.weight'] = torch.from_numpy(data_dict['bn4a_branch2a']['scale']).type(dtype)
    model_params['proj_layer3.bn1.bias'] = torch.from_numpy(data_dict['bn4a_branch2a']['offset']).type(dtype)

    model_params['proj_layer3.conv2.weight'] = torch.from_numpy(data_dict['res4a_branch2b']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['proj_layer3.conv2.bias'] =  torch.FloatTensor(model_params['proj_layer3.conv2.weight'].size()[0]).zero_()
    model_params['proj_layer3.bn2.weight'] = torch.from_numpy(data_dict['bn4a_branch2b']['scale']).type(dtype)
    model_params['proj_layer3.bn2.bias'] = torch.from_numpy(data_dict['bn4a_branch2b']['offset']).type(dtype)

    model_params['proj_layer3.conv3.weight'] = torch.from_numpy(data_dict['res4a_branch2c']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['proj_layer3.conv3.bias'] =  torch.FloatTensor(model_params['proj_layer3.conv3.weight'].size()[0]).zero_()
    model_params['proj_layer3.bn3.weight'] = torch.from_numpy(data_dict['bn4a_branch2c']['scale']).type(dtype)
    model_params['proj_layer3.bn3.bias'] =  torch.from_numpy(data_dict['bn4a_branch2c']['offset']).type(dtype)

    model_params['skip_layer3_1.conv1.weight'] = torch.from_numpy(data_dict['res4b_branch2a']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer3_1.conv1.bias'] =  torch.FloatTensor(model_params['skip_layer3_1.conv1.weight'].size()[0]).zero_()
    model_params['skip_layer3_1.bn1.weight'] = torch.from_numpy(data_dict['bn4b_branch2a']['scale']).type(dtype)
    model_params['skip_layer3_1.bn1.bias'] =  torch.from_numpy(data_dict['bn4b_branch2a']['offset']).type(dtype)

    model_params['skip_layer3_1.conv2.weight'] = torch.from_numpy(data_dict['res4b_branch2b']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer3_1.conv2.bias'] =  torch.FloatTensor(model_params['skip_layer3_1.conv2.weight'].size()[0]).zero_()
    model_params['skip_layer3_1.bn2.weight'] = torch.from_numpy(data_dict['bn4b_branch2b']['scale']).type(dtype)
    model_params['skip_layer3_1.bn2.bias'] =  torch.from_numpy(data_dict['bn4b_branch2b']['offset']).type(dtype)

    model_params['skip_layer3_1.conv3.weight'] = torch.from_numpy(data_dict['res4b_branch2c']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer3_1.conv3.bias'] =  torch.FloatTensor(model_params['skip_layer3_1.conv3.weight'].size()[0]).zero_()
    model_params['skip_layer3_1.bn3.weight'] = torch.from_numpy(data_dict['bn4b_branch2c']['scale']).type(dtype)
    model_params['skip_layer3_1.bn3.bias'] =  torch.from_numpy(data_dict['bn4b_branch2c']['offset']).type(dtype)

    model_params['skip_layer3_2.conv1.weight'] = torch.from_numpy(data_dict['res4c_branch2a']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer3_2.conv1.bias'] =  torch.FloatTensor(model_params['skip_layer3_2.conv1.weight'].size()[0]).zero_()
    model_params['skip_layer3_2.bn1.weight'] = torch.from_numpy(data_dict['bn4c_branch2a']['scale']).type(dtype)
    model_params['skip_layer3_2.bn1.bias'] =  torch.from_numpy(data_dict['bn4c_branch2a']['offset']).type(dtype)

    model_params['skip_layer3_2.conv2.weight'] = torch.from_numpy(data_dict['res4c_branch2b']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer3_2.conv2.bias'] =  torch.FloatTensor(model_params['skip_layer3_2.conv2.weight'].size()[0]).zero_()
    model_params['skip_layer3_2.bn2.weight'] = torch.from_numpy(data_dict['bn4c_branch2b']['scale']).type(dtype)
    model_params['skip_layer3_2.bn2.bias'] =  torch.from_numpy(data_dict['bn4c_branch2b']['offset']).type(dtype)

    model_params['skip_layer3_2.conv3.weight'] = torch.from_numpy(data_dict['res4c_branch2c']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer3_2.conv3.bias'] =  torch.FloatTensor(model_params['skip_layer3_2.conv3.weight'].size()[0]).zero_()
    model_params['skip_layer3_2.bn3.weight'] = torch.from_numpy(data_dict['bn4c_branch2c']['scale']).type(dtype)
    model_params['skip_layer3_2.bn3.bias'] =  torch.from_numpy(data_dict['bn4c_branch2c']['offset']).type(dtype)

    model_params['skip_layer3_3.conv1.weight'] = torch.from_numpy(data_dict['res4d_branch2a']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer3_3.conv1.bias'] =  torch.FloatTensor(model_params['skip_layer3_3.conv1.weight'].size()[0]).zero_()
    model_params['skip_layer3_3.bn1.weight'] = torch.from_numpy(data_dict['bn4d_branch2a']['scale']).type(dtype)
    model_params['skip_layer3_3.bn1.bias'] =  torch.from_numpy(data_dict['bn4d_branch2a']['offset']).type(dtype)

    model_params['skip_layer3_3.conv2.weight'] = torch.from_numpy(data_dict['res4d_branch2b']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer3_3.conv2.bias'] =  torch.FloatTensor(model_params['skip_layer3_3.conv2.weight'].size()[0]).zero_()
    model_params['skip_layer3_3.bn2.weight'] = torch.from_numpy(data_dict['bn4d_branch2b']['scale']).type(dtype)
    model_params['skip_layer3_3.bn2.bias'] =  torch.from_numpy(data_dict['bn4d_branch2b']['offset']).type(dtype)

    model_params['skip_layer3_3.conv3.weight'] = torch.from_numpy(data_dict['res4d_branch2c']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer3_3.conv3.bias'] =  torch.FloatTensor(model_params['skip_layer3_3.conv3.weight'].size()[0]).zero_()
    model_params['skip_layer3_3.bn3.weight'] = torch.from_numpy(data_dict['bn4d_branch2c']['scale']).type(dtype)
    model_params['skip_layer3_3.bn3.bias'] =  torch.from_numpy(data_dict['bn4d_branch2c']['offset']).type(dtype)

    model_params['skip_layer3_4.conv1.weight'] = torch.from_numpy(data_dict['res4e_branch2a']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer3_4.conv1.bias'] =  torch.FloatTensor(model_params['skip_layer3_4.conv1.weight'].size()[0]).zero_()
    model_params['skip_layer3_4.bn1.weight'] = torch.from_numpy(data_dict['bn4e_branch2a']['scale']).type(dtype)
    model_params['skip_layer3_4.bn1.bias'] =  torch.from_numpy(data_dict['bn4e_branch2a']['offset']).type(dtype)

    model_params['skip_layer3_4.conv2.weight'] = torch.from_numpy(data_dict['res4e_branch2b']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer3_4.conv2.bias'] =  torch.FloatTensor(model_params['skip_layer3_4.conv2.weight'].size()[0]).zero_()
    model_params['skip_layer3_4.bn2.weight'] = torch.from_numpy(data_dict['bn4e_branch2b']['scale']).type(dtype)
    model_params['skip_layer3_4.bn2.bias'] =  torch.from_numpy(data_dict['bn4e_branch2b']['offset']).type(dtype)

    model_params['skip_layer3_4.conv3.weight'] = torch.from_numpy(data_dict['res4e_branch2c']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer3_4.conv3.bias'] =  torch.FloatTensor(model_params['skip_layer3_4.conv3.weight'].size()[0]).zero_()
    model_params['skip_layer3_4.bn3.weight'] = torch.from_numpy(data_dict['bn4e_branch2c']['scale']).type(dtype)
    model_params['skip_layer3_4.bn3.bias'] =  torch.from_numpy(data_dict['bn4e_branch2c']['offset']).type(dtype)

    model_params['skip_layer3_5.conv1.weight'] = torch.from_numpy(data_dict['res4f_branch2a']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer3_5.conv1.bias'] =  torch.FloatTensor(model_params['skip_layer3_5.conv1.weight'].size()[0]).zero_()
    model_params['skip_layer3_5.bn1.weight'] = torch.from_numpy(data_dict['bn4f_branch2a']['scale']).type(dtype)
    model_params['skip_layer3_5.bn1.bias'] =  torch.from_numpy(data_dict['bn4f_branch2a']['offset']).type(dtype)

    model_params['skip_layer3_5.conv2.weight'] = torch.from_numpy(data_dict['res4f_branch2b']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer3_5.conv2.bias'] =  torch.FloatTensor(model_params['skip_layer3_5.conv2.weight'].size()[0]).zero_()
    model_params['skip_layer3_5.bn2.weight'] = torch.from_numpy(data_dict['bn4f_branch2b']['scale']).type(dtype)
    model_params['skip_layer3_5.bn2.bias'] =  torch.from_numpy(data_dict['bn4f_branch2b']['offset']).type(dtype)

    model_params['skip_layer3_5.conv3.weight'] = torch.from_numpy(data_dict['res4f_branch2c']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer3_5.conv3.bias'] =  torch.FloatTensor(model_params['skip_layer3_5.conv3.weight'].size()[0]).zero_()
    model_params['skip_layer3_5.bn3.weight'] = torch.from_numpy(data_dict['bn4f_branch2c']['scale']).type(dtype)
    model_params['skip_layer3_5.bn3.bias'] =  torch.from_numpy(data_dict['bn4f_branch2c']['offset']).type(dtype)

    ####

    model_params['proj_layer4.conv4.weight'] = torch.from_numpy(data_dict['res5a_branch1']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['proj_layer4.conv4.bias'] = torch.FloatTensor(model_params['proj_layer4.conv4.weight'].size()[0]).zero_()
    model_params['proj_layer4.bn4.weight'] = torch.from_numpy(data_dict['bn5a_branch1']['scale']).type(dtype)
    model_params['proj_layer4.bn4.bias'] = torch.from_numpy(data_dict['bn5a_branch1']['offset']).type(dtype)

    model_params['proj_layer4.conv1.weight'] = torch.from_numpy(data_dict['res5a_branch2a']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['proj_layer4.conv1.bias'] = torch.FloatTensor(model_params['proj_layer4.conv1.weight'].size()[0]).zero_()
    model_params['proj_layer4.bn1.weight'] = torch.from_numpy(data_dict['bn5a_branch2a']['scale']).type(dtype)
    model_params['proj_layer4.bn1.bias'] = torch.from_numpy(data_dict['bn5a_branch2a']['offset']).type(dtype)

    model_params['proj_layer4.conv2.weight'] = torch.from_numpy(data_dict['res5a_branch2b']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['proj_layer4.conv2.bias'] =  torch.FloatTensor(model_params['proj_layer4.conv2.weight'].size()[0]).zero_()
    model_params['proj_layer4.bn2.weight'] = torch.from_numpy(data_dict['bn5a_branch2b']['scale']).type(dtype)
    model_params['proj_layer4.bn2.bias'] = torch.from_numpy(data_dict['bn5a_branch2b']['offset']).type(dtype)

    model_params['proj_layer4.conv3.weight'] = torch.from_numpy(data_dict['res5a_branch2c']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['proj_layer4.conv3.bias'] =  torch.FloatTensor(model_params['proj_layer4.conv3.weight'].size()[0]).zero_()
    model_params['proj_layer4.bn3.weight'] = torch.from_numpy(data_dict['bn5a_branch2c']['scale']).type(dtype)
    model_params['proj_layer4.bn3.bias'] =  torch.from_numpy(data_dict['bn5a_branch2c']['offset']).type(dtype)

    model_params['skip_layer4_1.conv1.weight'] = torch.from_numpy(data_dict['res5b_branch2a']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer4_1.conv1.bias'] =  torch.FloatTensor(model_params['skip_layer4_1.conv1.weight'].size()[0]).zero_()
    model_params['skip_layer4_1.bn1.weight'] = torch.from_numpy(data_dict['bn5b_branch2a']['scale']).type(dtype)
    model_params['skip_layer4_1.bn1.bias'] =  torch.from_numpy(data_dict['bn5b_branch2a']['offset']).type(dtype)

    model_params['skip_layer4_1.conv2.weight'] = torch.from_numpy(data_dict['res5b_branch2b']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer4_1.conv2.bias'] =  torch.FloatTensor(model_params['skip_layer4_1.conv2.weight'].size()[0]).zero_()
    model_params['skip_layer4_1.bn2.weight'] = torch.from_numpy(data_dict['bn5b_branch2b']['scale']).type(dtype)
    model_params['skip_layer4_1.bn2.bias'] =  torch.from_numpy(data_dict['bn5b_branch2b']['offset']).type(dtype)

    model_params['skip_layer4_1.conv3.weight'] = torch.from_numpy(data_dict['res5b_branch2c']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer4_1.conv3.bias'] =  torch.FloatTensor(model_params['skip_layer4_1.conv3.weight'].size()[0]).zero_()
    model_params['skip_layer4_1.bn3.weight'] = torch.from_numpy(data_dict['bn5b_branch2c']['scale']).type(dtype)
    model_params['skip_layer4_1.bn3.bias'] =  torch.from_numpy(data_dict['bn5b_branch2c']['offset']).type(dtype)

    model_params['skip_layer4_2.conv1.weight'] = torch.from_numpy(data_dict['res5c_branch2a']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer4_2.conv1.bias'] =  torch.FloatTensor(model_params['skip_layer4_2.conv1.weight'].size()[0]).zero_()
    model_params['skip_layer4_2.bn1.weight'] = torch.from_numpy(data_dict['bn5c_branch2a']['scale']).type(dtype)
    model_params['skip_layer4_2.bn1.bias'] =  torch.from_numpy(data_dict['bn5c_branch2a']['offset']).type(dtype)

    model_params['skip_layer4_2.conv2.weight'] = torch.from_numpy(data_dict['res5c_branch2b']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer4_2.conv2.bias'] =  torch.FloatTensor(model_params['skip_layer4_2.conv2.weight'].size()[0]).zero_()
    model_params['skip_layer4_2.bn2.weight'] = torch.from_numpy(data_dict['bn5c_branch2b']['scale']).type(dtype)
    model_params['skip_layer4_2.bn2.bias'] =  torch.from_numpy(data_dict['bn5c_branch2b']['offset']).type(dtype)

    model_params['skip_layer4_2.conv3.weight'] = torch.from_numpy(data_dict['res5c_branch2c']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['skip_layer4_2.conv3.bias'] =  torch.FloatTensor(model_params['skip_layer4_2.conv3.weight'].size()[0]).zero_()
    model_params['skip_layer4_2.bn3.weight'] = torch.from_numpy(data_dict['bn5c_branch2c']['scale']).type(dtype)
    model_params['skip_layer4_2.bn3.bias'] =  torch.from_numpy(data_dict['bn5c_branch2c']['offset']).type(dtype)


    ####

    model_params['conv2.weight'] = torch.from_numpy(data_dict['layer1']['weights']).type(dtype).permute(3,2,0,1)
    model_params['conv2.bias'] = torch.from_numpy(data_dict['layer1']['biases']).type(dtype)
    model_params['bn2.weight'] = torch.from_numpy(data_dict['layer1_BN']['scale']).type(dtype)
    model_params['bn2.bias'] = torch.from_numpy(data_dict['layer1_BN']['offset']).type(dtype)
    #
    # # #####
    #
    model_params['up_conv1.conv1.weight'] = torch.from_numpy(data_dict['layer2x_br1_ConvA']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv1.conv1.bias'] = torch.from_numpy(data_dict['layer2x_br1_ConvA']['biases']).type(dtype)

    model_params['up_conv1.conv2.weight'] = torch.from_numpy(data_dict['layer2x_br1_ConvB']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv1.conv2.bias'] = torch.from_numpy(data_dict['layer2x_br1_ConvB']['biases']).type(dtype)

    model_params['up_conv1.conv3.weight'] = torch.from_numpy(data_dict['layer2x_br1_ConvC']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv1.conv3.bias'] = torch.from_numpy(data_dict['layer2x_br1_ConvC']['biases']).type(dtype)

    model_params['up_conv1.conv4.weight'] = torch.from_numpy(data_dict['layer2x_br1_ConvD']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv1.conv4.bias'] = torch.from_numpy(data_dict['layer2x_br1_ConvD']['biases']).type(dtype)

    model_params['up_conv1.bn1_1.weight'] = torch.from_numpy(data_dict['layer2x_br1_BN']['scale']).type(dtype)
    model_params['up_conv1.bn1_1.bias'] = torch.from_numpy(data_dict['layer2x_br1_BN']['offset']).type(dtype)

    model_params['up_conv1.conv5.weight'] = torch.from_numpy(data_dict['layer2x_br2_ConvA']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv1.conv5.bias'] = torch.from_numpy(data_dict['layer2x_br2_ConvA']['biases']).type(dtype)

    model_params['up_conv1.conv6.weight'] = torch.from_numpy(data_dict['layer2x_br2_ConvB']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv1.conv6.bias'] = torch.from_numpy(data_dict['layer2x_br2_ConvB']['biases']).type(dtype)

    model_params['up_conv1.conv7.weight'] = torch.from_numpy(data_dict['layer2x_br2_ConvC']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv1.conv7.bias'] = torch.from_numpy(data_dict['layer2x_br2_ConvC']['biases']).type(dtype)

    model_params['up_conv1.conv8.weight'] = torch.from_numpy(data_dict['layer2x_br2_ConvD']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv1.conv8.bias'] = torch.from_numpy(data_dict['layer2x_br2_ConvD']['biases']).type(dtype)

    model_params['up_conv1.bn1_2.weight'] = torch.from_numpy(data_dict['layer2x_br2_BN']['scale']).type(dtype)
    model_params['up_conv1.bn1_2.bias'] = torch.from_numpy(data_dict['layer2x_br2_BN']['offset']).type(dtype)

    model_params['up_conv1.conv9.weight'] = torch.from_numpy(data_dict['layer2x_Conv']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv1.conv9.bias'] = torch.from_numpy(data_dict['layer2x_Conv']['biases']).type(dtype)

    model_params['up_conv1.bn2.weight'] = torch.from_numpy(data_dict['layer2x_BN']['scale']).type(dtype)
    model_params['up_conv1.bn2.bias'] = torch.from_numpy(data_dict['layer2x_BN']['offset']).type(dtype)

    #####

    model_params['up_conv2.conv1.weight'] = torch.from_numpy(data_dict['layer4x_br1_ConvA']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv2.conv1.bias'] = torch.from_numpy(data_dict['layer4x_br1_ConvA']['biases']).type(dtype)

    model_params['up_conv2.conv2.weight'] = torch.from_numpy(data_dict['layer4x_br1_ConvB']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv2.conv2.bias'] = torch.from_numpy(data_dict['layer4x_br1_ConvB']['biases']).type(dtype)

    model_params['up_conv2.conv3.weight'] = torch.from_numpy(data_dict['layer4x_br1_ConvC']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv2.conv3.bias'] = torch.from_numpy(data_dict['layer4x_br1_ConvC']['biases']).type(dtype)

    model_params['up_conv2.conv4.weight'] = torch.from_numpy(data_dict['layer4x_br1_ConvD']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv2.conv4.bias'] = torch.from_numpy(data_dict['layer4x_br1_ConvD']['biases']).type(dtype)

    model_params['up_conv2.bn1_1.weight'] = torch.from_numpy(data_dict['layer4x_br1_BN']['scale']).type(dtype)
    model_params['up_conv2.bn1_1.bias'] = torch.from_numpy(data_dict['layer4x_br1_BN']['offset']).type(dtype)

    model_params['up_conv2.conv5.weight'] = torch.from_numpy(data_dict['layer4x_br2_ConvA']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv2.conv5.bias'] = torch.from_numpy(data_dict['layer4x_br2_ConvA']['biases']).type(dtype)

    model_params['up_conv2.conv6.weight'] = torch.from_numpy(data_dict['layer4x_br2_ConvB']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv2.conv6.bias'] = torch.from_numpy(data_dict['layer4x_br2_ConvB']['biases']).type(dtype)

    model_params['up_conv2.conv7.weight'] = torch.from_numpy(data_dict['layer4x_br2_ConvC']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv2.conv7.bias'] = torch.from_numpy(data_dict['layer4x_br2_ConvC']['biases']).type(dtype)

    model_params['up_conv2.conv8.weight'] = torch.from_numpy(data_dict['layer4x_br2_ConvD']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv2.conv8.bias'] = torch.from_numpy(data_dict['layer4x_br2_ConvD']['biases']).type(dtype)

    model_params['up_conv2.bn1_2.weight'] = torch.from_numpy(data_dict['layer4x_br2_BN']['scale']).type(dtype)
    model_params['up_conv2.bn1_2.bias'] = torch.from_numpy(data_dict['layer4x_br2_BN']['offset']).type(dtype)

    model_params['up_conv2.conv9.weight'] = torch.from_numpy(data_dict['layer4x_Conv']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv2.conv9.bias'] = torch.from_numpy(data_dict['layer4x_Conv']['biases']).type(dtype)

    model_params['up_conv2.bn2.weight'] = torch.from_numpy(data_dict['layer4x_BN']['scale']).type(dtype)
    model_params['up_conv2.bn2.bias'] = torch.from_numpy(data_dict['layer4x_BN']['offset']).type(dtype)

    #####

    model_params['up_conv3.conv1.weight'] = torch.from_numpy(data_dict['layer8x_br1_ConvA']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv3.conv1.bias'] = torch.from_numpy(data_dict['layer8x_br1_ConvA']['biases']).type(dtype)

    model_params['up_conv3.conv2.weight'] = torch.from_numpy(data_dict['layer8x_br1_ConvB']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv3.conv2.bias'] = torch.from_numpy(data_dict['layer8x_br1_ConvB']['biases']).type(dtype)

    model_params['up_conv3.conv3.weight'] = torch.from_numpy(data_dict['layer8x_br1_ConvC']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv3.conv3.bias'] = torch.from_numpy(data_dict['layer8x_br1_ConvC']['biases']).type(dtype)

    model_params['up_conv3.conv4.weight'] = torch.from_numpy(data_dict['layer8x_br1_ConvD']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv3.conv4.bias'] = torch.from_numpy(data_dict['layer8x_br1_ConvD']['biases']).type(dtype)

    model_params['up_conv3.bn1_1.weight'] = torch.from_numpy(data_dict['layer8x_br1_BN']['scale']).type(dtype)
    model_params['up_conv3.bn1_1.bias'] = torch.from_numpy(data_dict['layer8x_br1_BN']['offset']).type(dtype)

    model_params['up_conv3.conv5.weight'] = torch.from_numpy(data_dict['layer8x_br2_ConvA']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv3.conv5.bias'] = torch.from_numpy(data_dict['layer8x_br2_ConvA']['biases']).type(dtype)

    model_params['up_conv3.conv6.weight'] = torch.from_numpy(data_dict['layer8x_br2_ConvB']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv3.conv6.bias'] = torch.from_numpy(data_dict['layer8x_br2_ConvB']['biases']).type(dtype)

    model_params['up_conv3.conv7.weight'] = torch.from_numpy(data_dict['layer8x_br2_ConvC']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv3.conv7.bias'] = torch.from_numpy(data_dict['layer8x_br2_ConvC']['biases']).type(dtype)

    model_params['up_conv3.conv8.weight'] = torch.from_numpy(data_dict['layer8x_br2_ConvD']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv3.conv8.bias'] = torch.from_numpy(data_dict['layer8x_br2_ConvD']['biases']).type(dtype)

    model_params['up_conv3.bn1_2.weight'] = torch.from_numpy(data_dict['layer8x_br2_BN']['scale']).type(dtype)
    model_params['up_conv3.bn1_2.bias'] = torch.from_numpy(data_dict['layer8x_br2_BN']['offset']).type(dtype)

    model_params['up_conv3.conv9.weight'] = torch.from_numpy(data_dict['layer8x_Conv']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv3.conv9.bias'] = torch.from_numpy(data_dict['layer8x_Conv']['biases']).type(dtype)

    model_params['up_conv3.bn2.weight'] = torch.from_numpy(data_dict['layer8x_BN']['scale']).type(dtype)
    model_params['up_conv3.bn2.bias'] = torch.from_numpy(data_dict['layer8x_BN']['offset']).type(dtype)

    #####

    model_params['up_conv4.conv1.weight'] = torch.from_numpy(data_dict['layer16x_br1_ConvA']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv4.conv1.bias'] = torch.from_numpy(data_dict['layer16x_br1_ConvA']['biases']).type(dtype)

    model_params['up_conv4.conv2.weight'] = torch.from_numpy(data_dict['layer16x_br1_ConvB']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv4.conv2.bias'] = torch.from_numpy(data_dict['layer16x_br1_ConvB']['biases']).type(dtype)

    model_params['up_conv4.conv3.weight'] = torch.from_numpy(data_dict['layer16x_br1_ConvC']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv4.conv3.bias'] = torch.from_numpy(data_dict['layer16x_br1_ConvC']['biases']).type(dtype)

    model_params['up_conv4.conv4.weight'] = torch.from_numpy(data_dict['layer16x_br1_ConvD']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv4.conv4.bias'] = torch.from_numpy(data_dict['layer16x_br1_ConvD']['biases']).type(dtype)

    model_params['up_conv4.bn1_1.weight'] = torch.from_numpy(data_dict['layer16x_br1_BN']['scale']).type(dtype)
    model_params['up_conv4.bn1_1.bias'] = torch.from_numpy(data_dict['layer16x_br1_BN']['offset']).type(dtype)

    model_params['up_conv4.conv5.weight'] = torch.from_numpy(data_dict['layer16x_br2_ConvA']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv4.conv5.bias'] = torch.from_numpy(data_dict['layer16x_br2_ConvA']['biases']).type(dtype)

    model_params['up_conv4.conv6.weight'] = torch.from_numpy(data_dict['layer16x_br2_ConvB']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv4.conv6.bias'] = torch.from_numpy(data_dict['layer16x_br2_ConvB']['biases']).type(dtype)

    model_params['up_conv4.conv7.weight'] = torch.from_numpy(data_dict['layer16x_br2_ConvC']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv4.conv7.bias'] = torch.from_numpy(data_dict['layer16x_br2_ConvC']['biases']).type(dtype)

    model_params['up_conv4.conv8.weight'] = torch.from_numpy(data_dict['layer16x_br2_ConvD']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv4.conv8.bias'] = torch.from_numpy(data_dict['layer16x_br2_ConvD']['biases']).type(dtype)

    model_params['up_conv4.bn1_2.weight'] = torch.from_numpy(data_dict['layer16x_br2_BN']['scale']).type(dtype)
    model_params['up_conv4.bn1_2.bias'] = torch.from_numpy(data_dict['layer16x_br2_BN']['offset']).type(dtype)

    model_params['up_conv4.conv9.weight'] = torch.from_numpy(data_dict['layer16x_Conv']['weights']).type(dtype).permute(3,2,0,1)
    model_params['up_conv4.conv9.bias'] = torch.from_numpy(data_dict['layer16x_Conv']['biases']).type(dtype)

    model_params['up_conv4.bn2.weight'] = torch.from_numpy(data_dict['layer16x_BN']['scale']).type(dtype)
    model_params['up_conv4.bn2.bias'] = torch.from_numpy(data_dict['layer16x_BN']['offset']).type(dtype)

    ###

    model_params['conv3.weight'] = torch.from_numpy(data_dict['ConvPred']['weights']).type(dtype).permute(3,2,0,1)
    model_params['conv3.bias'] = torch.from_numpy(data_dict['ConvPred']['biases']).type(dtype)
    # print('Conv3 bias loaded from pretrained: ')
    # print(model_params['conv3.bias'])
    return model_params
