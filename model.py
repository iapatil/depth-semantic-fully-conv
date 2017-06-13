#CHECK
# NOT DONE DROPOUT, AS THEY HAVE USED KEEP_PROB=1
# NOT DONE RELU AFTER THE LAST CONV. IT IS DONE IN THE FIGURE GIVEN IN THE PAPER, BUT NOT IN THEIR CODE
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import matplotlib
import tensorflow as tf
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, d1, d2, skip=False, stride = 1, print_=False):
        super(ResidualBlock, self).__init__()
        self.skip = skip
        self.print_ = print_

        self.conv1 = nn.Conv2d(in_channels, d1, 1, stride = stride,bias = False)
        self.bn1 = nn.BatchNorm2d(d1)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(d1, d1, 3, padding = 1,bias = False)
        self.bn2 = nn.BatchNorm2d(d1)

        self.conv3 = nn.Conv2d(d1, d2, 1,bias = False)
        self.bn3 = nn.BatchNorm2d(d2)

        if not self.skip:
            self.conv4 = nn.Conv2d(in_channels, d2, 1, stride=stride,bias = False)
            self.bn4 = nn.BatchNorm2d(d2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.print_:
            print('')
            output_1 = np.swapaxes(np.squeeze(out.data.numpy()[0,0,:,:]),0,1)
            output_1 = np.uint8(255*(output_1-np.min(output_1))/(np.max(output_1)-np.min(output_1)))

            if self.skip:
                print('SKip layer')
            else:
                print('proj layer')
            fig = plt.figure()
            ii = plt.imshow(output_1, interpolation='nearest')
            fig.colorbar(ii)
            #plt.show()

        if self.skip:
            residual = x
        else:
            residual = self.conv4(x)
            residual = self.bn4(residual)
            #plt.imshow(output_1, cmap='gray')

            if self.print_:
                print('residual')
                output_1 = np.swapaxes(np.squeeze(residual.data.numpy()[0,0,:,:]),0,1)
                output_1 = np.uint8(255*(output_1-np.min(output_1))/(np.max(output_1)-np.min(output_1)))
                fig = plt.figure()
                ii = plt.imshow(output_1, interpolation='nearest')
                fig.colorbar(ii)
                #plt.show()

        out += residual
        out = self.relu(out)
        if not self.skip:
            if self.print_:
                print('final proj layer ')
                output_1 = np.swapaxes(out.data.numpy()[0,0,:,:],0,1)
                output_1 = np.uint8(255*(output_1-np.min(output_1))/(np.max(output_1)-np.min(output_1)))
                fig = plt.figure()
                ii = plt.imshow(output_1, interpolation='nearest')
                fig.colorbar(ii)
                #plt.show()
        return out

class UpProj_Block(nn.Module):
    def __init__(self, in_channels, out_channels, batch_size):
        super(UpProj_Block, self).__init__()
        self.batch_size = batch_size

        self.conv1 = nn.Conv2d(in_channels, out_channels, (3,3))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (2,3))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (3,2))
        self.conv4 = nn.Conv2d(in_channels, out_channels, (2,2))

        self.conv5 = nn.Conv2d(in_channels, out_channels, (3,3))
        self.conv6 = nn.Conv2d(in_channels, out_channels, (2,3))
        self.conv7 = nn.Conv2d(in_channels, out_channels, (3,2))
        self.conv8 = nn.Conv2d(in_channels, out_channels, (2,2))

        self.bn1_1 = nn.BatchNorm2d(out_channels)
        self.bn1_2 = nn.BatchNorm2d(out_channels)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv9 = nn.Conv2d(out_channels, out_channels , 3,padding = 1)

    def prepare_indices(self, before, row, col, after, dims):

        x0, x1, x2, x3 = np.meshgrid(before, row, col, after)
        dtype = torch.cuda.FloatTensor
        x_0 = torch.from_numpy(x0.reshape([-1]))
        x_1 = torch.from_numpy(x1.reshape([-1]))
        x_2 = torch.from_numpy(x2.reshape([-1]))
        x_3 = torch.from_numpy(x3.reshape([-1]))

        linear_indices = x_3 + dims[3] * x_2  + 2 * dims[2] * dims[3] * x_0 * 2 * dims[1] + 2 * dims[2] * dims[3] * x_1
        linear_indices_int = linear_indices.int()
        return linear_indices_int
        # else:
        #     x0, x1, x2, x3 = np.meshgrid(before, row, col, after)

        #     x_0 = tf.Variable(x0.reshape([-1]), name = 'x_0')
        #     x_0.initializer.run()

        #     x_1 = tf.Variable(x1.reshape([-1]), name = 'x_1')
        #     x_1.initializer.run()
        #     x_2 = tf.Variable(x2.reshape([-1]), name = 'x_2')
        #     x_2.initializer.run()
        #     x_3 = tf.Variable(x3.reshape([-1]), name = 'x_3')
        #     x_3.initializer.run()
        #     # print('-'*50)

        #     # print('-'*50)
        #     linear_indices = x_3 + dims[3].value * x_2  + 2 * dims[2].value * dims[3].value * x_0 * 2 * dims[1].value + 2 * dims[2].value * dims[3].value * x_1
        #     #print(linear_indices.eval())
        #     linear_indices_int = tf.to_int32(linear_indices)
        #     #print(linear_indices_int.eval())
        #     #print('-'*50)

        #     #print('-'*50)

        #     return linear_indices_int

    def forward(self, x, BN=True):
        out1 = self.unpool_as_conv(x, id=1)
        out1 = self.conv9(out1)

        if BN:
            out1 = self.bn2(out1)

        out2 = self.unpool_as_conv(x, ReLU=False, id=2)

        out = out1+out2

        out = self.relu(out)
        return out

    def unpool_as_conv(self, x, BN=True, ReLU=True, id=1):
        if(id==1):
            out1 = self.conv1(torch.nn.functional.pad(x,(1,1,1,1)))
            out2 = self.conv2(torch.nn.functional.pad(x,(1,1,1,0)))
            out3 = self.conv3(torch.nn.functional.pad(x,(1,0,1,1)))
            out4 = self.conv4(torch.nn.functional.pad(x,(1,0,1,0)))
        else:
            out1 = self.conv5(torch.nn.functional.pad(x,(1,1,1,1)))
            out2 = self.conv6(torch.nn.functional.pad(x,(1,1,1,0)))
            out3 = self.conv7(torch.nn.functional.pad(x,(1,0,1,1)))
            out4 = self.conv8(torch.nn.functional.pad(x,(1,0,1,0)))

        out1 = out1.permute(0,2,3,1)
        out2 = out2.permute(0,2,3,1)
        out3 = out3.permute(0,2,3,1)
        out4 = out4.permute(0,2,3,1)

        dims = out1.size()
        dim1 = dims[1] * 2
        dim2 = dims[2] * 2

        A_row_indices = range(0, dim1, 2)
        A_col_indices = range(0, dim2, 2)
        B_row_indices = range(1, dim1, 2)
        B_col_indices = range(0, dim2, 2)
        C_row_indices = range(0, dim1, 2)
        C_col_indices = range(1, dim2, 2)
        D_row_indices = range(1, dim1, 2)
        D_col_indices = range(1, dim2, 2)

        all_indices_before = range(int(self.batch_size))
        all_indices_after = range(dims[3])

        A_linear_indices = self.prepare_indices(all_indices_before, A_row_indices, A_col_indices, all_indices_after, dims)
        B_linear_indices = self.prepare_indices(all_indices_before, B_row_indices, B_col_indices, all_indices_after, dims)
        C_linear_indices = self.prepare_indices(all_indices_before, C_row_indices, C_col_indices, all_indices_after, dims)
        D_linear_indices = self.prepare_indices(all_indices_before, D_row_indices, D_col_indices, all_indices_after, dims)

        A_flat = (out1.permute(1, 0, 2, 3)).contiguous().view(-1)
        B_flat = (out2.permute(1, 0, 2, 3)).contiguous().view(-1)
        C_flat = (out3.permute(1, 0, 2, 3)).contiguous().view(-1)
        D_flat = (out4.permute(1, 0, 2, 3)).contiguous().view(-1)

        #print(A_linear_indices)
        size_ = A_linear_indices.size()[0] + B_linear_indices.size()[0]+C_linear_indices.size()[0]+D_linear_indices.size()[0]

        #Y_flat = np.zeros(size_)
        #Y_flat = torch.cuda.FloatTensor(size_).zero_()
        Y_flat = torch.cuda.FloatTensor(size_).zero_()

        # Y_flat.scatter_(0, A_linear_indices.type(torch.cuda.LongTensor).squeeze(),A_flat.data)
        # Y_flat.scatter_(0, B_linear_indices.type(torch.cuda.LongTensor).squeeze(),B_flat.data)
        # Y_flat.scatter_(0, C_linear_indices.type(torch.cuda.LongTensor).squeeze(),C_flat.data)
        # Y_flat.scatter_(0, D_linear_indices.type(torch.cuda.LongTensor).squeeze(),D_flat.data)
        Y_flat.scatter_(0, A_linear_indices.type(torch.cuda.LongTensor).squeeze(),A_flat.data)
        Y_flat.scatter_(0, B_linear_indices.type(torch.cuda.LongTensor).squeeze(),B_flat.data)
        Y_flat.scatter_(0, C_linear_indices.type(torch.cuda.LongTensor).squeeze(),C_flat.data)
        Y_flat.scatter_(0, D_linear_indices.type(torch.cuda.LongTensor).squeeze(),D_flat.data)


        Y = Y_flat.view(-1, dim1, dim2, dims[3])
        Y=Variable(Y.permute(0,3,1,2))

        if(id==1):
            if BN:
                Y = self.bn1_1(Y)
        else:
            if BN:
                Y = self.bn1_2(Y)

        if ReLU:
            Y = self.relu(Y)

        return Y


class Model(nn.Module):
    def __init__(self, block1, block2, batch_size):
        super(Model, self).__init__()
        self.batch_size=batch_size

        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride=2, padding=4)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(3,stride=2)

        self.proj_layer1 = self.make_proj_layer(block1, 64 , d1 = 64, d2 = 256, stride = 1, print_=False)
        self.skip_layer1_1 = self.make_skip_layer(block1, 256, d1 = 64, d2 = 256, stride=1, print_=False)
        self.skip_layer1_2 = self.make_skip_layer(block1, 256, d1 = 64, d2 = 256, stride=1)

        self.proj_layer2 = self.make_proj_layer(block1, 256 , d1 = 128, d2 = 512, stride = 2)
        self.skip_layer2_1 = self.make_skip_layer(block1, 512, d1 = 128, d2 = 512)
        self.skip_layer2_2 = self.make_skip_layer(block1, 512, d1 = 128, d2 = 512)
        self.skip_layer2_3 = self.make_skip_layer(block1, 512, d1 = 128, d2 = 512)

        self.proj_layer3 = self.make_proj_layer(block1, 512 , d1 = 256, d2 = 1024, stride=2)
        self.skip_layer3_1 = self.make_skip_layer(block1, 1024, d1 = 256, d2 = 1024)
        self.skip_layer3_2 = self.make_skip_layer(block1, 1024, d1 = 256, d2 = 1024)
        self.skip_layer3_3 = self.make_skip_layer(block1, 1024, d1 = 256, d2 = 1024)
        self.skip_layer3_4 = self.make_skip_layer(block1, 1024, d1 = 256, d2 = 1024)
        self.skip_layer3_5 = self.make_skip_layer(block1, 1024, d1 = 256, d2 = 1024)

        self.proj_layer4 = self.make_proj_layer(block1, 1024 , d1 = 512, d2 = 2048, stride=2)
        self.skip_layer4_1 = self.make_skip_layer(block1, 2048, d1 = 512, d2 = 2048)
        self.skip_layer4_2 = self.make_skip_layer(block1, 2048, d1 = 512, d2 = 2048)

        self.conv2 = nn.Conv2d(2048,1024,1)
        self.bn2 = nn.BatchNorm2d(1024)

        self.up_conv1 = self.make_up_conv_layer(block2, 1024, 512, self.batch_size)
        self.up_conv2 = self.make_up_conv_layer(block2, 512, 256, self.batch_size)
        self.up_conv3 = self.make_up_conv_layer(block2, 256, 128, self.batch_size)
        self.up_conv4 = self.make_up_conv_layer(block2, 128, 64, self.batch_size)

        self.conv3 = nn.Conv2d(64,1,3, padding=1)
        #self.bn3 = nn.BatchNorm2d(1, affine=False)
        self.up_conv5 = self.make_up_conv_layer(block2,128 ,64 ,self.batch_size)
        self.conv4 = nn.Conv2d(64,48,3,padding=1) #for semantic labeling
        self.bn4 = nn.BatchNorm2d(48)
        self.conv5 = nn.Conv2d(48,38,3,padding=1) #for semantic labeling
        self.bn5 = nn.BatchNorm2d(38)
        self.dropout = nn.Dropout2d(p=1)

        #self.up_conv6 = self.make_up_conv_layer(block2,32 ,16 ,self.batch_size)

        self.upsample = nn.UpsamplingBilinear2d(size = (480,640))

    def make_proj_layer(self, block, in_channels, d1, d2, stride = 1, pad=0, print_=False):
        return block(in_channels, d1, d2, skip=False, stride = stride, print_=print_)

    def make_skip_layer(self, block, in_channels, d1, d2, stride=1, pad=0, print_=False):
        return block(in_channels, d1, d2, skip=True, stride=stride,print_=print_)

    def make_up_conv_layer(self, block, in_channels, out_channels, batch_size):
        return block(in_channels, out_channels, batch_size)

    def forward(self,x_1,x_2):
        out_1 = self.conv1(x_1)
        out = self.bn1(out_1)
        out_1 = self.relu(out_1)
        out_1 = self.max_pool(out_1)
        out_1 = self.proj_layer1(out_1)
        out_1 = self.skip_layer1_1(out_1)
        out_1 = self.skip_layer1_2(out_1)
        out_1 = self.proj_layer2(out_1)
        out_1 = self.skip_layer2_1(out_1)
        out_1 = self.skip_layer2_2(out_1)
        out_1 = self.skip_layer2_3(out_1)
        out_1 = self.proj_layer3(out_1)
        out_1 = self.skip_layer3_1(out_1)
        out_1 = self.skip_layer3_2(out_1)
        out_1 = self.skip_layer3_3(out_1)
        out_1 = self.skip_layer3_4(out_1)
        out_1 = self.skip_layer3_5(out_1)
        out_1 = self.proj_layer4(out_1)
        out_1 = self.skip_layer4_1(out_1)
        out_1 = self.skip_layer4_2(out_1)
        out_1 = self.conv2(out_1)
        #print(out.size())
        out_1 = self.bn2(out_1)
        out_1 = self.up_conv1(out_1)
        out_1 = self.up_conv2(out_1)
        out_1 = self.up_conv3(out_1)
        out_1 = self.up_conv4(out_1)

        temp_out_1 = out_1

        out_2 = self.conv1(x_2)
        out_2 = self.bn1(out_2)
        out_2 = self.relu(out_2)
        out_2 = self.max_pool(out_2)
        out_2 = self.proj_layer1(out_2)
        out_2 = self.skip_layer1_1(out_2)
        out_2 = self.skip_layer1_2(out_2)
        out_2 = self.proj_layer2(out_2)
        out_2 = self.skip_layer2_1(out_2)
        out_2 = self.skip_layer2_2(out_2)
        out_2 = self.skip_layer2_3(out_2)
        out_2 = self.proj_layer3(out_2)
        out_2 = self.skip_layer3_1(out_2)
        out_2 = self.skip_layer3_2(out_2)
        out_2 = self.skip_layer3_3(out_2)
        out_2 = self.skip_layer3_4(out_2)
        out_2 = self.skip_layer3_5(out_2)
        out_2 = self.proj_layer4(out_2)
        out_2 = self.skip_layer4_1(out_2)
        out_2 = self.skip_layer4_2(out_2)
        out_2 = self.conv2(out_2)
        #print(out_2.size())
        out_2 = self.bn2(out_2)
        out_2 = self.up_conv1(out_2)
        out_2 = self.up_conv2(out_2)
        out_2 = self.up_conv3(out_2)
        out_2 = self.up_conv4(out_2)


        out_1 = self.conv3(out_1) #for depth
        #out1 = self.bn3(out1)
        out_1 = self.upsample(out_1)
        #print(out_1.size())
        #print('Output 1 size: ',out_1.size())


        #print(out.size())# (1L, 64L, 128L, 160L)


        # semantic labels
        out_2 = torch.cat((temp_out_1,out_2),dim = 1)
        #print('Just after concatenation: ',out_2.size())
        out_2 = self.up_conv5(out_2)
        #print(out2.size())
        out_2 = self.conv4(out_2)
        out_2 = self.bn4(out_2)
        out_2 = self.relu(out_2)
        out_2 = self.conv5(out_2)
        out_2 = self.bn5(out_2)
        out_2 = self.relu(out_2)


        #out2 = self.up_conv6(out2)
        #print(out2.size())  #(1L, 16L, 512L, 640L)
        out_2 = self.upsample(out_2)
        #print('Output 2 size: ',out_2.size())
        #assert(False)
        #print(out2.size()) #(1L, 32L, 480L, 640L)
        # out2 = (1,37,480,640)
        #assert(False)
        return out_1, out_2
