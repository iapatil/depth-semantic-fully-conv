import numpy as np
import random
import os
import PIL
from PIL import Image
from model import *
from torch.autograd import Variable
from weights import load_weights
from scipy import misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision.transforms as transforms
from torchvision import utils
import flow_transforms
import torch
from nyu_dataset_loader import *
from loss_function import *
from utils import *

#saved_predictions_dir = 'depth_predictions/'

#input_rgb_images_dir = 'nyu_datasets/input/'
input_rgb_images_dir = 'data/nyu_datasets_changed/input/'
target_depth_images_dir = 'data/nyu_datasets_changed/target_depths/'
target_labels_images_dir = 'data/nyu_datasets_changed/labels_38/'

dtype = torch.cuda.FloatTensor ## change when running on GPU

train_on = 1000 #make this 1000, val=449
val_on = 100
test_on = 50
batch_size= 16

print('Loading images...')

NUM_TRAIN = 1000
NUM_VAL = 300
NUM_TEST = 149

listing = random.sample(os.listdir(input_rgb_images_dir),1449)
train_listing = listing[:min(NUM_TRAIN,train_on)]
val_listing = listing[NUM_TRAIN:min(NUM_VAL+NUM_TRAIN,val_on+NUM_TRAIN)]
test_listing = listing[NUM_VAL+NUM_TRAIN:min(NUM_VAL+NUM_TRAIN+NUM_TEST,test_on+NUM_VAL+NUM_TRAIN)]

data_dir = (input_rgb_images_dir,target_depth_images_dir,target_labels_images_dir)

input_transform = transforms.Compose([flow_transforms.Scale(228),flow_transforms.ArrayToTensor()])
target_depth_transform = transforms.Compose([flow_transforms.Scale_Single(228),flow_transforms.ArrayToTensor()])
target_labels_transform = transforms.Compose([flow_transforms.ArrayToTensor()])


##Apply this transform on input, ground truth depth images and labeled images

co_transform=flow_transforms.Compose([
            flow_transforms.RandomCrop((480,640)),
            flow_transforms.RandomHorizontalFlip()
        ])


##Splitting in train, val and test sets [No data augmentation on val and test, only on train]

train_dataset = ListDataset(data_dir,train_listing,input_transform,target_depth_transform,\
                            target_labels_transform,co_transform)

val_dataset = ListDataset(data_dir,val_listing,input_transform,target_depth_transform,\
                            target_labels_transform)

test_dataset = ListDataset(data_dir,test_listing,input_transform,target_depth_transform,\
                            target_labels_transform)

print("Loading data...")
train_loader = data_utils.DataLoader(train_dataset,batch_size,shuffle = True, drop_last=True)
val_loader = data_utils.DataLoader(val_dataset,batch_size,shuffle = True, drop_last=True)
test_loader = data_utils.DataLoader(test_dataset,batch_size,shuffle = True, drop_last=True)


model = Model(ResidualBlock, UpProj_Block, batch_size)
model.type(dtype)

# Loading pretrained weights
model.load_state_dict(load_weights(model,weights_file,dtype))


loss_fn = torch.nn.NLLLoss2d().type(dtype)

# Uncomment When transfer learning the model parameters of the semantic segmentation branch
# for name, param in model.named_parameters():
#     if name.startswith('up_conv5') or\
#        name.startswith('conv4') or\
#        name.startswith('bn4') or\
#        name.startswith('conv5') or\
#        name.startswith('bn5'):
#         param.requires_grad = True
#     else:
#         param.requires_grad = False


# Uncomment when fine tuning the model by allowing backpropogation through all layers of the model
for param in model.parameters():
    param.requires_grad = True


# framework to define different learning rate for different set of parameters in the model
## reduce learning rate when loss states to plateau
#scheduler = ReduceLROnPlateau(optimizer, 'min') # set up scheduler

# Train the entire model for a few more epochs, checking accuracy on the
# train and validation sets after each epoch.

loss_history = []
num_epochs = 50
train_acc_history = []
val_acc_history = []
epoch_history = []
best_val_acc = 0
# Depends on whether transfer learning (0.05 [Determined by overfitting on validation set]) or fine-tuning (1e-5)
learning_rate = 1e-5
start_epoch = 0

resume_from_file = True
resume_file = 'model_best.pth.tar'
resumed_file = False


#print('Transfer learning the weights for uprojection blocks and last conv layer ...')
#print('on {} training examples with a batch size of {} for {} epochs'.format(train_on,batch_size,num_epochs))

if resume_from_file:
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file)
            start_epoch = checkpoint['epoch']
            best_val_acc = checkpoint['best_val_acc']
            model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_file, checkpoint['epoch']))
            resumed_file = True
        else:
            print("=> no checkpoint found at '{}'".format(resume_file))

for epoch in range(num_epochs):

    # Uncomment When transfer learning the model parameters of the semantic segmentation branch

    # optimizer = torch.optim.Adam([
    #                 # {'params': model.up_conv5.parameters()},
    #                 # {'params': model.conv4.parameters()},
    #                 # {'params': model.bn4.parameters()},
    #                 # {'params': model.bn5.parameters()},
    #                 # {'params': model.conv5.parameters()}
    #                 #{'params': model.classifier.parameters(), 'lr': 1e-3}
    #             ], lr = learning_rate)

    # Uncomment when fine tuning the model by allowing backpropogation through all layers of the model
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
    print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
    print('Learning Rate for this epoch: {}'.format(learning_rate))
    loss = run_epoch(model, loss_fn, train_loader, optimizer, dtype)
    #scheduler.step(loss, epoch) # update lr if loss plateaus (by a factor of 0.05, wait time is 4 epochs)
    print('Loss for epoch {} : {}'.format(start_epoch+epoch+1,loss))
    if epoch % 2 == 0 or epoch == num_epochs-1:
        if epoch % 8 == 0 and epoch != 0:
            learning_rate = learning_rate*0.5
        loss_history.append(loss)
        epoch_history.append(start_epoch+epoch+1)
        train_acc = check_accuracy(model,train_loader,start_epoch+epoch,dtype,visualize = False)
        train_acc_history.append(train_acc)
        print('Train accuracy for epoch {}: {} '.format(start_epoch+epoch + 1,train_acc))
        val_acc = check_accuracy(model,val_loader,start_epoch+epoch,dtype,visualize = True)
        val_acc_history.append(val_acc)
        print('Validation accuracy for epoch {} : {} '.format(start_epoch+epoch + 1,val_acc))
        plot_performance_curves(loss_history,train_acc_history,val_acc_history,epoch_history,\
                                train_on,batch_size,num_epochs,resumed_file)
        is_best = val_acc > best_val_acc
        best_val_acc = max(val_acc,best_val_acc)
        save_checkpoint({
            'epoch': start_epoch+epoch + 1,
            'state_dict': model.state_dict(),
            'best_val_acc': best_val_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


#test_acc = check_accuracy(model,test_loader,start_epoch+epoch,dtype,visualize = True)
#print('Test accuracy: ', val_acc)
