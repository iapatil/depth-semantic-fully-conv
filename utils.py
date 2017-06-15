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
from eval_depth import *
import warnings
from torch.optim.optimizer import Optimizer
import shutil

weights_file = "NYU_ResNet-UpProj.npy"
color = np.array([(0,0,0),(0,0,255),(255,0,0),(0,255,0),(255,255,0),(255,0,255), #magenta
        (192,192,192), #silver
        (128,128,128), #gray
        (128,0,0) ,#maroon
        (128,128,0) ,#olive
        (0,128,0) ,#green
        (128,0,128), # purple
        (0,128,128) , # teal
        (65,105,225) , #royal blue
        (255,250,205) , #lemon chiffon
        (255,20,147) , #deep pink
        (218,112,214) , #orchid]
        (135,206,250) , #light sky blue
        (127,255,212),  #aqua marine
        (0,255,127) , #spring green
        (255,215,0) , #gold
        (165,42,42) , #brown
        (148,0,211) , #violet
        (210,105,30) , # chocolate
        (244,164,96),  # sandy brown
        (240,255,240),  #honeydew
        (112,128,144), (64,224,208) ,(100,149,237) ,(30,144,255),(221,160,221),
        (205,133,63),(255,240,245),(255,255,240),(255,165,0),(255,160,122),(205,92,92),(240,248,255)])


def run_epoch(model, loss_fn, loader, optimizer, dtype):
  """
  Train the model for one epoch.
  """
  # Set the model to training mode
  #model.train()
  running_loss = 0
  count = 0
  for x,z,y in loader:

    x_var = Variable(x.type(dtype))
    z_var = Variable(z.type(dtype))
    y_var = Variable(y.type(dtype).long())
    m = nn.LogSoftmax()

    pred_depth,pred_labels = model(x_var,z_var)
    y_var = y_var.squeeze()

    loss = loss_fn(m(pred_labels), y_var)
    running_loss += loss.data.cpu().numpy()
    count += 1

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  return running_loss/count


def check_accuracy(model, loader,epoch, dtype, visualize = False):
  """
  Check the accuracy of the model.
  """

  num_correct, num_samples = 0, 0
  for x,z,y in loader:

    x_var = Variable(x.type(dtype),volatile=True)
    z_var = Variable(z.type(dtype),volatile=True)


    pred_depth,pred_labels = model(x_var,z_var)

    _,preds = pred_labels.data.cpu().max(1)
    if visualize == True:
        
        #Save the input RGB image, Ground truth depth map, Ground Truth Coloured Semantic Segmentation Map, 
        #Predicted Coloured Semantic Segmentation Map, Predicted Depth Map for one image in the current batch
        
        input_rgb_image = x_var[0].data.permute(1,2,0).cpu().numpy().astype(np.uint8)
        plt.imsave('input_rgb_epoch_{}.png'.format(epoch),input_rgb_image)

        input_gt_depth_image =  z_var[0].data.permute(1,2,0).cpu().numpy().astype(np.uint8)
        plt.imsave('input_gt_depth_epoch_{}.png'.format(epoch),input_gt_depth_image)

        colored_gt_label = color[y[0].squeeze().cpu().numpy().astype(int)].astype(np.uint8)
        plt.imsave('gt_label_epoch_{}.png'.format(epoch),colored_gt_label)

        colored_pred_label = color[preds[0].squeeze().cpu().numpy().astype(int)].astype(np.uint8)
        plt.imsave('pred_label_epoch_{}.png'.format(epoch),colored_pred_label)

        pred_depth_image = pred_depth[0].data.squeeze().cpu().numpy().astype(np.uint8)
        plt.imsave('pred_depth_epoch_{}.png'.format(epoch),pred_depth_image,cmap = "gray")

    # Computing pixel-wise accuracy    
    num_correct += (preds.long() == y.long()).sum()
    num_samples += preds.numel()

  acc = float(num_correct) / num_samples
  return acc

def plot_performance_curves(loss_history,train_acc_history,val_acc_history,epoch_history,train_on,batch_size,num_epochs,resumed_file):
        plt.figure()
        plt.plot(np.array(epoch_history),np.array(loss_history))
        plt.ylabel('Loss')
        plt.xlabel('Number of Epochs')
        plt.title('Loss history for training model on {} examples with batch size of {}'.format(train_on,batch_size))
        if resumed_file == False:
            plt.savefig('loss_plot_train_on_{}_batch_size_{}.png'.format(train_on,batch_size))
        else:
            plt.savefig('loss_plot_train_on_{}_batch_size_{}_resumed.png'.format(train_on,batch_size))
        plt.figure()
        plt.plot(np.array(epoch_history),np.array(train_acc_history),label = 'Training accuracy')
        plt.plot(np.array(epoch_history),np.array(val_acc_history), label = 'Validation accuracy')
        plt.title('Accuracy history for training model on {} examples with batch size of {}'.format(train_on,batch_size))
        plt.ylabel('Accuracy')
        plt.xlabel('Number of Epochs')
        plt.legend()
        if resumed_file == False:
            plt.savefig('acc_plots_train_on_{}_batch_size_{}.png'.format(train_on,batch_size))
        else:
            plt.savefig('acc_plots_train_on_{}_batch_size_{}_resumed.png'.format(train_on,batch_size))

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
