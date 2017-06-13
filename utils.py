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


def normalize_tensor(input_):

    mean_scores = -1*torch.mean(input_)
    std_scores = torch.std(input_)
    input_ = torch.add(input_,mean_scores.repeat(input_.size()))
    input_ = torch.div(input_,std_scores.repeat(input_.size()))

    return input_


def run_epoch(model, loss_fn, loader, optimizer, dtype):
  """
  Train the model for one epoch.
  """
  # Set the model to training mode
  #model.train()
  running_loss = 0
  count = 0
  for x,z,y in loader:
    # The DataLoader produces Torch Tensors, so we need to cast them to the
    # correct datatype and wrap them in Variables.
    #
    # Note that the labels should be a torch.LongTensor on CPU and a
    # torch.cuda.LongTensor on GPU; to accomplish this we first cast to dtype
    # (either torch.FloatTensor or torch.cuda.FloatTensor) and then cast to
    # long; this ensures that y has the correct type in both cases.
    x_var = Variable(x.type(dtype))
    z_var = Variable(z.type(dtype))
    y_var = Variable(y.type(dtype).long())
    m = nn.LogSoftmax()
    # Run the model forward to compute scores and loss.
    pred_depth,pred_labels = model(x_var,z_var)
    y_var = y_var.squeeze()
    #print(torch.max(y_var))
    loss = loss_fn(m(pred_labels), y_var)
    running_loss += loss.data.cpu().numpy()
    count += 1
    # for param_group in optimizer.param_groups:
    #     print(param_group['lr'])

    # Run the model backward and take a step using the optimizer.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  return running_loss/count

def visualize_model(rgb_image,target_depth,predicted_depth):

    plt.subplot(1, 3, 1)
    plt.imshow(np.uint8(255.0*rgb_image))
    plt.title('Input RGB image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(np.uint8(255.0*predicted_depth),cmap= "gray")
    plt.title('Predicted depth image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(np.uint8(255.0*target_depth),cmap = "gray")
    plt.title('Ground truth depth image')

    plt.axis('off')

def check_accuracy(model, loader,epoch, dtype, visualize = False,get_top_5 = False):
  """
  Check the accuracy of the model.
  """
  # Set the model to eval mode
  #model.eval()

  #count = 0
  num_correct, num_samples = 0, 0
  for x,z,y in loader:
    # Cast the image data to the correct type and wrap it in a Variable. At
    # test-time when we do not need to compute gradients, marking the Variable
    # as volatile can reduce memory usage and slightly improve speed.
    x_var = Variable(x.type(dtype),volatile=True)
    z_var = Variable(z.type(dtype),volatile=True)

    # Run the model forward, and compare the argmax score with the ground-truth
    # category.
    pred_depth,pred_labels = model(x_var,z_var)

    _,preds = pred_labels.data.cpu().max(1)

    if get_top_5:
        for i in range(x_var.size()[0]):
            num_correct = (preds[i].long() == y[i].long()).sum()
            num_total = preds[i].numel()
            im_acc = num_correct/num_total
            if im_acc > 0.4:
                input_rgb_image = x_var[i].data.permute(1,2,0).cpu().numpy().astype(np.uint8)
                plt.imsave('input_rgb_acc_{}.png'.format(im_acc),input_rgb_image)

                input_gt_depth_image =  z_var[i].data.permute(1,2,0).cpu().numpy().astype(np.uint8)
                plt.imsave('input_gt_depth_acc_{}.png'.format(im_acc),input_gt_depth_image)

                colored_gt_label = color[y[i].squeeze().cpu().numpy().astype(int)].astype(np.uint8)
                plt.imsave('gt_label_acc_{}.png'.format(im_acc),colored_gt_label)

                colored_pred_label = color[preds[i].squeeze().cpu().numpy().astype(int)].astype(np.uint8)
                plt.imsave('pred_label_acc_{}.png'.format(im_acc),colored_pred_label)

                pred_depth_image = pred_depth[i].data.squeeze().cpu().numpy().astype(np.uint8)
                plt.imsave('pred_depth_acc_{}.png'.format(im_acc),pred_depth_image,cmap = "gray")

    # if visualize == True:
    #     ##save input
    #     input_rgb_image = x_var[0].data.permute(1,2,0).cpu().numpy().astype(np.uint8)
    #     plt.imsave('input_rgb_epoch_{}.png'.format(epoch),input_rgb_image)
    #
    #     input_gt_depth_image =  z_var[0].data.permute(1,2,0).cpu().numpy().astype(np.uint8)
    #     plt.imsave('input_gt_depth_epoch_{}.png'.format(epoch),input_gt_depth_image)
    #
    #     colored_gt_label = color[y[0].squeeze().cpu().numpy().astype(int)].astype(np.uint8)
    #     plt.imsave('gt_label_epoch_{}.png'.format(epoch),colored_gt_label)
    #
    #     colored_pred_label = color[preds[0].squeeze().cpu().numpy().astype(int)].astype(np.uint8)
    #     plt.imsave('pred_label_epoch_{}.png'.format(epoch),colored_pred_label)
    #
    #     pred_depth_image = pred_depth[0].data.squeeze().cpu().numpy().astype(np.uint8)
    #     plt.imsave('pred_depth_epoch_{}.png'.format(epoch),pred_depth_image,cmap = "gray")
    #
    #
    # num_correct += (preds.long() == y.long()).sum()
    # num_samples += preds.numel()
  assert(False)

  # Return the fraction of datapoints that were correctly classified.
  acc = float(num_correct) / num_samples
  return acc

class ReduceLROnPlateau(object):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        factor: factor by which the learning rate will
            be reduced. new_lr = lr * factor
        patience: number of epochs with no improvement
            after which learning rate will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {min, max}. In `min` mode,
            lr will be reduced when the quantity
            monitored has stopped decreasing; in `max`
            mode it will be reduced when the quantity
            monitored has stopped increasing.
        epsilon: threshold for measuring the new optimum,
            to only focus on significant changes.
        cooldown: number of epochs to wait before resuming
            normal operation after lr has been reduced.
        min_lr: lower bound on the learning rate.


    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_acc, val_loss = validate(...)
        >>>     scheduler.step(val_loss, epoch)
    """

    def __init__(self, optimizer, mode='min', factor=0.05, patience= 4,
                 verbose=0, epsilon=1e-4, cooldown=0, min_lr=0):
        super(ReduceLROnPlateau, self).__init__()

        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau '
                             'does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.monitor_op = None
        self.wait = 0
        self.best = 0
        self.mode = mode
        assert isinstance(optimizer, Optimizer)
        self.optimizer = optimizer
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['min', 'max']:
            raise RuntimeError('Learning Rate Plateau Reducing mode %s is unknown!')
        if self.mode == 'min' :
            self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.epsilon)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0
        self.lr_epsilon = self.min_lr * 1e-4

    def reset(self):
        self._reset()

    def step(self, metrics, epoch):
        current = metrics
        if current is None:
            warnings.warn('Learning Rate Plateau Reducing requires metrics available!', RuntimeWarning)
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                if self.wait >= self.patience:
                    for param_group in self.optimizer.param_groups:
                        old_lr = float(param_group['lr'])
                        if old_lr > self.min_lr + self.lr_epsilon:
                            new_lr = old_lr * self.factor
                            new_lr = max(new_lr, self.min_lr)
                            param_group['lr'] = new_lr
                            if self.verbose > 0:
                                print('\nEpoch %05d: reducing learning rate to %s.' % (epoch, new_lr))
                            self.cooldown_counter = self.cooldown
                            self.wait = 0
                self.wait += 1

    def in_cooldown(self):
        return self.cooldown_counter > 0


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
