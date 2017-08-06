# Fully Convolutional Network for Depth Estimation and Semantic Segmentation

Course Project by Ishan Patil, Yokila Arora, Thao Nguyen for CS231N (Spring 2017)

Our model is based on Laina, Iro, et al. "Deeper depth prediction with fully convolutional residual networks." 3D Vision (3DV), 2016 Fourth International Conference on. IEEE, 2016.

We extend the above model for semantic segmentation (per-pixel class labelling) task for 37(+1 others) most frequent classes in the NYU Depth Dataset V2. More particularly, we transfer learn using the pre-trained weights (for the depth estimation task) from this model on an extension of the model as shown below. Finally, our model is able to output both the depth map and the semantic segmentation of the input (single) RGB image. 

![Our model](https://github.com/iapatil/depth-semantic-fully-conv/blob/master/model_fig.png)


We provide the implementation in PyTorch, while the original implementation (https://github.com/iro-cp/FCRN-DepthPrediction) is in TensorFlow and MatConvNet frameworks. 

To accomplish data augmentation (applying co-transforms on both input and target images simultaneously), we have taken code (flow_transforms.py) from https://github.com/ClementPinard/FlowNetPytorch and modified it for our task, as PyTorch itself currently doesn't support this feature. 

Before running the main file (main.py) to train the model (in model.py) for semantic segmentation task, perform the following steps -

1. Download the NYU Depth Dataset V2 Labelled Dataset (Can be downloaded from http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
2. Download the pre-trained TensorFlow weights as a .npy file for a part of the model from Laina et al. from http://campar.in.tum.de/files/rupprecht/depthpred/NYU_ResNet-UpProj.npy.
3. With the above two files in the same directory as the code, run data_process.py to preprocess the ground truth depth maps and semantic segmentation maps for the labeled dataset.

More details can be found in our report (http://cs231n.stanford.edu/reports/2017/pdfs/209.pdf) 




