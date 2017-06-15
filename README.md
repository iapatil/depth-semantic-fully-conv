# Fully Convolutional Network for Depth Estimation and Semantic Segmentation

Course Project by Ishan Patil, Yokila Arora, Thao Nguyen for CS231N (Spring 2017)

Our model is based on Laina, Iro, et al. "Deeper depth prediction with fully convolutional residual networks." 3D Vision (3DV), 2016 Fourth International Conference on. IEEE, 2016.

We extend the above for the semantic segmentation (per-pixel class labelling) task for 38 most occuring classes in the NYU Depth Dataset V2. More particularly, we transfer learn with the pre-trained weights (for the depth estimation task) from this model on an extension of the model as described in our report. Finally, our extened model is able to output both a depth map and a semantic segmentation of an input single RGB image. 

We also provide the implementation in PyTorch, while the original implementation (https://github.com/iro-cp/FCRN-DepthPrediction) is in TensorFlow and MatConvnet. 

To accomplish data augmentation (applying co-transforms on both input and target images simultaneously, we have used code from https://github.com/ClementPinard/FlowNetPytorch as PyTorch itself currently doesn't support this feature. 

Before running the main file (main.py) to train the model (in model.py) for semantic segmentation task, the below two files must be present in the same directory-

1. NYU Depth Dataset V2 Labelled Dataset (Can be downloaded from http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
2. Pre-trained TensorFlow weights as a .npy file for a part of the model from Laina et al. from http://campar.in.tum.de/files/rupprecht/depthpred/NYU_ResNet-UpProj.npy .



