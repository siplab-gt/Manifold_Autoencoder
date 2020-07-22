from __future__ import division

import random
import scipy.misc
import numpy as np

import cv2
import glob
import scipy as sp
import os
import scipy.io as sio

    
def load_mnist(data_type,y_dim=10):
        data_dir = os.path.join("./Data/", 'mnist')
        print(os.path.join(data_dir,'train-images-idx3-ubyte'))
        fd = open(os.path.join(data_dir,'train-images.idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)
    
        fd = open(os.path.join(data_dir,'train-labels.idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)
    
        fd = open(os.path.join(data_dir,'t10k-images.idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)
    
        fd = open(os.path.join(data_dir,'t10k-labels.idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)
    
        trY = np.asarray(trY)
        teY = np.asarray(teY)
        
        if data_type == "train":
            X = trX[0:50000,:,:,:]
            y = trY[0:50000].astype(np.int)
        elif data_type == "test":
            X = teX
            y = teY.astype(np.int)
        elif data_type == "val":
            X = trX[50000:60000,:,:,:]
            y = trY[50000:60000].astype(np.int)
            
        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        y_vec = np.zeros((len(y), y_dim), dtype=np.float)
        for i, label in enumerate(y):
          y_vec[i,y[i]] = 1.0
        
        return X/255.,y_vec 


def transform_image(input_data,labels,class_transform,input_size,maxAng):
    batch_size = input_data.shape[0]
    input_h = input_size
    input_w = input_size
    c_dim = input_data.shape[3]
    imgOut = np.zeros((batch_size,input_h,input_w,c_dim))
    angOut = np.zeros((batch_size))
    for k in range(0,batch_size):
        imgTemp = np.pad(input_data[k,:,:,0],((2,2),(2,2)),'constant',constant_values=((0, 0),(0, 0)))
        classUse = np.where(labels[k,:] != 0)[0]
        img_h = imgTemp.shape[0]
        img_w = imgTemp.shape[1]
        
        class_check = np.in1d(classUse,class_transform)
        if class_check: 
            angle_use = np.random.randint(low = 0,high = maxAng, size = 1)
        else:
            angle_use = 0
        angOut[k] = angle_use
        M = cv2.getRotationMatrix2D((img_h/2,img_w/2),angle_use,1)
        imgOut[k,:,:,:] = np.expand_dims(cv2.warpAffine(imgTemp,M,(img_h,img_w)),axis=3)
        
    return imgOut,angOut

def transform_image_pair(input_data,labels,class_transform,input_size,maxAng,angDiff):
    batch_size = input_data.shape[0]
    input_h = input_size
    input_w = input_size
    c_dim = input_data.shape[3]
    imgOut_0 = np.zeros((batch_size,input_h,input_w,c_dim))
    imgOut_1 = np.zeros((batch_size,input_h,input_w,c_dim))
    angOut = np.zeros((batch_size,2))
    for k in range(0,batch_size):
        imgTemp = np.pad(input_data[k,:,:,0],((2,2),(2,2)),'constant',constant_values=((0, 0),(0, 0)))
        classUse = np.where(labels[k,:] != 0)[0]
        img_h = imgTemp.shape[0]
        img_w = imgTemp.shape[1]
        
        class_check = np.in1d(classUse,class_transform)
        if class_check: 
            angle_use_0 = np.random.randint(low = 0,high = maxAng, size = 1)
            angle_use_1 = angle_use_0 + angDiff
            #angle_use_1 = angle_use_0 + np.random.randint(low = 1,high = angDiff, size = 1)
        else:
            angle_use_0 = 0
            angle_use_1 = 0
        angOut[k,0] = angle_use_0
        angOut[k,1] = angle_use_1
        M_0 = cv2.getRotationMatrix2D((img_h/2,img_w/2),angle_use_0,1)
        imgOut_0[k,:,:,:] = np.expand_dims(cv2.warpAffine(imgTemp,M_0,(img_h,img_w)),axis=3)
        M_1 = cv2.getRotationMatrix2D((img_h/2,img_w/2),angle_use_1,1)
        imgOut_1[k,:,:,:] = np.expand_dims(cv2.warpAffine(imgTemp,M_1,(img_h,img_w)),axis=3)
        
    return imgOut_0,imgOut_1,angOut
    
def transform_image_specific(input_data,input_size,rotAng):
    batch_size = input_data.shape[0]
    input_h = input_size
    input_w = input_size
    c_dim = input_data.shape[3]
    imgOut = np.zeros((batch_size,input_h,input_w,c_dim))
    angOut = np.zeros((batch_size))
    for k in range(0,batch_size):
        imgTemp = np.pad(input_data[k,:,:,0],((2,2),(2,2)),'constant',constant_values=((0, 0),(0, 0)))
        img_h = imgTemp.shape[0]
        img_w = imgTemp.shape[1]
        
        angOut[k] = rotAng
        M = cv2.getRotationMatrix2D((img_h/2,img_w/2),rotAng,1)
        imgOut[k,:,:,:] = np.expand_dims(cv2.warpAffine(imgTemp,M,(img_h,img_w)),axis=3)
        
    return imgOut,angOut
    
def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
  return (images+1.)/2.

def imsave(images, size, path):
  return sp.misc.imsave(path, merge(images, size))

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h * size[0], w * size[1], 3))
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image
  return img

    
def get_neighbor_batch(input_data,distAll,batch_size):
    N = input_data.shape[1]  
    numEx = input_data.shape[0]
    x0 = np.zeros((batch_size,N))
    x1 = np.zeros((batch_size,N))
    for k in range(0,batch_size):
        x0Idx = random.randint(0,numEx-1)
        x0[k,:] = input_data[x0Idx,:]
        
        distPoss = distAll[x0Idx,:]
        sortIdx = np.argsort(distPoss)
        sampUse = random.randint(5,15); 
        idxUse = sortIdx[sampUse]
        x1[k,:] = input_data[idxUse,:]
              
        
    return x0,x1

def load_gait_data_pair(folderUse,batch_size,feat_size,frameDiff,data_type):
    normFile = fileList = sorted(glob.glob(folderUse + '/meanMaxData_*.mat'))  
    norm_info = sio.loadmat(normFile[0])
    channel_max = norm_info['channels_max_condense']
    channel_mean = norm_info['channels_mean_condense']
    
    if data_type =='train':
        fileList = sorted(glob.glob(folderUse + '/train/*.mat'))  
    elif data_type == 'val':
        fileList = sorted(glob.glob(folderUse + '/val/*.mat'))  
    elif data_type == 'test':
        fileList = sorted(glob.glob(folderUse + '/test/*.mat'))  
    numSeq = len(fileList)
    
    x0 = np.zeros((batch_size,feat_size))
    x1 = np.zeros((batch_size,feat_size))
    frameUseStart = np.zeros((batch_size))
    frameUseEnd = np.zeros((batch_size))
    for k in range(0,batch_size):
        seqUse = random.randint(0,numSeq-1)
        seqFile = fileList[seqUse]
        seq_info = sio.loadmat(seqFile)
        feat = seq_info['channels_feat']
        numFrame = feat.shape[0]
        startFrame = random.randint(0,numFrame-frameDiff-1)
        endFrame = startFrame+ frameDiff
        
        x0[k,:] = np.divide(feat[startFrame,:]-channel_mean,channel_max)
        x1[k,:] = np.divide(feat[endFrame,:]-channel_mean,channel_max)
        frameUseStart[k] = startFrame
        frameUseEnd[k] = startFrame
    return x0,x1,frameUseStart,frameUseEnd
        
def load_gait_data(folderUse,batch_size,feat_size,data_type):
    normFile = sorted(glob.glob(folderUse + '/meanMaxData_*.mat'))  
    norm_info = sio.loadmat(normFile[0])
    channel_max = norm_info['channels_max_condense']
    channel_mean = norm_info['channels_mean_condense']
    
    if data_type =='train':
        fileList = sorted(glob.glob(folderUse + '/train/*.mat'))  
    elif data_type == 'val':
        fileList = sorted(glob.glob(folderUse + '/val/*.mat'))  
    elif data_type == 'test':
        fileList = sorted(glob.glob(folderUse + '/test/*.mat'))  
    numSeq = len(fileList)
    
    x0 = np.zeros((batch_size,feat_size))
    frameUse = np.zeros((batch_size))
    for k in range(0,batch_size):
        seqUse = random.randint(0,numSeq-1)
        seqFile = fileList[seqUse]
        seq_info = sio.loadmat(seqFile)

        feat = seq_info['channels_feat']
        numFrame = feat.shape[0]
        startFrame = random.randint(0,numFrame-1)
    
        x0[k,:] = np.divide(feat[startFrame,:]-channel_mean,channel_max)
        frameUse[k] = startFrame
    return x0,frameUse       

