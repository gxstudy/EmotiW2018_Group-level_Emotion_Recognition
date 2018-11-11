# -*- coding: utf-8 -*-
"""
Created on Fri May 20 18:34:56 2016

@author: fanyin
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:49:18 2016

@author: fanyin
"""

import os
import numpy as np
from sklearn import svm
import sys
import string
import re
from pylab import * 
import glob
import sys
sys.path.append('/home/xin/caffe_LargeMargin_Softmax_Loss/python')
import caffe
caffe.set_mode_gpu()


def classify_video(frames, net, transformer, is_flow):
  
  #clip_length = 16
  #offset = 8
  input_images = []
  num_frames = len(frames)
  predict_image = np.zeros((num_frames,3))
  image_weight = np.zeros((num_frames,))
  fc6_features = np.zeros((num_frames, 2048))
  index = 0
  for im in frames:   
    input_im = caffe.io.load_image(im)
    if (input_im.shape[0] != 227 or input_im.shape[1] != 227):
      input_im = caffe.io.resize_image(input_im, (227,227))
   
    net.blobs['data'].data[...] = transformer.preprocess('data', input_im)
    out = net.forward()
    fc6_features[index] = net.blobs['pool5/7x7_s1'].data[0].reshape((2048,))
    predict_image[index] = out['probs']
    sp = re.split('/',im)
    image_name = re.split('.jpg',sp[-1])
    image_dim = re.split('_',image_name[0])
    image_weight[index] = int(image_dim[-1])*int(image_dim[-2])
    index = index+1
  average_predict_prob = np.mean(predict_image,0)
  fc6_mean = np.mean(fc6_features,0)
  print average_predict_prob
  print fc6_mean
  weighted_predict_prob = np.zeros((3,))  
  norm_facter = np.sum(image_weight) 
  #print norm_facter
  for i in range(num_frames):
    weighted_predict_prob = weighted_predict_prob + image_weight[i]*predict_image[i]/norm_facter
  print weighted_predict_prob
  return average_predict_prob, weighted_predict_prob, fc6_mean
    

def initialize_transformer(is_flow):
  shape = (1, 3, 224, 224)
  transformer = caffe.io.Transformer({'data': shape})
  channel_mean = np.zeros((3,224,224))

  #transformer.set_mean('data', np.load('group_train_face_aligned_lmdb_mean.npy').mean(1).mean(1))
  image_mean =  [86, 96, 124]
  for channel_index, mean_val in enumerate(image_mean):
    channel_mean[channel_index, ...] = mean_val
  transformer.set_mean('data', channel_mean)
  transformer.set_raw_scale('data', 255)
  transformer.set_channel_swap('data', (2, 1, 0))
  transformer.set_transpose('data', (2, 0, 1))
  #transformer.set_is_flow('data', is_flow)
  return transformer



def get_face_prob():
  print 'Let us play!'
  transformer_RGB = initialize_transformer(False)
  deploy_file = 'senet50_ft_group_deploy.prototxt'
  RGB_model = 'senet50_face.caffemodel'
  RGB_net =  caffe.Net(deploy_file, RGB_model, caffe.TEST)
  val_face_dir = '../data/test_faces_MTCNN'
  val_dir = '../data/test'

  lcount=0
  for ename in sorted(os.listdir(val_dir)):
    lcount += 1
  print lcount

  avg_pred_faces = np.zeros((lcount,3))
  weighted_pred_faces = np.zeros((lcount,3))
  fc6_whole= np.zeros((lcount,2048))


    
  num_count = 0
  for iname in sorted(os.listdir(val_dir)):
    print iname
    fname = iname.split('.')
    postfix_name = fname[-1]
    fname3 = iname[0:-(len(postfix_name)+1)]
    if os.path.exists(os.path.join(val_face_dir,fname3)): 
        RGB_frames = glob.glob('%s/%s/*.jpg' %(val_face_dir,fname3))

        average_predict_prob, weighed_predict_probs, fc6_feautre = classify_video(RGB_frames, RGB_net, transformer_RGB, False) 
        avg_pred_faces[num_count,:] = average_predict_prob
        weighted_pred_faces[num_count,:] = weighed_predict_probs
        fc6_whole[num_count,:] = fc6_feautre
        num_count += 1  
    else:
        print('No faces detected in this image')
        avg_pred_faces[num_count,:] = [0,0,0]
        weighted_pred_faces[num_count,:] = [0,0,0]
        fc6_whole[num_count,:]=np.zeros((1,2048))
        num_count += 1     
        print iname 

  del RGB_net
  np.save('senet50_face.npy', weighted_pred_faces)
 


get_face_prob()

    
