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
sys.path.append('/home/xin/caffe_Senet/distribute/python')
import caffe
caffe.set_mode_gpu()
#caffe.set_device(1)
import Image
import ImageOps

def classify_video(image_input, net, transformer, is_flow):
  
  input_im = caffe.io.load_image(image_input)
  if (input_im.shape[0] != 256 or input_im.shape[1] != 256):
    input_im = caffe.io.resize_image(input_im, (256,256))
  net.blobs['data'].data[...] = transformer.preprocess('data', input_im)
  out = net.forward()
  predict_image = out['prob']
  features = net.blobs['pool5/7x7_s1'].data[0].reshape((2048,))
  return predict_image, features

def initialize_transformer(is_flow):
  shape = (1, 3, 224, 224)
  transformer = caffe.io.Transformer({'data': shape})
  channel_mean = np.zeros((3,224,224))

  image_mean = [6, 8, 6]
  for channel_index, mean_val in enumerate(image_mean):
    channel_mean[channel_index, ...] = mean_val
  transformer.set_mean('data', channel_mean)

  transformer.set_raw_scale('data', 255)
  transformer.set_channel_swap('data', (2, 1, 0))
  transformer.set_transpose('data', (2, 0, 1))
  #transformer.set_is_flow('data', is_flow)
  return transformer



def get_face_prob():
  print 'hello'
  transformer_RGB = initialize_transformer(False)
  deploy_file = 'senet50_ft_group_deploy.prototxt'
  RGB_model = 'skeleton_model.caffemodel'
  RGB_net =  caffe.Net(deploy_file, RGB_model, caffe.TEST)

  val_dir = '../data/test'
  pose_dir = '../data/test_pose_hand_face'


  lcount = 0
  for ename in sorted(os.listdir(val_dir)):
      lcount += 1
  print lcount

  face_pose_hand_senet_preds = np.zeros((lcount,3))
  features_total =  np.zeros((lcount,2048))

  num_count = 0
  for iname in sorted(os.listdir(val_dir)):

      print iname
      fname = iname.split('.')
      postfix_name = fname[-1]
      truc_name = iname[0:-(len(postfix_name)+1)]
      pose_name = truc_name + '_rendered.png'
      print pose_name
      image_name = os.path.join(pose_dir,pose_name)             
      face_pose_hand_senet_preds[num_count,:], features_total[num_count,:] = classify_video(image_name, RGB_net, transformer_RGB, False) 
      print face_pose_hand_senet_preds[num_count,:] 
      print features_total[num_count,:]

      num_count += 1  
         
  del RGB_net
  np.save('face_pose_hand_senet_preds',face_pose_hand_senet_preds)


get_face_prob()

    
