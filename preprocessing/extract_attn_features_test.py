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
sys.path.append('/home/xin/caffe-Senet/distribute/python')
import caffe
caffe.set_mode_gpu()
#caffe.set_device(1)
import Image
import ImageOps

def initialize_transformer(is_flow):
  shape = (1, 3, 224, 224)
  transformer = caffe.io.Transformer({'data': shape})
  channel_mean = np.zeros((3,224,224))

  image_mean = [107, 95, 91]
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
  lstm_model = 'SENet-154-deploy.prototxt'
  RGB_lstm = 'SENet.caffemodel'
  #RGB_lstm = 'group_inception_v2_3_iter_2000.caffemodel'
  RGB_lstm_net =  caffe.Net(lstm_model, RGB_lstm, caffe.TEST)
  val_dir = '../data/test_new_attentions'
  val_face_dir = val_dir
  print 'hello'
   
  #classes = ['Negative', 'Neutral', 'Positive']
  classes = ['test']
  save_dir = '../data/test_attentions_senet_152_features'  
    
  for ite, subclass in enumerate(classes):
    for iname in sorted(os.listdir(os.path.join(val_face_dir, subclass))):
        save_folder = os.path.join(save_dir,subclass, iname)
        if not os.path.exists(save_folder):
           os.makedirs(save_folder)
   
        for image_name in sorted(os.listdir(os.path.join(val_face_dir, subclass,iname))):
            print image_name
            save_name = image_name.split('.')[-2] +'.npy'
            save_img_path = os.path.join(save_folder,save_name)
            if os.path.isfile(save_img_path):
                print "exist"
                continue

            single_image = os.path.join(val_face_dir, subclass,iname,image_name)
            input_im = caffe.io.load_image(single_image)
	    if (input_im.shape[0] != 256 or input_im.shape[1] != 256):
	        input_im = caffe.io.resize_image(input_im, (256,256))
            RGB_lstm_net.blobs['data'].data[...] = transformer_RGB.preprocess('data', input_im)
            out = RGB_lstm_net.forward()
            senet_152_features = RGB_lstm_net.blobs['pool5/7x7_s1'].data[0]

            print senet_152_features.shape            
            print save_name
            np.save(save_img_path,senet_152_features)

  del RGB_lstm_net

get_face_prob()

    
