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
import Image
import ImageOps



print 'Start'


   
classes ='test'
val_dir = '..data/test'
save_dir = '..data/test_attentions_senet_152_features'  
save_name = '../data/test_inorder.npy'
num_files = 0    
for iname in sorted(os.listdir(val_dir)):
    num_files=num_files+1
print num_files
dataset = np.zeros((num_files, 16, 2048))
#label = np.zeros((num_files, 3))
file_index=0
 
for iname in sorted(os.listdir(val_dir)):        
    fname = iname.split('.')[-2]
    print fname
    file_folder = os.path.join(save_dir,classes, fname)
    sequence_index=0
    for npy_name in sorted(os.listdir(file_folder)):

        print npy_name
        features = np.load(os.path.join(file_folder, npy_name))
        dataset[file_index,sequence_index,:]= features.reshape((2048,))
        print dataset[file_index,sequence_index,:]            
        sequence_index=sequence_index+1
    file_index=file_index+1
np.save(save_name,dataset)





    
