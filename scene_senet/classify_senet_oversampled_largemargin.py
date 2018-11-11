import sys
sys.path.append('/home/xin/caffe/distribute/python')
import caffe
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


model_def = 'SE-ResNet-50_deploy_largemargin.prototxt'

model_weights ="group_senet_face.caffemodel"

mu = np.array([103, 108, 120])
net = caffe.Classifier(model_def, model_weights,
                       mean=mu,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

image_dir = '../data/test'
lcount=0  
for ename in sorted(os.listdir(image_dir)):
    lcount += 1
print lcount  

scene_senet_preds = np.zeros((lcount,3))
features = np.zeros((lcount,2048))  
num_count = 0
for iname in sorted(os.listdir(image_dir)):
    print iname
    image_name = os.path.join(image_dir,iname)
    input_image = caffe.io.load_image(image_name)
    if len(input_image.shape)>3:
        print input_image.shape
        converted_image = input_image[0]
        print converted_image.shape
        
    else:
        converted_image=input_image
    img = caffe.io.resize_image(converted_image, (256,256), interp_order=3 )
    prediction = net.predict([img])  # predict takes any number of images, and formats them for the Caffe net automatically
    print 'predicted class:', prediction[0].argmax()
    scene_senet_preds[num_count]=prediction[0]
    features[num_count]= net.blobs['pool5/7x7_s1'].data[0].reshape((2048,))
    print scene_senet_preds[num_count]
    print features[num_count]
    num_count += 1  
np.save('scene_senet_preds.npy',scene_senet_preds)
#np.save('scene_senet_features_largemargin.npy',features)
