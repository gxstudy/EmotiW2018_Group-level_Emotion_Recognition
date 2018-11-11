# EmotiW2018_Group-level_Emotion_Recognition

#################################Prerequisites##################################
1. ubuntu 16.04
2. Caffe: https://github.com/BVLC/caffe  (with cuda installed)
3. Caffe-Senet (https://github.com/hujie-frank/SENet) 
4. caffe_LargeMargin_Softmax_Loss (https://github.com/wy1iu/LargeMargin_Softmax_Loss) 
5. Matlab R2015a
6. Tensorflow: https://www.tensorflow.org/

#################################Preprocesssing: ##############################

Extract faces using MTCNN (https://github.com/kpzhang93/MTCNN_face_detection_alignment)
  1) Install MTCNN accroding to the above link into folder ./group_2018_code/preprocessing/MTCNN
  2) Put test images in ./group_2018_code/data/test folder
  3) Under ./group_2018_code/preprocessing folder, change corresponding path in file MTCNN_face_detect_align.m and run it using Matlab, the extracted faces will be stored in folder ./group_2018_code/data/test_faces_MTCNN
----------------------------------------------------------------------------------------------------------------------  

Human skeleton features extraction using openpose from https://github.com/CMU-Perceptual-Computing-Lab/openpose
  1) Install openpose from link above, save openpose in folder ./group_2018_code/preprocessing/openpose-master. 
  2) bash extract_skeletons_faces_poses_hands.sh in terminal, the extracted skeleton image will be saved in  ./group_2018_code/data/test_pose_hand_face
----------------------------------------------------------------------------------------------------------------------  

Extract attention paches using a bottom-up attention model (https://github.com/peteanderson80/bottom-up-attention)
  1) Install bottom-up-attention-master according to the above link into folder ./group_2018_code/preprocessing/bottom-up-attention-master
  2) Run python extract_images_test.py in terminal, attention paches will be saved in folder ./group_2018_code/datatest_new_attentions
  3) Install Senet (https://github.com/hujie-frank/SENet).
  4) Download 'SENet-154-deploy.prototxt' and 'SENet.caffemodel' and save them to folder ./group_2018_code/preprocessing/
  5) Modify to use Caffe-Senet in extract_attn_features_test.py, the feature files will be stored in './group_2018_code/data/test_attentions_senet_152_features'  
  6) Read all test features into a data file using build_dataset_test.py
  
#################################Extract predictions ##############################

From ./group_2018_code folder

Extract predictions for vgg faces
  1) cd face_vgg
  2) python classify_vgg_single.py to extract prediction of vgg face classifier.
  3) Predictions based on vgg face classifiers are not good at recognize negative and neutral images compared to scene classifiers, so to lower the weight of the face classifiers on neutral and negative images. Two additional face classifiers, vgg_faces_positive and vgg_faces_unpositive, are extracted using python convert_face_preditions.py
  4) cd ..
  ----------------------------------------------------------------------------------------------------------------------  
  
Extract predictions for senet faces
  1) cd face_senet
  2) python classify_senet_single.py to extract prediction of vgg face classifier.
  3) Predictions based on vgg face classifiers are not good at recognize negative and neutral images compared to scene classifiers, so to lower the weight of the face classifiers on neutral and negative images. Two additional face classifiers, senet50_face_avg_positive and senet50_face_avg_unpositive, are extracted using python convert_face_preditions.py
  4) cd ..
----------------------------------------------------------------------------------------------------------------------  

Extract scene senet predictions
  1) cd scene_senet
  2) python classify_senet_oversampled_largemargin.py
----------------------------------------------------------------------------------------------------------------------  

Extract scene inception predictions
  1) cd scene_inception
  2) python classify_inception_v2.py
----------------------------------------------------------------------------------------------------------------------  

Extract attention predictions
  1) cd attention
  2) python tensorflow_lstm_test.py
  3) cd ..
----------------------------------------------------------------------------------------------------------------------  

Extract skelton predictions
  1) cd skeleton
  2) python tensorflow_lstm_test.py
  3) cd ..
----------------------------------------------------------------------------------------------------------------------  

python fusion.py, the resulting label will be saved in folder 7 - UD-ECE - Group.

----------------------------------------------------------------------------------------------------------------------  

If you use any of the code, please cite the following paper:

@inproceedings{Guo:2018:GER:3242969.3264990,\
 author = {Guo, Xin and Zhu, Bin and Polan\'{\i}a, Luisa F. and Boncelet, Charles and Barner, Kenneth E.},
 title = {Group-Level Emotion Recognition Using Hybrid Deep Models Based on Faces, Scenes, Skeletons and Visual Attentions},
 booktitle = {Proceedings of the 20th ACM International Conference on Multimodal Interaction},
 series = {ICMI '18},
 year = {2018},
 isbn = {978-1-4503-5692-3},
 location = {Boulder, CO, USA},
 pages = {635--639},
 numpages = {5},
 url = {http://doi.acm.org/10.1145/3242969.3264990},
 doi = {10.1145/3242969.3264990},
 acmid = {3264990},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {emotiw 2018, group-level emotion recognition, multi-model, scene understanding, visual attention},
} 

Thanks for your interest. 
