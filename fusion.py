import os
import numpy as np
import re 

class_dict = {}
class_dict[0] = 'Negative'
class_dict[1] = 'Neutral'
class_dict[2] = 'Positive'
np.set_printoptions(threshold=np.nan)
def compute_fusion(pred_1, pred_2, pred_3,pred_4, pred_5, pred_6, pred_7, pred_8, p1, p2,p3, p4, p5, p6, p7):
    print 1-p1-p2-p3-p4-p5-p6-p7
    fusion_pred = p1*pred_1 + p2*pred_2 + p3*pred_3 + p4*pred_4 + p5*pred_5 + p6*pred_6 + p7*pred_7 +(1-p1-p2-p3-p4-p5-p6-p7)*pred_8
    return fusion_pred 
 

#load scene inception
scene_inception_preds = np.load('./scene_inception/scene_inception_preds.npy')
print scene_inception_preds.shape
#print scene_inception_preds


#load skeleton with hand senet
face_pose_hand_senet_preds= np.load('./skeleton/face_pose_hand_senet_preds.npy')
print face_pose_hand_senet_preds.shape
#print face_pose_hand_senet_preds

#load face vgg
vgg_face= np.load('./face_vgg/vgg_face.npy')
print vgg_face.shape
#print vgg_face

# load face vgg unpositive
vgg_face_unpositive= np.load('./face_vgg/vgg_face_unpositive.npy')
print vgg_face_unpositive.shape
#print face_vgg_avg_MTCNN_largeMargin_5_iter_1500_unpositive

# load face vgg positive
attention= np.load('./attention/test_attention_2_combine_probs.npy')
print attention.shape
#print attention

#load face senet
senet50_face= np.load('./face_senet/senet50_face.npy')
print senet50_face.shape
#print senet50_face_avg

# load face senet unpositive
senet50_face_unpositive= np.load('./face_senet/senet50_face_unpositive.npy')
print senet50_face_unpositive.shape
#print senet50_face_avg_unpositive


# load image senet
scene_senet_preds= np.load('./scene_senet/scene_senet_preds_largemargin.npy')
print scene_senet_preds.shape
#print scene_senet_preds





pred_combine = compute_fusion(scene_inception_preds, face_pose_hand_senet_preds,vgg_face,vgg_face_unpositive, attention,senet50_face, senet50_face_unpositive, scene_senet_preds, 0.05, 0.1, 0.05, 0.15, 0.15, 0.15, 0.3)
print pred_combine.shape
print pred_combine
pred_label = np.argmax(pred_combine ,1)
print pred_label.shape
print pred_label
np.save('submit_7.npy', pred_label)    

save_dir = './7 - UD-ECE - Group'
os.mkdir(save_dir)

image_dir = './data/test'
icount=0
for iname in sorted(os.listdir(image_dir)):
    print iname
    print icount
    fname = iname.split('.')
    postfix_name = fname[-1]
    truc_name = iname[0:-(len(postfix_name)+1)]
    save_name = truc_name + '.txt'
    print pred_combine[icount]
    label = pred_label[icount]
    print label
    fp = open(os.path.join(save_dir, save_name), 'w')
    fp.write(class_dict[label])
    fp.close()   
    icount +=1
