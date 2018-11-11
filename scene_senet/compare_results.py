import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
np.set_printoptions(threshold=np.nan)


pred_label = np.load('./scene_senet_preds_largemargin (copy).npy')
#pred_label = np.argmax(preds ,1)
print pred_label.shape
print pred_label

truth = np.load('./scene_senet_preds_largemargin.npy')
print truth
print np.sum(pred_label==truth)
#face_weight_senet50_face_train_val_1_iter_60000 71.80
#face_avg_senet50_face_train_val_1_iter_60000 71.40


#face_vgg_weight_train_val_1_iter_6000 72.66
#face_vgg_avg_train_val_1_iter_6000 72.06

#scene_senet_preds  69.28
#scene_senet_preds_largemargin, 71.70

#scene_inception_preds 70.07
#scene_inception_preds_largemargin 62.7

#face_pose_hand_senet_preds  65.32
#face_pose_hand_senet_preds_largemargin 62.93

#test_attention_2_combine_probs 64.96
