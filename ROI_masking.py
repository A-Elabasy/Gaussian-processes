
"""
Created on Fri Jun 21 16:11:02 2019

@author: yhosni18
"""
import numpy as np
from numpy import load
import nibabel as nib
import sys
import math
import matplotlib.pyplot as plt
import time
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import VotingClassifier
from sklearn import tree

#uploading the data of the ROIs
Amgdala_Right_con=load('/data/fmri/Folder/Amygdala_con_Right.npz')['masked_voxels']
Amgdala_Right_ad=load('/data/fmri/Folder/Amygdala_AD_Right.npz')['masked_voxels']
Amgdala_Left_con=load('/data/fmri/Folder/Amygdala_con_Left.npz')['masked_voxels']
Amgdala_Left_ad=load('/data/fmri/Folder/Amygdala_AD_Left.npz')['masked_voxels']

Putamen_Right_con=load('/data/fmri/Folder/Putamen_CON_Right.npz')['masked_voxels']
Putamen_Right_ad=load('/data/fmri/Folder/Putamen_AD_Right.npz')['masked_voxels']
Putamen_Left_con=load('/data/fmri/Folder/Putamen_CON_Left.npz')['masked_voxels']
Putamen_Left_ad=load('/data/fmri/Folder/Putamen_AD_Left.npz')['masked_voxels']

Thalamus_Right_con=load('/data/fmri/Folder/Right_thalamus_con.npz')['masked_voxels']
Thalamus_Right_ad=load('/data/fmri/Folder/Right_thalamus_ad.npz')['masked_voxels']
Thalamus_Left_con=load('/data/fmri/Folder/Left_thalamus_con.npz')['masked_voxels']
Thalamus_Left_ad=load('/data/fmri/Folder/Left_thalamus_ad.npz')['masked_voxels']

Hippocampus_con=load('/data/fmri/cv/hippocampus/hipp_con.npz')['masked_voxels']
Hippocampus_ad=load('/data/fmri/cv/hippocampus/hipp_ad.npz')['masked_voxels']

lateral_ventricles_con=load('/data/fmri/cv/hippocampus/vntr_con.npz')['masked_voxels']
lateral_ventricles_ad=load('/data/fmri/cv/hippocampus/vntr_ad.npz')['masked_voxels']

cerebellum_con=load('/data/fmri/Folder/cerebellum_con.npz')['masked_voxels']
cerebellum_ad=load('/data/fmri/Folder/cerebellum_AD.npz')['masked_voxels']

frontal_lobe_con=load('/data/fmri/Folder/frontal_lobe_con.npz')['masked_voxels']
frontal_lobe_ad=load('/data/fmri/Folder/frontal_lobe_AD.npz')['masked_voxels']

frontal_pole_con=load('/data/fmri/Folder/frontal_pole_con.npz')['masked_voxels']
frontal_pole_ad=load('/data/fmri/Folder/frontal_pole_AD.npz')['masked_voxels']

# Removing the zeros for the cerebellum data
imp_mean = SimpleImputer(missing_values=0, strategy='mean')
imp_mean.fit(cerebellum_con)
cerebellum_con=imp_mean.transform(cerebellum_con)
imp_mean.fit(cerebellum_ad)
imp_mean.fit(cerebellum_ad)
cerebellum_ad=imp_mean.transform(cerebellum_ad)

# concatenating the data
con_data=np.vstack((Putamen_Right_con,Putamen_Left_con,Thalamus_Right_con,Thalamus_Left_con,lateral_ventricles_con))
ad_data=np.vstack((Putamen_Right_ad,Putamen_Left_ad,Thalamus_Right_ad,Thalamus_Left_ad,lateral_ventricles_ad))

#data standardization 

scaler = StandardScaler()
scaler.fit(con_data)
con_data=scaler.transform(con_data)

scaler = StandardScaler()
scaler.fit(ad_data)
ad_data=scaler.transform(ad_data)


lst=np.hstack((ad_data,con_data)).T
print(np.shape(lst))
labels_ad=np.ones((np.shape(ad_data)[1],1))
labels_con=np.zeros((np.shape(con_data)[1],1))
print(np.shape(labels_ad))
print(np.shape(labels_con))
labels=np.vstack((labels_ad,labels_con))
print(np.shape(labels))


data_train, data_test, labels_train, labels_test = train_test_split(lst, labels, test_size=0.20, random_state=42)


k=5
fold_length=math.floor(np.shape(data_train)[0]/k)
#out_array = np.zeros(in_shape, dtype=float)
feature_weight=np.zeros(np.shape(data_train)[1])
threshold=3# act as a hyperparameter
for i in range(k):
 if i < k-1:
  data_train_validation=data_train[i*fold_length:(i+1)*fold_length,:]
  data_train_train=np.delete(data_train,np.s_[i*fold_length:(i+1)*fold_length],axis=0)
  labels_train_validation=labels_train[i*fold_length:(i+1)*fold_length,:]  
  labels_train_train=np.delete(labels_train,np.s_[i*fold_length:(i+1)*fold_length],axis=0)
  
 else:
  data_train_validation=data_train[i*fold_length:-1,:]
  data_train_train=np.delete(data_train,np.s_[i*fold_length:-1],axis=0)
  labels_train_validation=labels_train[i*fold_length:-1,:]  
  labels_train_train=np.delete(labels_train,np.s_[i*fold_length:-1],axis=0)
  

 
 svc = SVC(kernel="linear")
 
 rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(5),
              scoring='accuracy',n_jobs=-1,min_features_to_select=600)
 rfecv.fit(data_train_train,labels_train_train )       
 indecies=rfecv.get_support( indices=True)
 feature_weight[indecies]=feature_weight[indecies]+1
 print("Optimal number of features : %d" % rfecv.n_features_)   
 
# choose inly the important features
mask=np.array(feature_weight>threshold,dtype=int) 
selected_features_train_data=data_train*mask
selected_features_data_test=data_test*mask

#data standardization 
scaler = StandardScaler()
scaler.fit(selected_features_train_data)
stand_selected_features_train_data=scaler.transform(selected_features_train_data)

scaler = StandardScaler()
scaler.fit(selected_features_data_test)
stand_selected_features_data_test=scaler.transform(selected_features_data_test)





# Gussian process classifer
kernel = 1.0 * RBF(np.sum(mask))
gpc = GaussianProcessClassifier(kernel=kernel,n_restarts_optimizer=5,random_state=None,multi_class="one_vs_rest",max_iter_predict=170,n_jobs=-1)
gpc=gpc.fit(selected_features_train_data,labels_train)



#SVM classifer 
clf = svm.SVC(kernel='linear',gamma='scale', decision_function_shape='ovo')
clf.fit(selected_features_train_data,labels_train)
training_pred=clf.predict(selected_features_train_data)
test_pred=clf.predict(selected_features_data_test)

#decison tree classifer 
tree_clf = tree.DecisionTreeClassifier()
tree_clf = tree_clf.fit(selected_features_train_data,labels_train)
test_tree_pred=clf.predict(selected_features_data_test)

# ensamble classifer 
estimators=[('Gussian_process',gpc),('svm classifer',clf),('Decision tree',tree_clf)]
ensemble = VotingClassifier(estimators, voting='hard',)
ensemble.fit(selected_features_train_data,labels_train)

'''
#calculating AUC 
probs = clf.predict_proba(selected_features_data_test) 
probs = probs[:, 1]  
auc_SVM = roc_auc_score(labels_test, probs)  

#calculating AUC 
probs = ensemble.predict_proba(selected_features_data_test) 
probs = probs[:, 1]  
auc_ensemble = roc_auc_score(labels_test, probs)  
'''
#calculating AUC 
probs = gpc.predict_proba(selected_features_data_test) 
probs = probs[:, 1]  
auc_GP = roc_auc_score(labels_test, probs)  


print('')
print('training accuracy GP classifer:',gpc.score(selected_features_train_data,labels_train))
print("training accuracy SVM classifer:",metrics.accuracy_score(training_pred, labels_train))

print('test accuracy GP classifer:',gpc.score(selected_features_data_test, labels_test))
print('test accuracy SVM classifer:',metrics.accuracy_score(test_pred,labels_test ))
print('test accuracy tree classifer:',metrics.accuracy_score(test_tree_pred,labels_test ))
print('test accuracy ensamble classifer:',ensemble.score(selected_features_data_test, labels_test))

print('AUC GP classifer: %.2f' % auc_GP)
#print('AUC SVM classifer: %.2f' % auc_SVM)
#print('AUC ensemble  classifer: %.2f' % auc_ensemble)

print('F1 score with GP classifer',f1_score(labels_test, gpc.predict(selected_features_data_test), average='weighted')) 
print('F1 score with SVM classifer ',f1_score(labels_test, test_pred, average='weighted')) 
print('F1 score with ensamble classifer ',f1_score(labels_test, ensemble.predict(selected_features_data_test), average='weighted')) 


print('the number of vosxels remained after thresholding',np.sum(mask))
