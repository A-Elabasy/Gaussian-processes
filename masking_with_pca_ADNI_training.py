#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 15:06:29 2019

@author: yhosni18
"""

import numpy as np
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
#from sklearn import decomposition
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import decomposition 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score 


# load the ADNI files for testing 
in_file_name_con = '/data/fmri/cv/CV_ADNI_CON.nii.gz'
in_img_con = nib.load(in_file_name_con)
in_array_con = in_img_con.get_fdata()
in_array_con=in_array_con.T
in_shape_con = in_array_con.shape

in_file_name_AD = '/data/fmri/cv/CV_ADNI_AD.nii.gz'
in_img_AD = nib.load(in_file_name_AD)
in_array_AD = in_img_AD.get_fdata()
in_array_AD=in_array_AD.T
in_shape_AD = in_array_AD.shape

ADNI_con_labels=np.ones((in_shape_con[0],1))
ADNI_AD_labels=np.zeros((in_shape_AD[0],1))

in_array_con_reshaped=np.reshape(in_array_con,(in_shape_con[0],-1))
in_array_AD_reshaped=np.reshape(in_array_AD,(in_shape_AD[0],-1))


ADNI_data=np.vstack((in_array_con_reshaped,in_array_AD_reshaped))
ADNI_labels=np.vstack((ADNI_con_labels,ADNI_AD_labels))

pca = decomposition.PCA()
pca.fit(ADNI_data)
ADNI_data = pca.transform(ADNI_data)

shufle_indicies=np.random.permutation(np.shape(ADNI_data)[0])
ADNI_data=ADNI_data[shufle_indicies,:]
ADNI_labels=ADNI_labels[shufle_indicies,:]
ADNI_data_train=ADNI_data

ADNI_labels_train=ADNI_labels

k=5
fold_length=math.floor(np.shape(ADNI_data_train)[0]/k)
out_array = np.zeros(in_shape, dtype=float)
feature_weight=np.zeros(np.shape(ADNI_data_train)[1])
threshold=4
for i in range(k):
 if i < k-1:
  ADNI_data_train_validation=ADNI_data_train[i*fold_length:(i+1)*fold_length,:]
  ADNI_data_train_train=np.delete(ADNI_data_train,np.s_[i*fold_length:(i+1)*fold_length],axis=0)
  ADNI_labels_train_validation=ADNI_labels_train[i*fold_length:(i+1)*fold_length,:]  
  ADNI_labels_train_train=np.delete(ADNI_labels_train,np.s_[i*fold_length:(i+1)*fold_length],axis=0)
  
 else:
  ADNI_data_train_validation=ADNI_data_train[i*fold_length:-1,:]
  ADNI_data_train_train=np.delete(ADNI_data_train,np.s_[i*fold_length:-1],axis=0)
  ADNI_labels_train_validation=ADNI_labels_train[i*fold_length:-1,:]  
  ADNI_labels_train_train=np.delete(ADNI_labels_train,np.s_[i*fold_length:-1],axis=0)
  
  start = time.time()

 # this could be used if PCA is not used as it will takes long time to excute all the data 
 #ADNI_data_train_feature_selection=ADNI_data_train[0:8,:]
 #ADNI_labels_train_feature_selection=ADNI_labels_train[0:8,:]

 
 svc = SVC(kernel="linear")

 
 rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy',n_jobs=-1)
 rfecv.fit(ADNI_data_train_train,ADNI_labels_train_train )       

 print("Optimal number of features : %d" % rfecv.n_features_) 

 indecies=rfecv.get_support( indices=True)
 feature_weight[indecies]=feature_weight[indecies]+1


 #rvec.ranking()#

 end = time.time()
 print(end - start)
 
mask=np.array(feature_weight>threshold,dtype=int) 
#remained_feature_indices=np.where(mask==1)


selected_features_ADNI_data=ADNI_data_train*mask
#selected_features_ADNI_data_test=ADNI_data_test*mask

print(selected_features_ADNI_data.shape)
print(ADNI_labels_train.shape)
#print(selected_features_ADNI_data_test.shape)
#print(ADNI_labels_test.shape)


#the gussian process classification
kernel = 1.0 * RBF(214)

gpc = GaussianProcessClassifier(kernel=kernel,n_restarts_optimizer=5,random_state=None,multi_class="one_vs_rest",max_iter_predict=100,n_jobs=-1)
gpc=gpc.fit(selected_features_ADNI_data,ADNI_labels_train)

print('')
print('accuracy on trainingset:',gpc.score(selected_features_ADNI_data,ADNI_labels_train))
#print('accuracy on testset of the hospital:',gpc.score(selected_features_ADNI_data_test, Oulu_labels_test)) 




in_file_name_Oulu = '/data/fmri/cv/CV_OULU.nii.gz'
mask_file_name = '/data/fmri/cv/4mm_brain_mask_bin.nii.gz'

# Load subjects for training 
# this data is from oulu hospital
in_img = nib.load(in_file_name_Oulu)
in_array = in_img.get_fdata()
in_array=in_array.T
in_shape = in_array.shape

Oulu_control_labels=np.ones((24,1))
Oulu_AD_labels=np.zeros((17,1))
Oulu_labels=np.vstack((Oulu_control_labels,Oulu_AD_labels))



Oulu_data=np.reshape(in_array,(41,-1))

Oulu_data = pca.transform(Oulu_data)
shufle_indicies=np.random.permutation(np.shape(Oulu_data)[0])
Oulu_data=Oulu_data[shufle_indicies,:]
Oulu_labels=Oulu_labels[shufle_indicies,:]




selected_features_Oulu_data=Oulu_data*mask


print('accuracy on Oulu Data:',gpc.score(selected_features_Oulu_data, Oulu_labels))
probs = gpc.predict_proba(selected_features_Oulu_data) 
probs = probs[:, 1]  
auc = roc_auc_score(Oulu_labels, probs)  
print('AUC: %.2f' % auc)
print(f1_score(Oulu_labels, gpc.predict(selected_features_Oulu_data), average='weighted')) 
