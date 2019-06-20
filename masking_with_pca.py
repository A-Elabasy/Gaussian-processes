#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 20:46:16 2019

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


in_file_name = '/data/fmri/cv/CV_OULU.nii.gz'
mask_file_name = '/data/fmri/cv/4mm_brain_mask_bin.nii.gz'

# Load subjects
in_img = nib.load(in_file_name)
in_array = in_img.get_fdata()
in_array=in_array.T
in_shape = in_array.shape

control_labels=np.ones((24,1))
AD_labels=np.zeros((17,1))




labels=np.vstack((control_labels,AD_labels))
'''
print(in_array[1])
print(np.shape(in_array[1]))
print('Shape: ', in_shape)

'''
pca = decomposition.PCA()


in_array_reshaped=np.reshape(in_array,(41,-1))
#print("the shape before the pca decomposition:%d" % np.shape(in_array_reshaped))
#print(in_array_reshaped)
pca.fit(in_array_reshaped)
in_array_reshaped = pca.transform(in_array_reshaped)

#print("the shape after the pca decomposition: %d" % np.shape(in_array_reshaped))
#print(in_array_reshaped)


data_train, data_test, labels_train, labels_test = train_test_split(in_array_reshaped, labels, test_size=0.20, random_state=42)
   



#print('Shape: ', in_shape)
        
'''
# Load mask
mask_img = nib.load(mask_file_name)
mask_array = mask_img.get_fdata()
assert mask_array.shape == in_shape[:-1]  # last (4th) dimension are the subjects
'''
# Do something with the input data, and end up with output
k=5
fold_length=math.floor(np.shape(data_train)[0]/k)
out_array = np.zeros(in_shape, dtype=float)
feature_weight=np.zeros(np.shape(data_train)[1])
threshold=3
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
 
 start = time.time()

 #data_train_feature_selection=data_train[0:8,:]
 #labels_train_feature_selection=labels_train[0:8,:]

 
 svc = SVC(kernel="linear")

 
 rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(3),
              scoring='accuracy',n_jobs=-1)
 rfecv.fit(data_train_train,labels_train_train )       

 print("Optimal number of features : %d" % rfecv.n_features_) 

 indecies=rfecv.get_support( indices=True)
 feature_weight[indecies]=feature_weight[indecies]+1


 #rvec.ranking()#

 end = time.time()
 print(end - start)

 # the selected features
mask=np.array(feature_weight>threshold,dtype=int) 
#remained_feature_indices=np.where(mask==1)


selected_features_data=data_train*mask
selected_features_data_test=data_test*mask

print(selected_features_data.shape)
print(labels_train.shape)
print(selected_features_data_test.shape)
print(labels_test.shape)


#the gussian process classification
kernel = 1.0 * RBF(214)

gpc = GaussianProcessClassifier(kernel=kernel,n_restarts_optimizer=5,random_state=None,multi_class="one_vs_rest",max_iter_predict=100,n_jobs=-1)
gpc=gpc.fit(selected_features_data,labels_train)

print('')
print('accuracy on trainingset:',gpc.score(selected_features_data,labels_train))
print('accuracy on testset:',gpc.score(selected_features_data_test, labels_test))


probs = gpc.predict_proba(selected_features_data_test) 
probs = probs[:, 1]  
auc = roc_auc_score(labels_test, probs)  
print('AUC: %.2f' % auc)
print(f1_score(labels_test, gpc.predict(selected_features_data_test), average='weighted')) 


'''                
# Now we want same dimensions and orientation as the input
hdr = in_img.header
aff = in_img.affine
out_img = nib.Nifti1Image(out_array, aff, hdr)

# Save to disk
out_file_name = 'masked.nii.gz'
nib.save(out_img, out_file_name)
'''

