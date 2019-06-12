from numpy import load
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import FastICA, PCA
from sklearn.metrics import classification_report

data_ad = load('/data/fmri/cv/hippocampus/hipp_ad.npz')
lst_ad=data_ad['masked_voxels']
data_con = load('/data/fmri/cv/hippocampus/hipp_con.npz')
import numpy as np
lst_con=data_con['masked_voxels']
print(np.shape(lst_ad))
print(np.shape(lst_con))
lst=np.hstack((lst_ad,lst_con)).T
print(np.shape(lst))
labels_ad=np.ones((np.shape(lst_ad)[1],1))
labels_con=np.zeros((np.shape(lst_con)[1],1))
print(np.shape(labels_ad))
print(np.shape(labels_con))
labels=np.vstack((labels_ad,labels_con))
print(np.shape(labels))
data_train, data_test, labels_train, labels_test = train_test_split(lst, labels, test_size=0.20, random_state=42)


kernel = 1.0 * RBF(214)
'''
S=data_train
T=data_test
S /= S.std(axis=0)
T /= T.std(axis=0)
ica = FastICA(n_components=14)
S_ = ica.fit_transform(S)
T_=ica.fit_transform(T)
'''
gpc = GaussianProcessClassifier(kernel=kernel,n_restarts_optimizer=5,random_state=None,multi_class="one_vs_rest",max_iter_predict=100,n_jobs=-1)
gpc=gpc.fit(data_train, labels_train)
print('')
print('accuracy on trainingset:',gpc.score(data_train,labels_train))
print('accuracy on testset:',gpc.score(data_test, labels_test))
print("confusion matrix for the training")
cm_train = confusion_matrix(labels_train, gpc.predict(data_train))
print(cm_train)
print(classification_report(labels_train, gpc.predict(data_train), labels=[0, 1]))
print("confusion matrix for the testing")
cm_test = confusion_matrix(labels_test, gpc.predict(data_test))
print(cm_test)
print('')
print(classification_report(labels_test, gpc.predict(data_test), labels=[0, 1]))
