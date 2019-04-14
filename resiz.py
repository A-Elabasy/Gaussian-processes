import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from torchvision import transforms, utils
#from skimage import io, transform


#torch.backends.cudnn.benchmark=True
cuda = torch.device("cuda")   


mnist_trainset = datasets.MNIST(root='./data', train=True, download=False, transform=torchvision.transforms.ToTensor())
mnist_testset =datasets.MNIST(root='./data', train=False, download=False, transform=torchvision.transforms.ToTensor())

mnist_testset=mnist_testset

print(mnist_trainset.data.shape)
print(mnist_trainset.data[0:2000].view(-1,28*28).shape)
images= torch.ones([60000,64],dtype=torch.float32)
images_test=torch.ones([10000,64],dtype=torch.float32)
print(images.shape)
print(images[0].shape)
for i in range(60000):
	trans=torchvision.transforms.ToPILImage(mode=None)
	resize = transforms.Resize(size=(8, 8))
	img=resize(trans(mnist_trainset.data[i]))
	tens=torchvision.transforms.ToTensor()
	img=tens(img).view(-1,8*8)
	images[i]= img
for i in range(10000):
	trans=torchvision.transforms.ToPILImage(mode=None)
	resize = transforms.Resize(size=(8, 8))
	img=resize(trans(mnist_testset.data[i]))
	tens=torchvision.transforms.ToTensor()
	img=tens(img).view(-1,8*8)
	images_test[i]= img

#tens(np.array(images))
print(images.shape)

kernel = 1.0 * RBF(8*8)


gpc = GaussianProcessClassifier(kernel=kernel,n_restarts_optimizer=3,random_state=None,multi_class="one_vs_rest",max_iter_predict=10,n_jobs=-1)
gpc=gpc.fit(images[0:4000], mnist_trainset.targets[0:4000])

print(gpc.score(images[0:4000], mnist_trainset.targets[0:4000]))
print(gpc.score(images_test[0:500], mnist_testset.targets[0:500]))
print(gpc.predict(images[0:20]))
print(mnist_trainset.targets[0:20])
print(gpc.predict(images_test[0:20]))
print(mnist_testset.targets[0:20])
