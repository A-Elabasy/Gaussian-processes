import numpy
import torch
import torchvision
import torchvision.datasets as datasets
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF




mnist_trainset = datasets.MNIST(root='./data', train=True, download=False, transform=torchvision.transforms.ToTensor())
mnist_testset =datasets.MNIST(root='./data', train=False, download=False, transform=torchvision.transforms.ToTensor())

mnist_testset=mnist_testset

print(mnist_trainset.data.shape)
print(mnist_trainset.data[0:2000].view(-1,28*28).shape)
kernel = 1.0 * RBF(28*28)



gpc = GaussianProcessClassifier(kernel=kernel,n_restarts_optimizer=3,random_state=None,multi_class="one_vs_rest",max_iter_predict=200,n_jobs=-1)
gpc=gpc.fit(mnist_trainset.data[0:1000].view(-1,28*28), mnist_trainset.targets[0:1000])

print(gpc.score(mnist_trainset.data[0:1000].view(-1,28*28), mnist_trainset.targets[0:1000]))
print(gpc.score(mnist_testset.data[0:500].view(-1,28*28), mnist_testset.targets[0:500]))
print(gpc.predict(mnist_trainset.data[0:20].view(-1,28*28)))
print(mnist_trainset.targets[0:20])
print(gpc.predict(mnist_testset.data[0:20].view(-1,28*28)))
print(mnist_testset.targets[0:20])