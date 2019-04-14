import math
import torch
import gpytorch
import matplotlib 
from matplotlib import pyplot as plt

#torch.backends.cudnn.benchmark=True
cuda = torch.device("cuda:0") 
import torchvision
import torchvision.datasets as datasets
torch.cuda.set_device(0)
#torch.set_num_threads(40)

mnist_trainset = datasets.MNIST(root='./data', train=True, download=False, transform=None)
mnist_testset =datasets.MNIST(root='./data', train=False, download=False, transform=None)
train_x=mnist_trainset.data[0:6000].view(-1,28*28).float()
train_y=mnist_trainset.targets[0:6000]
train_y=train_y
train_y=train_y.float()

from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy


class GPClassificationModel(AbstractVariationalGP):
    def __init__(self, train_x):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(self, train_x, variational_distribution)
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


# Initialize model and likelihood
model = GPClassificationModel(train_x)
likelihood = gpytorch.likelihoods.BernoulliLikelihood()


from gpytorch.mlls.variational_elbo import VariationalELBO

# Find optimal model hyperparameters
model=model
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# "Loss" for GPs - the marginal log likelihood
# num_data refers to the amount of training data
mll = VariationalELBO(likelihood, model, train_y.numel())
with torch.cuda.device(0):
 training_iter = 100
 for i in range(training_iter):
  for j in range(6):
    	# Zero backpropped gradients from previous iteration
   optimizer.zero_grad()
    	# Get predictive output
   output = model(train_x[1000*j:1000*(j+1)])
    	# Calc loss and backprop gradients
   loss = -mll(output, train_y[1000*j:1000*(j+1)])
   loss.backward()
   print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
   optimizer.step()


'''
training_iter = 100
for i in range(training_iter):
# Zero backpropped gradients from previous iteration
	optimizer.zero_grad()
# Get predictive output
	output = model(train_x)
# Calc loss and backprop gradients
	loss = -mll(output, train_y)
	loss.backward()
	print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
	optimizer.step()
'''

