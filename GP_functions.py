import numpy as np
import gpytorch
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
    
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        nDim = train_x.shape[1]
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=nDim))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactLRGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactLRGPModel, self).__init__(train_x, train_y, likelihood)
        nDim = train_x.shape[1]
        self.mean_module = gpytorch.means.LinearMean(input_size=nDim)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=nDim))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ZeroMeanGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ZeroMeanGPModel, self).__init__(train_x, train_y, likelihood)
        nDim = train_x.shape[1]
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=nDim))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    
class DiscrepancyGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,ref_emulator,a):
        super(DiscrepancyGPModel, self).__init__(train_x, train_y, likelihood)
        
        #self.register_parameter(name="a", parameter=torch.nn.Parameter(torch.randn(1)))
        self.a=a
        nDim = train_x.shape[1]
        
        ref_weights=ref_emulator.mean_module.weights
        ref_bias=ref_emulator.mean_module.bias
        ref_outputscale=ref_emulator.covar_module.outputscale
        ref_lengthscale=ref_emulator.covar_module.base_kernel.lengthscale
        
        self.mean_module = DiscrepancyMean(input_size=nDim,ref_weights=ref_weights,ref_bias=ref_bias,a=self.a)
        outscale=(self.a**2)*ref_outputscale
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=nDim))*aKernel(self.a)+gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=nDim))
        
        
        self.mean_module.ref_weights.requires_grad_(False)
        self.mean_module.ref_bias.requires_grad_(False)
        self.covar_module.kernels[0].kernels[0].base_kernel.raw_lengthscale.requires_grad_(False)
        self.covar_module.kernels[0].kernels[0].raw_outputscale.requires_grad_(False)
        self.covar_module.kernels[0].kernels[0].outputscale=ref_outputscale
        self.covar_module.kernels[0].kernels[0].base_kernel.lengthscale=ref_lengthscale

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class aKernel(gpytorch.kernels.Kernel):
    # the sinc kernel is stationary
    is_stationary = True

    # We will register the parameter when initializing the kernel
    def __init__(self,a, length_prior=None, length_constraint=None, **kwargs):
        super().__init__(**kwargs)
    
        self.a=a
    
    def forward(self, x1, x2, **params):
        
        diff=self.covar_dist(x1, x2, **params)
        diff=diff/diff

        a2=(self.a**2)
        a2=0
        return a2*diff
    
    
class DiscrepancyMean(gpytorch.means.Mean):
    def __init__(self, input_size,ref_weights,ref_bias,a, batch_shape=torch.Size(), bias=True):
        super().__init__()
        self.register_parameter(name="weights", parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, 1)))
        if bias:
            self.register_parameter(name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
        else:
            self.bias = None
        self.a=a
        self.ref_weights=ref_weights
        self.ref_bias=ref_bias

    def forward(self, x):
        ref = x.matmul(self.ref_weights).squeeze(-1)
        res1 = x.matmul(self.weights).squeeze(-1)
        
        res=self.a*(ref)+res1
        
        if self.bias is not None:
            res = self.a*(ref+self.ref_bias)+res1 + self.bias
            
        return res