import numpy as np
import gpytorch
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import TransformedTargetRegressor

    
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,kernel,kernel_params,mean_func,ref_model,ref_likelihood,a,a_indicator,poly_degree,norm_const):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        nDim = train_x.shape[1]
        
        if kernel =='RBF':
            self.covar_module =gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=nDim))     

        if kernel =='matern':
            self.covar_module =gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=kernel_params))

        if kernel =='periodic':
            self.covar_module =gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel(ard_num_dims=nDim))
        
        if mean_func=='constant':
            self.mean_module = gpytorch.means.ConstantMean()
        if mean_func=='linear': 
            self.mean_module = gpytorch.means.LinearMean(input_size=nDim)
        if mean_func=='zero':  
            self.mean_module = gpytorch.means.ZeroMean()   
        #if mean_func=='discrepancy':
            #if a==None:
           #     self.register_parameter(name="a", parameter=torch.nn.Parameter(torch.randn(1)))
           # elif a_indicator==True:
           #     self.register_parameter(name="a", parameter=torch.nn.Parameter(torch.randn(torch.sum(a!=0))))
            #    ref_model=ref_model[a!=0]
          #      ref_likelihood=ref_likelihood[a!=0]
                
           # else:
           #     self.a=a
           # self.mean_module = DiscrepancyMean(input_size=nDim,ref_model=ref_model,ref_likelihood=ref_likelihood,a=self.a)
           # self.covar_module = aKernel(self.a,ref_model,ref_likelihood)+self.covar_module
            
           # self.covar_module.kernels[0].ref_model.requires_grad_(False)
           # self.covar_module.kernels[0].ref_likelihood.requires_grad_(False)
            
        if mean_func=='discrepancy_cohort':
            if a==None:
                self.register_parameter(name="a", parameter=torch.nn.Parameter(torch.randn(len(ref_model))))
            elif a_indicator==True:
                
                ref_model=np.array(ref_model)[np.array(a!=0)].tolist()
                ref_likelihood=np.array(ref_likelihood)[np.array(a!=0)].tolist()
                self.register_parameter(name="a", parameter=torch.nn.Parameter(torch.randn(len(ref_model))))
            else:
                
                ref_model=np.array(ref_model)[np.array(a!=0)].tolist()
                ref_likelihood=np.array(ref_likelihood)[np.array(a!=0)].tolist()
                self.a=a[a!=0]
                
            self.mean_module = DiscrepancyMeanCohort(input_size=nDim,ref_model=ref_model,ref_likelihood=ref_likelihood,a=self.a)
            for i in range(len(ref_model)):

                self.covar_module = aKernel(self.a[i],ref_model[i],ref_likelihood[i]) + self.covar_module

                self.covar_module.kernels[0].ref_model.requires_grad_(False)
                self.covar_module.kernels[0].ref_likelihood.requires_grad_(False)
            
        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=nDim))
        if mean_func=='structured':
            self.mean_module = StructuredMean(input_size=nDim,poly_degree=poly_degree,norm_const=norm_const)
            
    def forward(self, x):
        
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



class aKernel(gpytorch.kernels.Kernel):
    # the sinc kernel is stationary
    is_stationary = True

    # We will register the parameter when initializing the kernel
    def __init__(self,a,ref_model,ref_likelihood, length_prior=None, length_constraint=None, **kwargs):
        super().__init__(**kwargs)
    
        self.a=a
        self.ref_model=ref_model
        self.ref_likelihood=ref_likelihood
    def forward(self, x1, x2, **params):

        
        self.ref_model.eval()
        self.ref_likelihood.eval() 
        #diff=self.covar_dist(x1, x2, **params)
        #diff=diff/diff

        a2=(self.a**2)

        n=x1.shape[0]

        with gpytorch.settings.fast_pred_var():
            out = self.ref_likelihood(self.ref_model(torch.cat([x1, x2], dim=-2)))
            diff = out.covariance_matrix[:n, n:]

        return a2*diff
    

class DiscrepancyMean(gpytorch.means.Mean):
    def __init__(self, input_size,ref_model,ref_likelihood,a, batch_shape=torch.Size(), bias=True):
        super().__init__()
        self.register_parameter(name="weights", parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, 1)))
        if bias:
            self.register_parameter(name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
        else:
            self.bias = None
        self.a=a
        self.ref_model=ref_model
        self.ref_likelihood=ref_likelihood
        
    def forward(self, x):
        self.ref_model.eval()
        self.ref_likelihood.eval() 
        res1 = x.matmul(self.weights).squeeze(-1)
        res2=self.ref_likelihood(self.ref_model(x)).mean
        res=res1+self.a*res2

        if self.bias is not None:
            res = res1 + self.bias+self.a*res2
            
        return res    


class DiscrepancyMeanCohort(gpytorch.means.Mean):
    def __init__(self, input_size,ref_model,ref_likelihood,a, batch_shape=torch.Size(), bias=True):
        super().__init__()
        self.register_parameter(name="weights", parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, 1)))
        if bias:
            self.register_parameter(name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
        else:
            self.bias = None
        self.a=a
        self.ref_model=ref_model
        self.ref_likelihood=ref_likelihood

    def forward(self, x):
        res2=0
        res1 = x.matmul(self.weights).squeeze(-1)
        for i in range(len(self.ref_model)): 
            self.ref_model[i].eval()
            self.ref_likelihood[i].eval() 
            res2+=self.a[i]*self.ref_likelihood[i](self.ref_model[i](x)).mean
        
        res=res1+res2

        if self.bias is not None:
            res = res1 + self.bias+res2

        return res 
    
class StructuredMean(gpytorch.means.Mean):
    def __init__(self, input_size,poly_degree,norm_const,batch_shape=torch.Size(), bias=True):
        super().__init__()
        
        if bias:
            self.register_parameter(name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
        else:
            self.bias = None
        self.poly_degree=poly_degree
        self.norm_const=norm_const
        self.poly=PolynomialFeatures(degree=self.poly_degree, include_bias=False,interaction_only=False)
        
        pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('preprocessor', PolynomialFeatures(degree=poly_degree, include_bias=False,interaction_only=False)),
    ('lasso', LassoCV(n_alphas=100,max_iter=10000))
])
        estimator = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())
        
        X_over_train = torch.cat((norm_const[2],1/norm_const[2]),axis=1)                 
        # fit to an order-3 polynomial data
        model = estimator.fit(X_over_train, norm_const[3])
        lasso_coef=model.regressor_.named_steps['lasso'].coef_
        self.structure_index=torch.tensor(np.abs(lasso_coef)>0)

        lasso_size=torch.sum(self.structure_index)
        self.register_parameter(name="weights", parameter=torch.nn.Parameter(torch.randn(*batch_shape, lasso_size, 1)))
        
    def forward(self, x):
        x=x*self.norm_const[1]+self.norm_const[0]
        x=torch.cat((x,1/x),axis=1)
        x=torch.tensor(self.poly.fit_transform(x)[:,self.structure_index]).float()
        x=(x-x.mean())/x.std()
        res = x.matmul(self.weights).squeeze(-1)

        if self.bias is not None:
            res = res + self.bias

        return res    