import numpy as np
import torch
import os

class LVE():
    def __init__(self,XL,Y,sigma2,p,N,L_dim,mean_func='zero',L_length=4,training_iter=1000000,lr=1e-5,kernel='RBF',kernel_params=None,scale_init=None,lengthscale_init=None):

        X_norm, self.training_input_mean, self.training_input_STD = self.normalise(XL[:,:L_dim])
        Y_norm, self.training_output_mean, self.training_output_STD = self.normalise(Y)
        
        XL[:,:L_dim] = X_norm
       
        self.XL = XL.float() 
        self.Y=Y_norm.float() 
        
        self.sigma2=sigma2
        self.iter = training_iter
        self.lr=lr
        self.L_size = XL[:,L_dim:].shape[1]
        self.X_size = XL[:,:L_dim].shape[1]
        self.L_dim=L_dim
        self.L_length=L_length
        self.p=p
        self.N=N
        self.L = torch.unique(self.XL[:,self.L_dim:],sorted=False,dim=0).clone().detach().requires_grad_(True)

        
        self.scale = (torch.rand(self.Y.shape[1])).clone().detach().requires_grad_(True)
        self.lengthscale = (torch.rand(self.Y.shape[1],self.X_size)).clone().detach().requires_grad_(True)

        self.history_gd = []
        self.scales=[]
        self.lengths=[]
        self.Ls=[]
        self.mean=0
        self.optimise(self.iter,self.lr)

        
        
    def normalise(self,data):
        dataMean = data.mean(axis=0)
        dataStd = data.std(axis=0)
        dataNorm = (data-dataMean)/(dataStd)
        return dataNorm,dataMean,dataStd  

    def normalise_test_data(self,input_data,output_data):
        
        inputNorm = (input_data-self.training_input_mean)/(self.training_input_STD)
        outputNorm = (output_data-self.training_output_mean)/(self.training_output_STD)
        return inputNorm,outputNorm

    def normalise_output(self,output_data):
        
        outputNorm = (output_data-self.training_output_mean)/(self.training_output_STD)
        return outputNorm
    
    def normalise_input(self,input_data):
        
        inputNorm = (input_data-self.training_input_mean)/(self.training_input_STD)
        return inputNorm
    
    def multivariate_ll(self,y,mu,K):
        det = 2*torch.sum(torch.log(torch.diag(torch.linalg.cholesky(K))))

        ll = -((y.shape[0]*y.shape[1])/2)*torch.log(2*torch.tensor(torch.pi))-0.5*y.shape[1]*det-0.5*torch.sum(torch.diag(torch.matmul((y-mu).T,torch.linalg.solve(K,y-mu))))
        #ll = -(y.shape[0]/2)*torch.log(2*torch.tensor(torch.pi))-0.5*det-0.5*torch.matmul((y-mu).T,torch.linalg.solve(K,y-mu))
        
        return ll

    def rbf(self,X1,X2,scaling,lengthscale):
    
        X_1 = X1/lengthscale
        X_2 = X2/lengthscale
        rbf = scaling * torch.exp(-0.5*torch.cdist(X_1,X_2,p=2)**2)
    
        return rbf

    def linear(self,X1,X2,scaling,lengthscale):
        
        X_1 = X1/lengthscale
        X_2 = X2/lengthscale
        
        lin =  scaling*torch.matmul(X_1, X_2.T) 
        return lin

    def posterior_theta(self,Y,X_test,X_train,scaling,lengthscale,sigma2,mean):
        K = self.rbf(X_train,X_train,scaling,lengthscale)+sigma2*torch.eye(X_train.shape[0])
        K_s = self.rbf(X_test,X_train,scaling,lengthscale)
        K_ss = self.rbf(X_test,X_test,scaling,lengthscale)
    
        mean_p = mean+torch.matmul(K_s,torch.linalg.solve(K,Y-mean))
        K_p = K_ss-torch.matmul(K_s,torch.linalg.solve(K,K_s.T))+sigma2*torch.eye(X_test.shape[0])
        return mean_p, K_p  


    def cost(self,y,XL,scaling,lengthscale,sigma2,L_dim,mean=0):
        c = 0
        for i in range(y.shape[1]):           
            K=self.rbf(XL,XL,scaling[i],lengthscale[i])+sigma2*torch.eye(XL.shape[0])
            c += self.multivariate_ll(y[:,[i]],mean,K)
        c+=self.multivariate_ll(XL[:,L_dim:],0,torch.eye(XL.shape[0]))
        return -c

    
    def optimise(self,iters,lr):
        
        X = self.XL[:,:self.L_dim].detach().clone()
        L = self.L
        XL = torch.cat((X,torch.repeat_interleave(L,self.p,dim=0)),axis=1)  
        scaling = self.scale
        lengthscale = self.lengthscale
        lengthscale_in=torch.cat([lengthscale,self.L_length*torch.ones(self.Y.shape[1],self.L_size)],axis=1)
                  
        gd = torch.optim.Adam([L,scaling,lengthscale], lr=self.lr)
        for i in range(iters):
            
            gd.zero_grad()
            objective = self.cost(self.Y,XL,scaling,lengthscale_in,self.sigma2,L_dim=self.L_dim)
            objective.backward()
            gd.step()
            self.history_gd.append(objective.item())
            self.scales.append(scaling.clone().detach())
            self.lengths.append(lengthscale.clone().detach())
            self.Ls.append(L.clone().detach())

            lengthscale_in=torch.cat([lengthscale,self.L_length*torch.ones(self.Y.shape[1],self.L_size)],axis=1)
            XL = torch.cat((X,torch.repeat_interleave(L,self.p,dim=0)),axis=1)
            
            if i%1000 ==0:
                print(objective)

        #self.L = L
        #self.scale = scaling
        #self.lengthscale = lengthscale
        self.XL = XL    
        return  


    def predict(self,XL_test):

        pred_mean = torch.zeros(XL_test.shape[0],self.Y.shape[1])
        pred_var = torch.zeros(XL_test.shape[0],self.Y.shape[1])
        scaling = self.scale
        

        XL_test[:,:self.L_dim] = self.normalise_input(XL_test[:,:self.L_dim])
        
        Y_std = self.training_output_STD
        Y_mean = self.training_output_mean
        
        for i in range(self.Y.shape[1]):
            lengthscale_in=torch.cat([self.lengthscale[i],self.L_length*torch.ones(self.L_size)])
            m_p,K_p = self.posterior_theta(self.Y[:,i],XL_test,self.XL,scaling[i],lengthscale_in,self.sigma2,self.mean)
            pred_mean[:,i] = m_p*Y_std[i] + Y_mean[i]
            pred_var[:,i] = K_p.diag()*Y_std[i]**2
        return pred_mean,pred_var
    
    def MSE(self, XL_test,Y_test):
        
        pred,_ = self.predict(XL_test)
        MSE = ((pred-Y_test)**2).mean(axis=0)
        return MSE
        
    def R2(self, XL_test,Y_test): 
        R2 = 1-self.MSE(XL_test,Y_test)/torch.var(Y_test,axis=0)
        return R2

    def cf_cost(self,XL_new,Y_new,L_dim,mean=0):
        c = 0
        for i in range(self.Y.shape[1]):
            lengthscale_in=torch.cat([self.lengthscale[i],self.L_length*torch.ones(self.L_size)])
            mean_p,K_p = self.posterior_theta(self.Y[:,[i]],XL_new,self.XL,self.scale[i],lengthscale_in,self.sigma2,self.mean)
            c += self.multivariate_ll(Y_new[:,[i]],mean_p,K_p)
        c+=self.multivariate_ll(XL_new[:,self.L_dim:],0,torch.eye(XL_new.shape[0]))
        return -c

    def cutting_feedback(self,XL_new,Y_new,iters,lr):
        
        L_new = torch.unique(XL_new[:,self.L_dim:],sorted=False,dim=0).clone().detach().requires_grad_(True)
        X_new = self.normalise_input(XL_new[:,:self.L_dim]).detach().clone()
        Y_new = self.normalise_output(Y_new).detach().clone()
        XL_new = torch.cat((X_new,torch.repeat_interleave(L_new,X_new.shape[0],dim=0)),axis=1)             
        gd = torch.optim.Adam([L_new], lr=self.lr)
        for i in range(iters):
            
            gd.zero_grad()
            objective = self.cf_cost(XL_new,Y_new,L_dim=self.L_dim)
            objective.backward()
            gd.step()

            XL_new = torch.cat((X_new,torch.repeat_interleave(L_new,X_new.shape[0],dim=0)),axis=1)
            
            if i%1000 ==0:
                print(objective)
  
        return L_new
