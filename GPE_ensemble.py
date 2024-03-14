import numpy as np
import GP_functions as GPF
import torch
import gpytorch
import os
from gpytorch.likelihoods import GaussianLikelihood
class ensemble():
    def __init__(self,X_train,y_train,mean_func='constant',training_iter=1000,restarts=0,X_val=torch.tensor([]),y_val=torch.tensor([])):


        self.training_input = X_train
        self.training_output = y_train
        self.mean_func = mean_func
        self.training_input_normalised, self.training_input_mean, self.training_input_STD = self.normalise(X_train)
        self.training_output_normalised, self.training_output_mean, self.training_output_STD = self.normalise(y_train)
        self.training_iter=training_iter
        
        self.X_val=X_val
        self.y_val=y_val
        
        
        if restarts == 0:
            self.models, self.likelihoods = self.create_ensemble()
        elif restarts >0:
            self.models,self.likelihoods=self.create_ensemble_restart()
        
        
    def normalise(self,data):
        dataMean = data.mean(axis=0)
        dataStd = data.std(axis=0)
        dataNorm = (data-dataMean)/(dataStd)
        return dataNorm,dataMean,dataStd
    
    def normalise_test_data(self,input_data,output_data):
        
        inputNorm = (input_data-self.training_input_mean)/(self.training_input_STD)
        outputNorm = (output_data-self.training_output_mean)/(self.training_output_STD)
        return inputNorm,outputNorm

    def create_ensemble(self):
        modelInput = self.training_input_normalised
        modelOutput = self.training_output_normalised
        meanFunc = self.mean_func

        models = []
        likelihoods = []
        nMod = modelOutput.shape[1]
        #nDim = modelOutput.shape[1]
        X=modelInput.float()

        for i in range(nMod):
            Y=modelOutput[:,i].float()
            print(i)
            likelihoods.append(gpytorch.likelihoods.GaussianLikelihood())
            if meanFunc=='constant':
                models.append(GPF.ExactGPModel(X, Y, likelihoods[i]))
            if meanFunc=='linear':
                models.append(GPF.ExactLRGPModel(X, Y, likelihoods[i]))
            if meanFunc=='zero':
                models.append(GPF.ZeroMeanGPModel(X, Y, likelihoods[i]))
            smoke_test = ('CI' in os.environ)
            training_iter = 2 if smoke_test else self.training_iter

            # Find optimal model hyperparameters
            models[i].train()
            likelihoods[i].train()


            # Use the adam optimizer
            optimizer = torch.optim.Adam(models[i] .parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihoods[i], models[i] )


            for j in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = models[i](X)
                # Calc loss and backprop gradients
                loss = -mll(output, Y)
                loss.backward()
                #print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                #    j + 1, training_iter, loss.item(),
                #    models[i].covar_module.base_kernel.lengthscale.item(),
                #    models[i].likelihood.noise.item()
                #))
                optimizer.step()
        return models, likelihoods
    
    
    def create_ensemble_restarts(self):
        modelInput = self.training_input_normalised
        modelOutput = self.training_output_normalised
        meanFunc = self.mean_func
        
        X_val_norm,y_val_norm=self.normalise_test_data(self.X_val,self.y_val)
        

        models = []
        likelihoods = []
        nMod = modelOutput.shape[1]
        #nDim = modelOutput.shape[1]
        X=torch.tensor(modelInput.values).float()

        for i in range(nMod):
            Y=torch.tensor(modelOutput.iloc[:,i].values).squeeze().float() 
            print(i)
            likelihoods.append(gpytorch.likelihoods.GaussianLikelihood())
            if meanFunc=='constant':
                models.append(GPF.ExactGPModel(X, Y, likelihoods[i]))
            if meanFunc=='linear':
                models.append(GPF.ExactLRGPModel(X, Y, likelihoods[i]))
            if meanFunc=='zero':
                models.append(GPF.ZeroMeanGPModel(X, Y, likelihoods[i]))
            smoke_test = ('CI' in os.environ)
            training_iter = 2 if smoke_test else self.training_iter

            # Find optimal model hyperparameters
            models[i].train()
            likelihoods[i].train()


            # Use the adam optimizer
            optimizer = torch.optim.Adam(models[i] .parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihoods[i], models[i] )


            for j in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = models[i](X)
                # Calc loss and backprop gradients
                loss = -mll(output, Y)
                loss.backward()
                #print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                #    j + 1, training_iter, loss.item(),
                #    models[i].covar_module.base_kernel.lengthscale.item(),
                #    models[i].likelihood.noise.item()
                #))
                optimizer.step()
        return models, likelihoods

    def predict(self,inputVals):

        models=self.models
        likelihoods=self.likelihoods
        outMean=self.training_output_mean
        outStd=self.training_output_STD

        #modelOutput = (modelOutputOrig-outMean)/outStd.T
        nMod = len(models)
        prediction=[]
        inputVals = ((inputVals-self.training_input_mean)/self.training_input_STD).float()
        for i in range(nMod):
            models[i].eval()
            likelihoods[i].eval()
            out = outStd[i]*(likelihoods[i](models[i](inputVals)).mean)+outMean[i]
            prediction.append(out)
        prediction=torch.stack(prediction).T
        return prediction
    
    def predict_sample(self,inputVals,n):

        models=self.models
        likelihoods=self.likelihoods
        outMean=self.training_output_mean
        outStd=self.training_output_STD

        #modelOutput = (modelOutputOrig-outMean)/outStd.T
        nMod = len(models)
        prediction=[]
        inputVals = ((inputVals-self.training_input_mean)/self.training_input_STD)
        for i in range(nMod):
            models[i].eval()
            likelihoods[i].eval()
            y_preds = likelihoods[i](models[i](inputVals))
            y_samples = y_preds.sample(sample_shape=torch.Size([n],))
            out = outStd[i]*(y_samples)+outMean[i]
            prediction.append(out)
        prediction=torch.stack(prediction).T
        return prediction

    def MSE(self,inputVals,outputVals):
        
        outputVals = outputVals
        MSE_score = ((self.predict(inputVals)-outputVals)**2).mean(axis=0)
        return MSE_score
    
    def MSE_sample(self,inputVals,outputVals,n=5):
        MSE_score=[]
        outputVals = outputVals
        pred = self.predict_sample(inputVals,n)
        for i in range(n):
            MSE_score.append(((pred[:,i,:]-outputVals)**2).mean(axis=0))
     
        MSE_mean = torch.tensor(MSE_score.mean(axis=0))
        MSE_std = torch.tensor(MSE_score.std(axis=0))
        return MSE_mean, MSE_std

    def R2(self,inputVals,outputVals):
        R2_score=1-self.MSE(inputVals,outputVals)/torch.tensor(torch.var(outputVals,axis=0))
        return R2_score
    
    def R2_sample(self,inputVals,outputVals,n=5):
        R2_score=[]
        pred = self.predict_sample(inputVals,n)
        for i in range(n):
            R2_score.append(1-((pred[:,i,:]-outputVals)**2).mean(axis=0)/torch.var(outputVals,axis=0))             
        R2_mean = torch.tensor(np.array(R2_score).mean(axis=0))
        R2_std = torch.tensor(np.array(R2_score).std(axis=0))
        return R2_mean, R2_std
    
    def ISE(self,inputVals,outputVals):
        pred=self.predict(inputVals)
        pMean=pred.mean(axis=0)
        pSTD=pred.std(axis=0)
        ISE_score = (100*sum(torch.FloatTensor.abs_(pMean-outputVals)<2*pSTD)/inputVals.shape[0])
        return ISE_score
    
    def ensemble_likelihood(self,candidateInput,outputVal):
        nMod = self.training_output_normalised.shape[1]
        models=self.models
        likelihoods=self.likelihoods
        likelihood_eval = np.zeros(nMod)
        inputNorm,outputNorm = self.normalise_test_data(candidateInput,outputVal)
        for i in range(nMod):
            models[i].eval()
            likelihoods[i].eval()
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihoods[i], models[i])
            likelihood_eval[i] = np.exp(mll(models[i](torch.tensor(inputNorm.values).float()),torch.tensor(outputNorm.iloc[:,i].values).float()).detach().numpy())
        return likelihood_eval 
    
    def ensemble_log_likelihood(self,candidateInput,outputVal):
        nMod = self.training_output_normalised.shape[1]
        models=self.models
        likelihoods=self.likelihoods
        likelihood_eval = torch.zeros(nMod)
        inputNorm,outputNorm = self.normalise_test_data(candidateInput,outputVal)
        for i in range(nMod):
            models[i].eval()
            likelihoods[i].eval()
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihoods[i], models[i])
            likelihood_eval[i] = mll(models[i](torch.tensor(inputNorm.values).float(),torch.tensor(outputNorm.iloc[:,i].values).float()))
        return likelihood_eval 
            
    def ensemble_log_likelihood_obs_error2(self,candidateInput,outputVal,sigma2): #NOT IN USE
        nMod = self.training_output_normalised.shape[1]
        nDim = self.training_output_normalised.shape[0]
        nP = candidateInput.shape[0]
        models=self.models
        likelihoods=self.likelihoods
        models=self.models
        likelihood_eval = torch.zeros((nMod,nP))
        inputNorm,outputNorm = self.normalise_test_data(candidateInput,outputVal)
        inputNorm=inputNorm.float()
        outputNorm=outputNorm.float()
        
        
        for i in range(nMod):
            models[i].eval()
            likelihoods[i].eval()
            sigma = sigma2/self.training_output_STD[i]
            m = likelihoods[i](models[i](inputNorm)).mean
            k = likelihoods[i](models[i](inputNorm)).covariance_matrix.diag()
            
            likelihood_manual=-0.5*((outputNorm[:,i]-m)**2)/(k+sigma) -0.5*nDim*torch.log(k+sigma) #- 0.5*nDim*torch.log(torch.tensor(2*torch.pi))
            likelihood_eval[i,:] = likelihood_manual
          
            
        return likelihood_eval
    
    def ensemble_log_likelihood_obs_error(self,candidateInput,outputVal,sigma2):
        nMod = self.training_output_normalised.shape[1]
        nDim = self.training_output_normalised.shape[0]
        nP = candidateInput.shape[0]
        models=self.models
        likelihoods=self.likelihoods
        models=self.models
        likelihood_eval = torch.zeros((nMod,nP))
        inputNorm,outputNorm = self.normalise_test_data(candidateInput,outputVal)
        inputNorm=inputNorm.float()
        outputNorm=outputNorm.float()
        
        
        for i in range(nMod):
            models[i].eval()
            likelihoods[i].eval()
            sigma = sigma2[i]
            m = likelihoods[i](models[i](inputNorm)).mean
            k = likelihoods[i](models[i](inputNorm)).covariance_matrix.diag()
            
            mean = self.training_output_STD[i]*m+self.training_output_mean[i]
            variance = (self.training_output_STD[i]**2)*k+sigma
            
            likelihood_manual=self.gaussian_ll(outputVal[:,i],mean,variance)
            
            #likelihood_manual=-0.5*((outputVal[:,i]-(self.training_output_STD[i]*m+self.training_output_mean[i]))**2)/((self.training_output_STD[i]**2)*k+sigma) -0.5*torch.log((self.training_output_STD[i]**2)*k+sigma) - 0.5*torch.log(torch.tensor(2*torch.pi))
            likelihood_eval[i,:] = likelihood_manual
          
            
        return likelihood_eval    
    
    def gaussian_ll(self,y,mean,variance):
        ll = -0.5*((y-mean)**2)/variance - 0.5*torch.log(variance) - 0.5*torch.log(torch.tensor(2*torch.pi))
        return ll