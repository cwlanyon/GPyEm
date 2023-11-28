import numpy as np
import GP_functions as GPF
import torch
import gpytorch
import os
from gpytorch.likelihoods import GaussianLikelihood
    
class ensemble():
    def __init__(self,X_train,y_train,mean_func='constant',training_iter=1000):


        self.training_input = X_train
        self.training_output = y_train
        self.mean_func = mean_func
        self.training_input_normalised, self.training_input_mean, self.training_input_STD = self.normalise(X_train)
        self.training_output_normalised, self.training_output_mean, self.training_output_STD = self.normalise(y_train)
        self.training_iter=training_iter
        self.models, self.likelihoods = self.create_ensemble()
        
    def normalise(self,data):
        dataMean = np.mean(data,axis=0)
        dataStd = np.std(data,axis=0)
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
        inputVals = torch.tensor(((inputVals-self.training_input_mean)/self.training_input_STD).values).float()
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
        inputVals = torch.tensor(((inputVals-self.training_input_mean)/self.training_input_STD).values).float()
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
        
        outputVals = torch.tensor(outputVals.values)
        MSE_score = ((self.predict(inputVals)-outputVals)**2).mean(axis=0)
        return MSE_score
    
    def MSE_sample(self,inputVals,outputVals,n=5):
        MSE_score=[]
        outputVals = torch.tensor(outputVals.values)
        pred = self.predict_sample(inputVals,n)
        for i in range(n):
            MSE_score.append(((pred[:,i,:]-outputVals)**2).mean(axis=0).detach().numpy())
     
        MSE_mean = torch.tensor(np.array(MSE_score).mean(axis=0))
        MSE_std = torch.tensor(np.array(MSE_score).std(axis=0))
        return MSE_mean, MSE_std

    def R2(self,inputVals,outputVals):
        R2_score=1-self.MSE(inputVals,outputVals)/torch.tensor(np.var(outputVals,axis=0))
        return R2_score
    
    def R2_sample(self,inputVals,outputVals,n=5):
        R2_score=[]
        pred = self.predict_sample(inputVals,n).detach().numpy()
        for i in range(n):
            R2_score.append(1-((pred[:,i,:]-outputVals)**2).mean(axis=0)/np.var(outputVals,axis=0))             
        R2_mean = torch.tensor(np.array(R2_score).mean(axis=0))
        R2_std = torch.tensor(np.array(R2_score).std(axis=0))
        return R2_mean, R2_std
    
    def ISE(self,inputVals,outputVals):
        pred=self.predict(inputVals)
        pMean=pred.mean(axis=0)
        pSTD=pred.std(axis=0)
        ISE_score = (100*sum(torch.FloatTensor.abs_(pMean-torch.tensor(outputVals.values))<2*pSTD)/inputVals.shape[0])
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
            