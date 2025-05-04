import numpy as np
from GPyEm import GP_functions as GPF
import torch
import gpytorch
import os
from gpytorch.likelihoods import GaussianLikelihood

class ensemble():
    def __init__(self,X_train,y_train,mean_func='constant',training_iter=1000,kernel='RBF',kernel_params=None,n_restarts=0,X_test=None,y_test=None,ref_emulator=None,a=None,a_indicator=False,poly_degree=None,train=True,save=False,load=False,save_loc=None):


        self.training_input = X_train
        self.training_output = y_train
        self.mean_func = mean_func
        self.training_input_normalised, self.training_input_mean, self.training_input_STD = self.normalise(X_train)
        self.training_output_normalised, self.training_output_mean, self.training_output_STD = self.normalise(y_train)
        self.training_iter=training_iter
        self.ref_emulator=ref_emulator
        self.a=a
        self.a_indicator=a_indicator
        self.kernel=kernel
        self.kernel_params=kernel_params
        self.train=train
        self.poly_degree=poly_degree
        
        if load==True: #Do not train model if loading previous hyperparameters
            self.train=False
        
        if n_restarts==0:
            self.models, self.likelihoods = self.create_ensemble()
        else:
            self.n_restarts=n_restarts
            self.X_test=X_test
            self.y_test=y_test
            self.models, self.likelihoods = self.create_ensemble_restarts()
            
        
        if save==True:
            self.save_emulator_state(save_loc)
        if load==True:
            self.load_emulator_state(save_loc)
        
        
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

    def create_ensemble(self):
        modelInput = self.training_input_normalised
        modelOutput = self.training_output_normalised
        meanFunc = self.mean_func
        models = []
        likelihoods = []
        nMod = modelOutput.shape[1]
        #nDim = modelOutput.shape[1]
        X=modelInput.float()
        
        #noise_prior=gpytorch.priors.SmoothedBoxPrior(0.15, 1.5, sigma=0.001))
        
        for i in range(nMod):
            norm_const=[self.training_input_mean, self.training_input_STD,self.training_input,self.training_output[:,i]]
            Y=modelOutput[:,i].float()
            print(i)
            likelihoods.append(gpytorch.likelihoods.GaussianLikelihood(noise_prior=gpytorch.priors.SmoothedBoxPrior(0.15, 1.5, sigma=0.1)))
            
            if self.ref_emulator == None:
                models.append(GPF.ExactGPModel(X, Y, likelihoods[i],self.kernel,self.kernel_params,self.mean_func,None,None,None,False,self.poly_degree,norm_const))
            elif self.mean_func == 'discrepancy_cohort':
                ref_models = []
                ref_likelihoods = []
                for k in range(len(self.ref_emulator)):
                    ref_models.append(self.ref_emulator[k].models[i])
                    ref_likelihoods.append(self.ref_emulator[k].likelihoods[i])
                if self.a==None:
                    models.append(GPF.ExactGPModel(X, Y, likelihoods[i],self.kernel,self.kernel_params,self.mean_func,ref_models,ref_likelihoods,None,False,self.poly_degree,norm_const))

                else:
                    models.append(GPF.ExactGPModel(X, Y, likelihoods[i],self.kernel,self.kernel_params,self.mean_func,ref_models,ref_likelihoods,self.a[i,:],self.a_indicator,self.poly_degree,norm_const))

            elif self.a!=None:    
                
                models.append(GPF.ExactGPModel(X, Y, likelihoods[i],self.kernel,self.kernel_params,self.mean_func,self.ref_emulator.models[i],self.ref_emulator.likelihoods[i],self.a[i],self.a_indicator,self.poly_degree,norm_const))
            else:
                models.append(GPF.ExactGPModel(X, Y, likelihoods[i],self.kernel,self.kernel_params,self.mean_func,self.ref_emulator.models[i],self.ref_emulator.likelihoods[i],None,False,self.poly_degree,norm_const))
            if self.train==True:
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
                    # print('Iter %d/%d - Loss: %.3f' % (
                    #     j + 1, training_iter, loss.item()
                    # ))
                    optimizer.step()
                    
        return models, likelihoods
    
    
    def create_ensemble_restarts(self):
        test_models=[]
        test_likelihoods=[]
        
        best_models=[]
        best_likelihoods=[]
        
        R2_vec=torch.zeros(self.n_restarts,self.y_test.shape[1])
        for i in range(self.n_restarts):
            
            self.models, self.likelihoods = self.create_ensemble()
            test_models.append(self.models)
            test_likelihoods.append(self.likelihoods)
            R2_vec[i]=self.R2(self.X_test,self.y_test)
        print(R2_vec)
        
        best = torch.argmax(R2_vec,axis=0)
        
        print(best)
        for j in range(self.y_test.shape[1]):
            best_models.append(test_models[best[j]][j])
            best_likelihoods.append(test_likelihoods[best[j]][j])
        
        return best_models, best_likelihoods
            
    def save_emulator_state(self,save_loc):
        dicts=[]
        for model in self.models:
            dicts.append(model.state_dict())
        torch.save(dicts,save_loc)   
        
    def load_emulator_state(self,save_loc):
        load = torch.load(save_loc)
        for i,model in enumerate(self.models):
            model.load_state_dict(load[i])    
    
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
        inputVals = ((inputVals-self.training_input_mean)/self.training_input_STD).float()
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
        MSE_score=torch.zeros(n,outputVals.shape[1])
        outputVals = outputVals
        pred = self.predict_sample(inputVals,n)
        for i in range(n):
            MSE_score[i,:]=((pred[:,i,:]-outputVals)**2).mean(axis=0)
     
        MSE_mean = torch.tensor(MSE_score.mean(axis=0))
        MSE_std = torch.tensor(MSE_score.std(axis=0))
        return MSE_mean, MSE_std

    def R2(self,inputVals,outputVals):
        R2_score=1-self.MSE(inputVals,outputVals)/torch.var(outputVals,axis=0)
        return R2_score
    
    def R2_sample(self,inputVals,outputVals,n=5):
        R2_score=torch.zeros(n,outputVals.shape[1])
        pred = self.predict_sample(inputVals,n)
        for i in range(n):
            R2_score[i,:]=(1-((pred[:,i,:]-outputVals)**2).mean(axis=0)/torch.var(outputVals,axis=0))             
        R2_mean = R2_score.mean(axis=0)
        R2_std = R2_score.std(axis=0)
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
            sigma_2 = sigma2[i]
            m = likelihoods[i](models[i](inputNorm)).mean
            k = likelihoods[i](models[i](inputNorm)).variance
            
            mean = self.training_output_STD[i]*m+self.training_output_mean[i]
            variance = (self.training_output_STD[i]**2)*k+sigma_2

            likelihood_manual=self.gaussian_ll(outputVal[:,i],mean,variance)

            #likelihood_manual=-0.5*((outputVal[:,i]-(self.training_output_STD[i]*m+self.training_output_mean[i]))**2)/((self.training_output_STD[i]**2)*k+sigma) -0.5*torch.log((self.training_output_STD[i]**2)*k+sigma) - 0.5*torch.log(torch.tensor(2*torch.pi))
            likelihood_eval[i,:] = likelihood_manual
          
            
        return likelihood_eval 
    
    
    def generate_variance(self,candidateInput):
        nMod = self.training_output_normalised.shape[1]
        nDim = self.training_output_normalised.shape[0]
        nP = candidateInput.shape[0]
        models=self.models
        likelihoods=self.likelihoods
        var = torch.zeros((nMod,nP))
        inputNorm,outputNorm = self.normalise_test_data(candidateInput,1)
        inputNorm=inputNorm.float()
        outputNorm=outputNorm.float()
        
        
        for i in range(nMod):
            models[i].eval()
            likelihoods[i].eval()
            k = likelihoods[i](models[i](inputNorm)).variance
            variance = (self.training_output_STD[i]**2)*k
            
            var[i,:] = variance
          
            
        return var
   
    
    def ensemble_log_likelihood_obs_error_no_U(self,candidateInput,outputVal,sigma2):
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
            sigma_2 = sigma2[i]
            m = likelihoods[i](models[i](inputNorm)).mean
            
            mean = self.training_output_STD[i]*m+self.training_output_mean[i]
            variance = sigma_2

            likelihood_manual=self.gaussian_ll(outputVal[:,i],mean,variance)

            #likelihood_manual=-0.5*((outputVal[:,i]-(self.training_output_STD[i]*m+self.training_output_mean[i]))**2)/((self.training_output_STD[i]**2)*k+sigma) -0.5*torch.log((self.training_output_STD[i]**2)*k+sigma) - 0.5*torch.log(torch.tensor(2*torch.pi))
            likelihood_eval[i,:] = likelihood_manual
          
            
        return likelihood_eval    
    
    def gaussian_ll(self,y,mean,variance):
        ll = -0.5*((y-mean)**2)/variance - 0.5*torch.log(variance) - 0.5*torch.log(torch.tensor(2*torch.pi))
        return ll