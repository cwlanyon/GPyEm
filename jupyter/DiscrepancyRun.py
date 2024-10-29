import math
import torch
import gpytorch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
import GPE_ensemble as GPE

import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
#from GPErks.gp.data.dataset import Dataset
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from torchmetrics import MeanSquaredError, R2Score

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score


#from GPErks.gp.experiment import GPExperiment
#from GPErks.train.emulator import GPEmulator
#from GPErks.perks.inference import Inference
#from GPErks.train.early_stop import NoEarlyStoppingCriterion
#from GPErks.train.early_stop import (
#    GLEarlyStoppingCriterion,
#    PQEarlyStoppingCriterion,
#    UPEarlyStoppingCriterion,
#)
#from GPErks.train.early_stop import PkEarlyStoppingCriterion



# set logger and enforce reproducibility
#from GPErks.log.logger import get_logger
#from GPErks.utils.random import set_seed
#log = get_logger()
seed = 7
#set_seed(seed)
from time import process_time
import scipy
from scipy.optimize import minimize

def m0_mat(y_test,emulators,x_test,output):

    m0=torch.zeros((y_test.shape[0],len(emulators)))
    for i in range(len(emulators)):
        m0[:,i]=(emulators[i].predict(x_test)[:,output]-y_train.mean(axis=0)[output])/y_train.std(axis=0)[output]


    return m0

def proxy(a,y_train,m0,output):
    m_t = (m0-y_train.mean(axis=0))/y_train.std(axis=0)
    y_t = (y_train-y_train.mean(axis=0))/y_train.std(axis=0)
    a=torch.tensor(a)
    res = ((a*m_t-y_t)**2).mean(axis=0).detach().numpy()
    return res[output]

mode_weights = pd.read_csv(
    r'/Users/pmzcwl/Library/CloudStorage/OneDrive-TheUniversityofNottingham/shared_simulations/modes_weights.csv',
    index_col=0, delim_whitespace=False, header=0)

mode_weights

# mode_weights=mode_weights.drop(15,axis=0)
# mode_weights=mode_weights.drop(14,axis=0)

meshes = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
          '19']

x_labels = pd.read_csv(
    r'/Users/pmzcwl/Library/CloudStorage/OneDrive-TheUniversityofNottingham/shared_simulations/EP_healthy/input/xlabels_EP.txt',
    delim_whitespace=True, header=None)
x_labels = x_labels.values.flatten().tolist() + mode_weights.columns.tolist()

y_labels = pd.read_csv(
    r'/Users/pmzcwl/Library/CloudStorage/OneDrive-TheUniversityofNottingham/shared_simulations/EP_healthy/output/ylabels.txt',
    delim_whitespace=True, header=None)

all_input = []
all_output = []
all_x = []
for i in range(len(meshes)):
    val = meshes[i]

    inputData = pd.read_csv(
        "/Users/pmzcwl/Library/CloudStorage/OneDrive-TheUniversityofNottingham/shared_simulations/EP_healthy/" + val + "/X_EP.txt",
        index_col=None, delim_whitespace=True, header=None).values
    outputData = pd.read_csv(
        "/Users/pmzcwl/Library/CloudStorage/OneDrive-TheUniversityofNottingham/shared_simulations/EP_healthy/" + val + "/Y.txt",
        index_col=None, delim_whitespace=True, header=None).values
    modeweights = np.tile(mode_weights.iloc[i, :].values, (inputData.shape[0], 1))
    input_modes = np.concatenate((inputData, modeweights), axis=1)
    all_x.append(torch.tensor(inputData))
    all_input.append(torch.tensor(input_modes))
    all_output.append(torch.tensor(outputData))
    print(val)
    print(np.max(outputData))
# all_input=pd.concat(all_input)
# all_output=pd.concat(all_output
# all_input.columns=x_labels
# all_output.columns=y_labels
train_input = []
test_input = []
train_output = []
test_output = []

train_input_modes = []
test_input_modes = []
train_output_modes = []
test_output_modes = []

for i in range(len(meshes)):
    X = all_x[i]
    y = all_output[i]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed + i
    )
    train_input.append(X_train)
    test_input.append(X_test)
    train_output.append(y_train)
    test_output.append(y_test)

for i in range(len(meshes)):
    X = all_input[i]
    y = all_output[i]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed + i
    )
    train_input_modes.append(X_train)
    test_input_modes.append(X_test)
    train_output_modes.append(y_train)
    test_output_modes.append(y_test)
emulators=[]
for i in range(len(meshes)):
    emulators.append(GPE.ensemble(train_input[i],train_output[i],mean_func="linear",training_iter=1000))

model = Pipeline(steps=[
    # ('scaler', StandardScaler()),
    ('preprocessor', PolynomialFeatures(degree=1, include_bias=False, interaction_only=False)),
    ('lasso', LassoCV(n_alphas=1000, max_iter=10000))
])

reps = 5
nn = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 80, 100, 120, 140]
R2 = torch.zeros(7, len(nn), 2, reps)
ISE = torch.zeros(7, len(nn), 2, reps)
Ti = torch.zeros(7, len(nn), reps)

for num, n in enumerate(nn):
    for k in range(len(emulators)):
        emulators2 = emulators.copy()
        emulators2.pop(k)
        print(len(emulators2))

        X_train = train_input[k]
        y_train = train_output[k]
        X_test = test_input[k]
        y_test = test_output[k]

        for i in range(reps):

            # b=np.random.choice(range(X_train.shape[0]),n,replace=False)

            X = X_train
            y = y_train
            X_train1, X_test1, y_train1, y_test1 = train_test_split(
                X,
                y,
                train_size=n,
                random_state=i
            )

            start = time.time()
            model_f = GPE.ensemble(X_train1, y_train1, mean_func="linear", training_iter=500)
            end = time.time()
            R2temp, R2std = model_f.R2_sample(X_test, y_test, 1000)
            R2[0, num, :, i] += R2temp / (len(emulators))
            ISE[0, num, :, i] += model_f.ISE(X_test, y_test) / (len(emulators))

            Ti[0, num, i] += (end - start) / (len(emulators))

            em = np.random.randint(len(emulators2))
            start = time.time()
            model_dc_1 = GPE.ensemble(X_train1, y_train1, mean_func="discrepancy_cohort", training_iter=500,
                                      ref_emulator=[emulators2[em]], a=torch.tensor([[1], [1]]))
            end = time.time()
            R2temp, R2std = model_dc_1.R2_sample(X_test, y_test, 1000)
            R2[1, num, :, i] += R2temp / (len(emulators))
            ISE[1, num, :, i] += model_dc_1.ISE(X_test, y_test) / (len(emulators))
            print(model_dc_1.R2(X_test, y_test))
            print(R2[1])

            Ti[1, num, i] += (end - start) / (len(emulators))

            start = time.time()
            m0 = emulators2[em].predict(X_train1)
            a_d = np.zeros((y_train.shape[1], 1))
            for l in range(y_train.shape[1]):
                result = scipy.optimize.minimize(proxy, 1, args=(y_train1, m0, l), method='Nelder-Mead', tol=1e-8)
                print(result.x)
                a_d[l] = result.x
            a_d = torch.tensor(a_d)
            model_dc_reg = GPE.ensemble(X_train1, y_train1, mean_func="discrepancy_cohort", training_iter=500,
                                        ref_emulator=[emulators2[em]], a=a_d)
            end = time.time()
            R2temp, R2std = model_dc_reg.R2_sample(X_test, y_test, 1000)
            R2[2, num, :, i] += R2temp / (len(emulators))
            ISE[2, num, :, i] += model_dc_reg.ISE(X_test, y_test) / (len(emulators))

            Ti[2, num, i] += (end - start) / (len(emulators))

            start = time.time()
            model_dc_learned = GPE.ensemble(X_train1, y_train1, mean_func="discrepancy_cohort", training_iter=500,
                                            ref_emulator=[emulators2[em]])
            end = time.time()
            R2temp, R2std = model_dc_learned.R2_sample(X_test, y_test, 1000)
            R2[3, num, :, i] += R2temp / (len(emulators))
            ISE[3, num, :, i] += model_dc_learned.ISE(X_test, y_test) / (len(emulators))

            Ti[3, num, i] += (end - start) / (len(emulators))

            start = time.time()
            model_dc_all = GPE.ensemble(X_train1, y_train1, mean_func="discrepancy_cohort", training_iter=500,
                                        ref_emulator=emulators2)
            end = time.time()
            R2temp, R2std = model_dc_all.R2_sample(X_test, y_test, 1000)
            R2[4, num, :, i] += R2temp / (len(emulators))
            ISE[4, num, :, i] += model_dc_all.ISE(X_test, y_test) / (len(emulators))

            Ti[4, num, i] += (end - start) / (len(emulators))

            start = time.time()
            a_d = torch.zeros((y_train1.shape[1], len(emulators2)))
            for j in range(y_train1.shape[1]):
                m0 = m0_mat(y_train1, emulators2, X_train1, j)
                # fit to an order-3 polynomial data
                y_t = (y_train1[:, j] - y_train1.mean(axis=0)[j]) / y_train1.std(axis=0)[j]
                model = model.fit(m0.detach().numpy(), y_t.detach().numpy())
                a_d[j] = torch.tensor(model.named_steps['lasso'].coef_)

            model_dc_lasso = GPE.ensemble(X_train1, y_train1, mean_func="discrepancy_cohort", training_iter=500,
                                          ref_emulator=emulators2, a=a_d)
            end = time.time()
            R2temp, R2std = model_dc_lasso.R2_sample(X_test, y_test, 1000)
            R2[5, num, :, i] += R2temp / (len(emulators))
            ISE[5, num, :, i] += model_dc_lasso.ISE(X_test, y_test) / (len(emulators))

            Ti[5, num, i] += (end - start) / (len(emulators))

            start = time.time()
            model_dc_lasso_learned = GPE.ensemble(X_train1, y_train1, mean_func="discrepancy_cohort", training_iter=500,
                                                  ref_emulator=emulators2, a=a_d, a_indicator=True)
            end = time.time()
            R2temp, R2std = model_dc_lasso_learned.R2_sample(X_test, y_test, 1000)
            R2[6, num, :, i] += R2temp / (len(emulators))
            ISE[6, num, :, i] += model_dc_lasso_learned.ISE(X_test, y_test) / (len(emulators))

            Ti[6, num, i] += (end - start) / (len(emulators))

R2_save = R2.reshape(7,len(nn)*reps*y_train.shape[1])

np.savetxt("DiscrepR2TrainNVaryDefinitive.csv", R2_save.detach().numpy(), delimiter=",")

ISE_save = ISE.reshape(7,len(nn)*reps*y_train.shape[1])

np.savetxt("DiscrepISETrainNVaryDefinitive.csv", ISE_save.detach().numpy(), delimiter=",")