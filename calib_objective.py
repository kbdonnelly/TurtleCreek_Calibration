#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibration Objective for Environmental Model Calibration
@author: kbdon

Last updated: 11/13/2025
"""
import sys
import torch
from torch import Tensor
from SWATrun import SWATrun
from scipy.stats import qmc
import pandas as pd
import numpy as np
from turbo_1 import Turbo1

import matplotlib.pyplot as plt

# torch.set_default_dtype(torch.double)

class ObjFunc:
    def __init__(self):
        self.simulator = SWATrun()
        self.dim = self.simulator.theta_dim

        self.LB = self.simulator.LB
        self.UB = self.simulator.UB
        
    def __call__(self,theta,rescaled=True):  
        
        # theta = self.LB + (self.UB - self.LB)*theta
        
        # Running model to obtain desired outputs:    
        sensors = self.simulator(theta)
        
        # Obtaining ground truth data from simulator:
        ground_truth = self.simulator.ground_truth
      
        output = torch.zeros(len(sensors))

        for i in range(len(sensors)):               
            output[i] = torch.sqrt(torch.square(sensors[i]-ground_truth[i]))/ground_truth[i]
             
        return torch.sum(output)

if __name__== '__main__':
    
    simulator = SWATrun()
    f = ObjFunc()
    dim = simulator.theta_dim
    run_type = ['TuRBO-1'] # Types accepted: ['Rand','Sobol','Input','TuRBO-1']
    plotting = False # Option for turning plotting on/off
    seed = 0    
    
    if run_type == ['Rand']:
        
        theta = torch.rand(1,dim)
        LB = simulator.LB
        UB = simulator.UB
        
        output = f(theta.squeeze(0))
    
        
    if run_type == ['Sobol']:
        
        theta = torch.quasirandom.SobolEngine(dimension=dim,  scramble=True, seed=seed).draw(10)
        
        
    if run_type == ['Input']:
        
        theta = torch.quasirandom.SobolEngine(dimension=dim,  scramble=True, seed=seed).draw(10)    
        
 
    if run_type == ['TuRBO-1']:
               
        f = ObjFunc()
        turbo1 = Turbo1(
             f = f,  # Handle to objective function
             lb = np.array(f.LB),  # Numpy array specifying lower bounds
             ub = np.array(f.UB),  # Numpy array specifying upper bounds
             n_init = 2*dim,  # Number of initial bounds from an Latin hypercube design
             max_evals = 500,  # Maximum number of evaluations
             batch_size = 10,  # How large batch size TuRBO uses
             verbose = True,  # Print information from each batch
             use_ard = True,  # Set to true if you want to use ARD for the GP kernel
             max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
             n_training_steps = 50,  # Number of steps of ADAM to learn the hypers
             min_cuda = 1024,  # Run on the CPU for small datasets
             device = "cpu",  # "cpu" or "cuda"
             dtype = "float64",  # float64 or float32
             seed=seed
         )
        turbo1.optimize()
        
        X = turbo1.X  # Evaluated points
        fX = turbo1.fX  # Observed values
        ind_best = np.argmin(fX)
        f_best, x_best = fX[ind_best], X[ind_best, :]
        
        print("Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best, np.around(x_best, 3)))
        
        df_theta_TuRBO1 =  pd.DataFrame(X)
        df_theta_TuRBO1.to_csv('df_theta_wetf.csv', sep=',', index = False, encoding='utf-8')
        
        df_output_TuRBO1 =  pd.DataFrame(fX)
        df_output_TuRBO1.to_csv('df_output_wetf.csv', sep=',', index = False, encoding='utf-8')

    if plotting == True:

        fig, ax = plt.subplots(figsize=(8, 6))  


