# -*- coding: utf-8 -*-
"""
Wrapper for Executing SWAT+ Model

Adapated from Jaya Hafner, Kalcic Lab @ UW Madison

Last updated: 08/14/2025

@author: kbdon
"""

import numpy as np
import pandas as pd
import subprocess
import os
import io
from pathlib import Path
import sys
import shutil
import torch
from datetime import datetime, timedelta
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time


class SWATrun:
    def __init__(self):
        """
        Defines parameter values and ranges for Single-Field SWAT Model.
        -----------------------------------------------------------------------
        param_list: List where oarameter names and bounds for SWAT model are specified.
            -> Parameter name should also have appropriate SWAT file extension.
        df_param: Dataframe of Param_list items.
        LB: Tensor of lower bounds for parameters.
        UB: Tensor of upper bounds for parameters.
             
        """

        self.wetland = "wetf" # Options: "wehb" or "wetf"
        
        if self.wetland == "wehb":
            self.nutrient_res = [("N_STL", 0.003188, 2.520617),
                                 ("P_STL", 0.003674, 1.021552)]
            self.sol_p = torch.tensor([20.8])
            self.sol_nox = torch.tensor([19.8])
            self.sol_nh3 = torch.tensor([28.2])

        if self.wetland == "wetf":
            self.nutrient_res = [("N_STL", 0.005424, 0.364669),
                                 ("P_STL", 0.003117, 0.180093)]
            self.sol_p = torch.tensor([32.1])
            self.sol_nox = torch.tensor([21.4])
            self.sol_nh3 = torch.tensor([59.4])
        
        self.ground_truth = torch.stack([self.sol_p,self.sol_nox,self.sol_nh3]).squeeze()        
        self.param_list = self.nutrient_res
        self.theta_dim = len(self.param_list)
        self.df_param = pd.DataFrame(self.param_list,columns=["parameter","LB","UB"])
        self.LB = torch.tensor(self.df_param.iloc[:,1].tolist())
        self.UB = torch.tensor(self.df_param.iloc[:,2].tolist())
               
        # Define paths to input, output, and executable:  
        self.output_wetland = "C:\\TurtleCreek_Calibration\\04100010_" + self.wetland
        self.nutrient_path = self.output_wetland + "\\wetland_yr.txt"
        
        # Nominal parameters path to search inputs files for:
        self.nutrient_nom_path = "C:\\TurtleCreek_Calibration\\Nominal_Inputs\\nutrients.res"

        # Parameter iteration files to add new thetas to:
        self.nutrient_iter_path = "C:\\TurtleCreek_Calibration\\Input_Iterations\\nutrients.res"                 
                
        
    def __call__(self, theta):
        
        pd.set_option('display.max_colwidth', None)
        
        file_name = 'nutrients'
        DefaultPath = self.nutrient_nom_path
        InputPath = self.output_wetland + "\\" + file_name + ".res"
        
        self.n_stl = f'{theta[0].squeeze(0).tolist():.4f}'
        self.p_stl = f'{theta[1].squeeze(0).tolist():.4f}'
        self.columns = f'{1}'.rjust(9) + f'{1}'.rjust(11) + f'{1.08}'.rjust(9) + f'{1.08}'.rjust(9) + f'{0}'.rjust(11) + f'{0}'.rjust(11)
        
        self.new_line = pd.read_csv(self.nutrient_iter_path, header=None).loc[0].to_string(index=False) + self.n_stl.rjust(11) + self.n_stl.rjust(7) + self.p_stl.rjust(11) + self.p_stl.rjust(7) + self.columns
        
        self.old_line = pd.read_csv(self.nutrient_nom_path, header=None).loc[24].to_string(index=False)

        shutil.copy(DefaultPath, InputPath)
        with open(InputPath, 'r') as file:  # Read in the .res file
            filedata = file.read() 
        
        # Find old lines in .sol file, replaces it with new theta:
        filedata = filedata.replace(self.old_line, self.new_line)

        with open(InputPath, 'w') as file:  # Write the file out again
            file.write(filedata)
        file.close() 

        #######################################################################
        # Executing SWAT run
        #######################################################################

        start = time.time()
        print('Running SWAT...')
        project_path = self.output_wetland
        self.exec_path = "swatplus-61.0.2.11-291-gd109457-ifx-win_amd64-Dbg.exe"
        self.swat_exe = os.path.join(project_path, self.exec_path)
        subprocess.run([self.swat_exe], cwd=project_path)
        end = time.time()
        print('SWAT run complete in' + ' ' + f'{end-start:.4f}' + ' ' + 'seconds.')

        #######################################################################
        # Obtaining Necessary Outputs
        #######################################################################

        self.nutrients = pd.read_fwf(self.nutrient_path, skiprows=2)
        self.wetland_code = 827770
        self.gis_indices = torch.tensor(self.nutrients.to_numpy()[:,5].astype(float))
        self.mask = (self.gis_indices == self.wetland_code)
        
        self.flow_out = torch.tensor(self.nutrients.to_numpy()[:,47].astype(float))[self.mask]
        self.no3_out = torch.tensor(self.nutrients.to_numpy()[:,51].astype(float))[self.mask]
        self.solp_out = torch.tensor(self.nutrients.to_numpy()[:,52].astype(float))[self.mask]
        self.nh3_out = torch.tensor(self.nutrients.to_numpy()[:,54].astype(float))[self.mask]
        self.no2_out = torch.tensor(self.nutrients.to_numpy()[:,55].astype(float))[self.mask]
        
        self.nox_out = self.no3_out + self.no2_out
        
        # Calculating averages, concentrations, and conducting dimensional analysis:
        FLOW = torch.mean(self.flow_out)*1000       # Converting from m^3 to liters
        SOLP = torch.mean(self.solp_out)*1e+9       # Converting from kg to micrograms
        NOX = torch.mean(self.nox_out)*1e+9         # Converting from kg to micrograms
        NH3 = torch.mean(self.nh3_out)*1e+9         # Converting from kg to micrograms
        
        C_SOLP = SOLP / FLOW
        C_NOX = NOX / FLOW
        C_NH3 = NH3 / FLOW
                      
        # Stacking outputs in order as seen in obtileQ:
        
        sensors = torch.stack([C_SOLP,C_NOX,C_NH3])    

        return sensors


if __name__== '__main__':
    a = SWATrun()
    dim = a.theta_dim
    run_type = ['Input'] # Types accepted: ['Rand','Input']
    
    plotting = False # Option for turning plotting on/off
    
    if run_type == ['Rand']:
        theta = torch.rand(dim)
        
        # Rescaling:
        LB = a.LB
        UB = a.UB
    
        theta_scaled = LB + (UB - LB)*theta
        sensors = a(theta_scaled)
        
    if run_type == ['Input']:
        theta = torch.tensor([0.168, 0.126])
    
        # Rescaling:
        LB = a.LB
        UB = a.UB
    
        theta_scaled = LB + (UB - LB)*theta
        sensors = a(theta_scaled)
       
                