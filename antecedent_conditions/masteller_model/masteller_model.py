# Trying Masteller model
# November 24th, 2025
# Using Particle Swarm Optimization (PSO) for an initial guess
# Using Nonlinear Least Squares (NLS) for an exhaustive optimization

"0. Directory and libraries --------------------------------------------------------"

# Checking working directory and environment
import os
import sys
os.getcwd()
sys.prefix

# Importing required libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyswarm import pso
from scipy.optimize import curve_fit

"1. Files location -----------------------------------------------------------------"

# csv file with the summary
csv_file = "D:/Flume data/tauc_summary_python.csv"
all_data = pd.read_csv(csv_file)

"2. Running PSO for an initial guess -----------------------------------------------"

# k1, k2 and k3 are fitting parameters
# k3 = epsilon
# x1 = flow magnitude, tau*
# x2 = flow duration, t in minutes
tauc_max = np.max(all_data["tau*c"])
tauc_min = np.min(all_data["tau*c"])
tau_ref = all_data["tau*c"].iloc[0]
gamma = 2.5
all_data["B"] = (tauc_max - all_data["tau*c_ini"])/(tauc_max - tauc_min)

# Definning function for PSO
def objective_function(K, X, Y, tauc, gamma, B):
    # Getting parameters
    k1, k2, k3 = K[0], K[1], K[2]
    x1, x2 = X[0], X[1]

    # Computing the estimated Y0 when H = 0, tau*/tau*c < 1
    #Y_est0 = (k1*x2*B)/(1 + (x1/tauc)**-gamma)
    # Computing the estimated Y1 when H = 1, tau*/tau*c > 1
    Y_est1 = (k1*x2*B)/(1 + (x1/tauc)**-gamma) - k2*(1-B)*x2*(x1/tauc - 1)**k3

    # Calculating MAE instead of SSE
    #MAE = np.mean(abs(Y - Y_est0 - Y_est1))
    MAE = np.mean(abs(Y - Y_est1))
    
    return MAE

# Setting lower bounds for k1, k2. amd k3
K = np.array([1*10^-3, 1*10^-3, 5])
lower_bounds = np.array([1*10^-6, 1*10^-6, 0])
upper_bounds = np.array([1, 1, 10])
X = np.row_stack((all_data["tau*"], all_data["dt"]))
Y = all_data["dtau*c"]
B = all_data["B"]
tauc = all_data["tau*c_ini"]

# Running the PSO optimization
k_opt_pso, fval = pso(
    func=objective_function,
    lb=lower_bounds,
    ub=upper_bounds,
    args=(X, Y, tauc, gamma, B),
    swarmsize=100,  # Number of particles (default is usually 10)
    maxiter=1000,    # Maximum iterations (default is usually 100)
    debug=False
)
all_data["dtau*c"]
print(f"Optimal parameters found by PSO (k1, k2, k3): {k_opt_pso}")
print(f"Minimum MAE (fval): {fval:.4f}")

"3. Integrating with NLS for a robust optimization ---------------------------------"

# Definning the model function for NLS
def model_function(X, k1, k2, k3, tauc, gamma, B): #tauc, gamma, B
    # Getting parameters
    #k1, k2, k3 = K[0], K[1], K[2]
    x1, x2 = X[0], X[1]

    # Computing Y with the model
    return (k1*x2*B)/(1 + (x1/tauc)**-gamma) - k2*(1-B)*x2*(x1/tauc - 1)**k3

def get_model_function(tauc, gamma, B):

    def model_function_nls(X, k1, k2, k3): #tauc, gamma, B
        # Getting parameters
        #k1, k2, k3 = K[0], K[1], K[2]
        x1, x2 = X[0], X[1]

        # Computing Y with the model
        return (k1*x2*B)/(1 + (x1/tauc)**-gamma) - k2*(1-B)*x2*(x1/tauc - 1)**k3
    
    return model_function_nls

model_function = get_model_function(tauc, gamma, B)

# Running the NLS
initial_guess = k_opt_pso

p_opt_nls, p_cov_nls = curve_fit(
    f=model_function, 
    xdata=X, 
    ydata=np.array(Y), 
    p0=initial_guess)

# Extract the optimal parameters
k1_opt, k2_opt, k3_opt = p_opt_nls
perr = np.sqrt(np.diag(p_cov_nls)) # Standard deviations

print("\n--- Final NLS Refinement Results ---")
print(f"Final Optimal k1: {k1_opt:.4f}")
print(f"Final Optimal k2: {k2_opt:.4f}")
print(f"Final Optimal k3: {k3_opt:.4f}")

# Calculate final SSE
Y_estimated_nls = model_function(X, *p_opt_nls)
residuals = all_data["dtau*c"] - Y_estimated_nls
sse = np.sum(residuals**2)
mse = sse / len(Y_estimated_nls)
final_rmse = np.sqrt(mse)
print(f"Final SSE after NLS refinement: {final_rmse:.4f}")