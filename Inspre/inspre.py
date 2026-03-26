import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.matrix_normal import MatrixNormal
from pyro.distributions import SpikeAndSlab, MixtureNormal
import pyro
import pyro.infer
import pyro.optim
import time

# Set random seed for reproducibility
np.random.seed(123)
torch.manual_seed(123)

# Define parameters
D = 50  # Dimension for the graph
graph_type = 'random'
p = 0.5  # Probability for edge creation

# Generate a random graph G
def generate_graph(D, graph_type, p):
    G = np.random.binomial(1, p, size=(D, D))
    if graph_type == 'random':
        return G
    else:
        raise ValueError("Only 'random' graph type is supported.")

G = generate_graph(D, graph_type, p)

# Function to simulate one instance of R given G
def generate_R(G):
    graph_size = G.shape[0]
    R = np.zeros((graph_size, graph_size))
    
    for i in range(graph_size):
        for j in range(graph_size):
            if G[i, j] == 1:
                R[i, j] = np.random.normal(1, 0.5)
            else:
                R[i, j] = np.random.normal(0, 0.5)
    
    return R

# Simulate multiple instances of R and store them
num_simulations = 100
R_simulations = [generate_R(G) for _ in range(num_simulations)]

# Calculate empirical covariance matrix S
def calculate_covariance_matrix_S(R_simulations):
    graph_size = R_simulations[0].shape[0]
    S = np.zeros((graph_size, graph_size, graph_size, graph_size))
    
    for i in range(graph_size):
        for j in range(graph_size):
            for k in range(graph_size):
                for l in range(graph_size):
                    R_ij_values = [R[i, j] for R in R_simulations]
                    R_kl_values = [R[k, l] for R in R_simulations]
                    S[i, j, k, l] = np.cov(R_ij_values, R_kl_values)[0, 1]
    
    return S

S = calculate_covariance_matrix_S(R_simulations)

# Set up the device (CPU or GPU)
device = 'cpu'

# Define the function for Kronecker ALS
def kronecker_als_efficient(S, D, max_iter=100, tol=1e-6):
    # Move data to the device
    S = torch.tensor(S, device=device)

    U = torch.rand(D, D, device=device)
    V = torch.rand(D, D, device=device)
    residuals = []
    total_norm = torch.norm(S, p='fro')

    for it in range(1, max_iter + 1):
        # Reshape S to a tensor of shape (D, D, D, D)
        S_tensor = S.reshape(D, D, D, D)

        # Update U
        VVT = V @ V.T
        for i in range(D):
            for j in range(D):
                numerator = torch.sum(S_tensor[i, :, j, :] * VVT)
                denominator = torch.sum(VVT * VVT)
                U[i, j] = numerator / (denominator + 1e-8)

        # Update V
        UUT = U @ U.T
        for i in range(D):
            for j in range(D):
                numerator = torch.sum(S_tensor[:, i, :, j] * UUT)
                denominator = torch.sum(UUT * UUT)
                V[i, j] = numerator / (denominator + 1e-8)

        # Compute residual
        S_approx = torch.kron(U, V)
        residual = torch.norm(S - S_approx, p='fro')
        residuals.append(residual.item())
        print(f"Iteration {it}, Residual: {residual.item():.6f}")

        if residual / total_norm < tol:
            break

    return U, V, residuals

# Run the Kronecker ALS function
start_time = time.time()
U_efficient, V_efficient, residuals_efficient = kronecker_als_efficient(S, D, max_iter=50, tol=1e-4)
end_time = time.time()
print(f"Final residual after {len(residuals_efficient)} iterations: {residuals_efficient[-1]:.6f}")
print(f"Time taken: {end_time - start_time:.2f} seconds")

# Apply spike-and-slab prior to G
slab_scale = 1.0
spike_scale = 0.1
prior_probs = (0.5, 0.5)  # (spike, slab) probabilities
G_prior = SpikeAndSlab(torch.tensor(G, device=device), slab_scale, spike_scale, prior_probs)

# Apply Bayesian inference methods
# NUTS (MCMC)
# SVI (Stochastic Variational Inference)
# Normalizing Flow

# Compute posterior metrics
# - Mode
# - Mean
# - 95% CI
# - Edge Presence Accuracy

# Apply empirical Ashr prior
ashr_loc = 0.0
ashr_scale = 1.0
ashr_mixture_probs = (0.5, 0.5)
G_ashr_prior = MixtureNormal(torch.tensor(G, device=device), ashr_loc, ashr_scale, ashr_mixture_probs)

# Final model
# Apply the final model with the Ashr prior to the dataset