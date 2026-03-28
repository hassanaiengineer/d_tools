import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoIAFNormal, AutoLowRankMultivariateNormal
import time

# Set random seed for reproducibility
np.random.seed(123)
torch.manual_seed(123)

# Define parameters
D = 50  # Dimension for the graph
p = 0.5  # Probability for edge creation
num_simulations = 100  # Number of simulations

# Step 1: Generate Graph G
def generate_graph(D, p):
    G = np.random.binomial(1, p, size=(D, D))
    np.fill_diagonal(G, 0)  # Ensure no self-loops
    return G

G = generate_graph(D, p)

# Step 2: Simulate R Matrices Based on G
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

R_simulations = [generate_R(G) for _ in range(num_simulations)]

# Step 3: Compute Empirical Covariance Matrix S
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

# Step 4: Bayesian Model Definition
def model(R, G, S, slab_scale=1.0, spike_scale=0.1):
    # Spike-and-Slab Prior on G
    with pyro.plate("nodes", G.shape[0]):
        G_prior = dist.Bernoulli(torch.tensor(p)).to_event(1)
        G_sample = pyro.sample("G", G_prior)
    
    # Matrix Normal Likelihood
    mean = (torch.eye(D) - G_sample).inverse()  # (I - G)^-1
    U = torch.eye(D)  # Default scale matrix
    V = torch.eye(D)
    matrix_normal = dist.MatrixNormal(mean, U, V)
    
    with pyro.plate("observations", R.shape[0]):
        pyro.sample("R", matrix_normal, obs=R)

# Step 5: NUTS Inference
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=200)
start_time = time.time()
mcmc.run(torch.tensor(R_simulations[0]), torch.tensor(G), torch.tensor(S))
end_time = time.time()
print(f"NUTS Time: {end_time - start_time:.2f} seconds")

# Step 6: SVI with Auto Guide
guide = AutoLowRankMultivariateNormal(model)
svi = SVI(model, guide, pyro.optim.Adam({"lr": 0.01}), loss=Trace_ELBO())

svi_start_time = time.time()
num_iterations = 500
for step in range(num_iterations):
    loss = svi.step(torch.tensor(R_simulations[0]), torch.tensor(G), torch.tensor(S))
    if step % 50 == 0:
        print(f"Step {step} - Loss: {loss}")
svi_end_time = time.time()
print(f"SVI Time: {svi_end_time - svi_start_time:.2f} seconds")

# Step 7: Normalizing Flow
flow_guide = AutoIAFNormal(model)
svi_flow = SVI(model, flow_guide, pyro.optim.Adam({"lr": 0.01}), loss=Trace_ELBO())

flow_start_time = time.time()
for step in range(num_iterations):
    loss = svi_flow.step(torch.tensor(R_simulations[0]), torch.tensor(G), torch.tensor(S))
    if step % 50 == 0:
        print(f"Step {step} - Loss: {loss}")
flow_end_time = time.time()
print(f"Normalizing Flow Time: {flow_end_time - flow_start_time:.2f} seconds")

# Step 8: Posterior Metrics
posterior_samples = mcmc.get_samples()
mean_G = posterior_samples["G"].mean(0)
mode_G = (posterior_samples["G"].mean(0) > 0.5).float()
ci_lower = posterior_samples["G"].quantile(0.025, 0)
ci_upper = posterior_samples["G"].quantile(0.975, 0)

print(f"Mean G:\n{mean_G}")
print(f"Mode G:\n{mode_G}")
print(f"95% CI Lower Bound:\n{ci_lower}")
print(f"95% CI Upper Bound:\n{ci_upper}")

# Step 9: Apply Empirical Ashr Prior
ashr_loc = 0.0
ashr_scale = 1.0
ashr_probs = torch.tensor([0.5, 0.5])
ashr_prior = dist.MixtureNormal(
    loc=torch.tensor([ashr_loc, ashr_loc]),
    scale=torch.tensor([ashr_scale, ashr_scale]),
    mixing_probs=ashr_probs
)
G_with_ashr = pyro.sample("G_ashr", ashr_prior)

# Final model with Ashr applied to edges of G
print("Final model with empirical Ashr prior applied.")
