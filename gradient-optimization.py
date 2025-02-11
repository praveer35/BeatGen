import torch
from Bayesian_Opt_Engine import Bayesian_Opt_Engine


chords = [6, 1, 2, 3]
melody = [[7, 0.5], [4, 0.5], [2, 0.5],
          [1, 1.0], [2, 0.5], [3, 0.5],
          [2, 0.5], [4, 1.0], [3, 1.0],
          [4, 0.5], [2, 1.0], [4, 0.5],
          [7, 0.5], [4, 0.5], [2, 0.5],
          [1, 1.0], [2, 0.5], [3, 0.5],
          [2, 0.5], [4, 1.0], [3, 1.0],
          [4, 0.5], [2, 1.0], [4, 0.5]]

eng = Bayesian_Opt_Engine(chords=chords, melody=melody)


# Initialize x as a 10-dimensional tensor with requires_grad=True
x = torch.randn(10, requires_grad=True)

# Define a wrapper for the likelihood function
def likelihood_wrapper(x):
    # Ensure x is a PyTorch tensor and convert to NumPy
    x_np = x.detach().numpy()
    # Compute the likelihood using the Engine
    likelihood = eng.get_likelihood(x_np)
    # Convert the result back to a PyTorch tensor
    return torch.tensor(likelihood, dtype=torch.float32)

# Define an optimizer (e.g., Adam)
optimizer = torch.optim.Adam([x], lr=0.01)

# Optimization loop
for step in range(1000):  # Number of optimization steps
    optimizer.zero_grad()  # Clear previous gradients
    
    # Compute the negative likelihood (objective to minimize)
    loss = likelihood_wrapper(x)
    
    # Backpropagate to compute gradients
    loss.backward()
    
    # Update x using the optimizer
    optimizer.step()
    
    # Print progress every 100 steps
    if step % 100 == 0:
        print(f"Step {step}: Loss = {loss.item()}")

# Final optimized parameters
print("Optimized x:", x.detach().numpy())