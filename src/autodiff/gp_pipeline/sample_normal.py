import torch

def sample_multivariate_normal(mu, cov, n_samples):
    """
    Sample from a multivariate normal distribution using the reparameterization trick.

    Parameters:
    - mu (torch.Tensor): The mean vector of the distribution.    1-D dimension [D]
    - cov (torch.Tensor): The covariance matrix of the distribution.  2-D dimension [D,D]
    - n_samples (int): The number of samples to generate.

    Returns:
    - torch.Tensor: Samples from the multivariate normal distribution.
    """
    # Ensure mu and cov are tensors
    #mu = torch.tensor(mu, dtype=torch.float32)
    #cov = torch.tensor(cov, dtype=torch.float32)

    # Cholesky decomposition of the covariance matrix
    L = torch.linalg.cholesky(cov + 1e-5 * torch.eye(cov.size(0), device=cov.device))

    #L = torch.linalg.cholesky(cov + 1e-8 * torch.eye(cov.size(0)))

    # Sample Z from a standard normal distribution
    Z = torch.randn(n_samples, mu.size(0)).to(device=cov.device)           # Z: [n_samples, D]

    # Transform Z to obtain samples from the target distribution
    samples = mu + Z @ L.T

    return samples    #[n_samples, D]