import torch
import gpytorch
import torch.nn as nn

from sample_normal import sample_multivariate_normal

def var_l2_loss_estimator(model, test_x, Predictor, device, N_iter):

    latent_posterior = model(test_x)
    latent_posterior_sample = latent_posterior.rsample(sample_shape=torch.Size([N_iter]))
    prediction = Predictor(test_x).squeeze()
    L_2_loss_each_point = torch.square(torch.subtract(latent_posterior_sample, prediction))
    L_2_loss_each_f = torch.mean(L_2_loss_each_point, dim=1)
    L_2_loss_variance = torch.var(L_2_loss_each_f)
    print("L_2_loss_variance:",L_2_loss_variance)

    L_2_loss_mean = torch.mean(L_2_loss_each_f)+model.likelihood.noise
    print("L_2_loss_mean:", L_2_loss_mean)

    return L_2_loss_variance

def l2_loss(test_x, test_y, Predictor, device):
    prediction = Predictor(test_x).squeeze()
    diff_square = torch.square(torch.subtract(test_y, prediction))
    return torch.mean(diff_square)

def var_l2_loss_custom_gp_estimator(mu, cov, noise, test_x, Predictor, device, N_iter):

    latent_posterior_sample = sample_multivariate_normal(mu, cov, N_iter)
    prediction = Predictor(test_x).squeeze()
    L_2_loss_each_point = torch.square(torch.subtract(latent_posterior_sample, prediction))
    L_2_loss_each_f = torch.mean(L_2_loss_each_point, dim=1)
    L_2_loss_variance = torch.var(L_2_loss_each_f)
    print("L_2_loss_variance:",L_2_loss_variance)

    L_2_loss_mean = torch.mean(L_2_loss_each_f)+noise
    print("L_2_loss_mean:", L_2_loss_mean)

    return L_2_loss_variance