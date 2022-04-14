#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
List of log-likelihoods for the types of variables considered in this paper.
Basically, we create the different layers needed in the decoder and during the
generation of new samples

The variable reuse indicates the mode of this functions
- reuse = None -> Decoder implementation
- reuse = True -> Samples generator implementation

"""

import sys

sys.path.append('../')

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F

from HL_VAE.utils import one_hot, sequence_mask


def loglik_real(batch_data, list_type, theta, normalization_params, extra_params=None):
    output = dict()
    epsilon = 3e-4
    min_log_vy = torch.Tensor([-8.0]).to(theta.device)

    # Data outputs
    data, missing_mask = batch_data
    missing_mask = missing_mask.float()

    if normalization_params != []:
        data_mean, data_var = normalization_params
        data_var = torch.clamp(data_var, epsilon, np.inf)
    else:
        data_mean = torch.tensor(0.)
        data_var = torch.tensor(1.)

    indx = np.r_[0:data.shape[1]]

    if extra_params == None:
        est_mean, est_var = theta[:, indx], theta[:, indx + len(indx)]
        est_log_vy = min_log_vy + F.softplus(est_var - min_log_vy)
        est_var = torch.exp(est_log_vy)
    else:
        est_mean = theta[:, indx]
        est_log_vy = min_log_vy + F.softplus(extra_params - min_log_vy)
        est_var = torch.exp(est_log_vy)

    # Affine transformation of the parameters
    est_mean = torch.sqrt(data_var) * est_mean + data_mean
    est_var = data_var * est_var
    # Compute loglik
    log_p_x = -0.5 * torch.pow(data - est_mean, 2) / est_var - 0.5 * np.log(2 * np.pi) - 0.5 * torch.log(est_var)
    normal = td.Normal(est_mean, torch.sqrt(est_var))

    # Outputs
    output['log_p_x'] = torch.mul(log_p_x, missing_mask)
    output['log_p_x_missing'] = torch.mul(log_p_x, 1.0 - missing_mask)
    if est_var.shape == est_mean.shape:
        output['params'] = [est_mean, est_var]
    else:
        output['params'] = est_mean
    output['samples'] = normal.rsample()

    return output


def loglik_pos(batch_data, list_type, theta, normalization_params, extra_params=None):
    # Log-normal distribution
    output = dict()
    epsilon = 1e-3

    # Data outputs
    log_data_mean, log_data_var = normalization_params
    log_data_var = torch.clamp(log_data_var, epsilon, np.inf)

    # data, missing_mask = batch_data
    data, missing_mask = batch_data
    log_data = torch.log(1.0 + data)
    missing_mask = missing_mask.float()

    indx = np.r_[0:data.shape[1]]
    try:
        est_mean, est_log_var = theta[:, indx], theta[:, indx + len(indx)]
    except:
        est_mean = theta[:, indx]

    # est_var = torch.clamp(torch.nn.Softplus()(est_var), epsilon, 1.0)

    # Affine transformation of the parameters
    est_mean = torch.sqrt(log_data_var) * est_mean + log_data_mean

    ##TODO:free logvar for pos
    try:
        est_var = log_data_var * torch.exp(extra_params)
        # Compute loglik
        log_p_x = -0.5 * torch.pow(log_data - est_mean, 2) / est_var - 0.5 * torch.log(2 * np.pi * est_var) - log_data
        normal = td.Normal(est_mean, torch.sqrt(est_var))
    except:
        est_var = log_data_var * torch.exp(est_log_var)
        # Compute loglik
        log_p_x = -0.5 * torch.pow(log_data - est_mean, 2) / est_var - 0.5 * torch.log(2 * np.pi * est_var) - log_data
        normal = td.Normal(est_mean, torch.sqrt(est_var))

    # Compute loglik
    # log_p_x = -0.5 * torch.pow(data_log - est_mean, 2) / est_var - 0.5 * torch.log(2 * np.pi * est_var) - data_log
    output['log_p_x'] = torch.mul(log_p_x, missing_mask)
    output['log_p_x_missing'] = torch.mul(log_p_x, 1.0 - missing_mask)
    if est_var.shape == est_mean.shape:
        output['params'] = [est_mean, est_var]
    else:
        output['params'] = est_mean
    output['samples'] = torch.clamp(
        torch.exp(normal.rsample()) - 1.0, 0, 1e20)

    return output


def loglik_cat(batch_data, list_type, theta, normalization_params, extra_params=None):
    output = dict()

    # Data outputs
    data, missing_mask = batch_data
    missing_mask = missing_mask.float()

    log_pi = theta.reshape((theta.shape[0], missing_mask.shape[1], -1))

    # Compute loglik
    log_pi = torch.sub(log_pi, torch.logsumexp(log_pi, 2).reshape(data.shape[0], -1, 1))
    log_p_x = torch.sum(data.reshape((log_pi.shape[0], missing_mask.shape[1], -1)) * F.log_softmax(log_pi, 2), -1)

    output['log_p_x'] = torch.mul(log_p_x.reshape(log_p_x.shape[0], -1), missing_mask)
    output['log_p_x_missing'] = torch.mul(log_p_x.reshape(log_p_x.shape[0], -1), 1.0 - missing_mask)
    output['params'] = log_pi
    try:
        output['samples'] = one_hot(td.Categorical(probs=nn.Softmax(1)(log_pi)).sample(),
                                depth=int(list_type[1])).to(torch.float64)
    except:
        print('mine')

    return output


def loglik_ordinal(batch_data, list_type, theta, normalization_params, extra_params=None):
    output = dict()
    epsilon = 1e-6

    # Data outputs
    data, missing_mask = batch_data
    missing_mask = missing_mask.float()
    batch_size = data.size()[0]
    cov_dim = missing_mask.shape[1]
    data = data.reshape(batch_size, cov_dim, -1)

    # We need to force that the outputs of the network increase with the categories
    theta = theta.reshape(batch_size, cov_dim, -1)
    partition_param, mean_param = theta[:, :, :-1], theta[:, :, -1]
    mean_value = nn.Softplus()(mean_param[:, :, None])
    theta_values = torch.cumsum(torch.clamp(nn.Softplus()(partition_param), epsilon, 1e20), 2)
    sigmoid_est_mean = nn.Sigmoid()(theta_values - mean_value)
    mean_probs = torch.cat([sigmoid_est_mean, torch.ones([batch_size, cov_dim, 1], dtype=torch.float64).to(theta.device)], 2) - torch.cat(
        [torch.zeros([batch_size, cov_dim, 1], dtype=torch.float64).to(theta.device), sigmoid_est_mean], 2)

    mean_probs = torch.clamp(mean_probs, epsilon, 1.0)

    # Code needed to compute samples from an ordinal distribution
    vals = torch.sum(data.detach().int(), 2)
    vals[missing_mask==0] = 1
    true_values = one_hot(vals - 1, int(list_type[1]))

    # Compute loglik

    mean_probs = torch.div(mean_probs, torch.sum(mean_probs, 2).reshape(batch_size, cov_dim, 1))
    log_p_x = torch.sum(true_values * F.log_softmax(torch.log(mean_probs), -1), -1)

    output['log_p_x'] = torch.mul(log_p_x, missing_mask)
    output['log_p_x_missing'] = torch.mul(log_p_x, 1.0 - missing_mask)
    output['params'] = mean_probs
    output['samples'] = sequence_mask(1 + td.Categorical(logits=torch.log(torch.clamp(mean_probs, epsilon, 1e20)))
                                      .sample(),
                                      int(list_type[1]), dtype=torch.float64)

    return output


def loglik_count(batch_data, list_type, theta, normalization_params, extra_params=None):
    output = dict()
    epsilon = 1e-6

    # Data outputs
    data, missing_mask = batch_data
    data = data.reshape(data.shape[0], -1)
    # data = torch.cat(data, 1)
    missing_mask = missing_mask.float()

    # est_lambda = theta[:, :, 0]
    est_lambda = theta.reshape(theta.shape[0], -1)
    est_lambda = torch.clamp(torch.nn.Softplus()(est_lambda), epsilon, 1e20)

    poisson = td.Poisson(est_lambda)
    log_p_x = poisson.log_prob(data)  # .sum(1)

    output['log_p_x'] = torch.mul(log_p_x, missing_mask)
    output['log_p_x_missing'] = torch.mul(log_p_x, 1.0 - missing_mask)
    output['params'] = est_lambda
    output['samples'] = poisson.sample()

    return output


def loglik_beta(batch_data, list_type, theta, normalization_params, extra_params=None):
    output = dict()
    epsilon = 1e-6

    # Data outputs
    data, missing_mask = batch_data
    normalization_params = normalization_params.reshape(data.shape[1], -1)
    data_min = torch.tensor(normalization_params[:,0]).to(data.device)
    data_max = torch.tensor(normalization_params[:,1]).to(data.device)
    data_converted = (data - data_min) / (data_max - data_min) + epsilon
    data_converted = data_converted.reshape(data_converted.shape[0], -1)
    missing_mask = missing_mask.float()

    indx = np.r_[0:data.shape[1]]
    try:
        est_alpha, est_beta = theta[:, indx], theta[:, indx + len(indx)]
    except:
        est_alpha, est_beta = torch.unsqueeze(theta[:, 0], 1), torch.unsqueeze(theta[:, 1], 1)


    # est_alpha = torch.clamp(nn.Softplus()(est_alpha), epsilon, 1e20)   # Must be positive
    # est_beta = torch.clamp(nn.Softplus()(est_beta), epsilon, 1e20) # Must be positive

    est_mean = est_alpha
    extra_params = torch.clamp(torch.nn.Softplus()(extra_params[0]), epsilon, 1e20)
    normal = td.Normal(0, 1)
    est_mean = normal.cdf(est_mean)

    est_alpha = torch.mul(extra_params, est_mean)
    est_beta = torch.mul(extra_params, (1 - est_mean))

    log_p_x = (est_alpha-1) * torch.log(data_converted) + (est_beta-1) * torch.log(1 - data_converted) \
              - torch.lgamma(est_alpha) - torch.lgamma(est_beta) + torch.lgamma(est_alpha + est_beta)

    beta = td.Beta(est_alpha, est_beta)
    output['log_p_x'] = torch.mul(log_p_x, missing_mask)
    output['log_p_x_missing'] = torch.mul(log_p_x, 1.0 - missing_mask)
    output['params'] = [est_alpha, est_beta]
    output['samples'] = beta.sample() * (data_max - data_min) + data_min

    return output
