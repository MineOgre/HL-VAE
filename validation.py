import os

from torch.utils.data import DataLoader

import torch
import pandas as pd
import numpy as np

from HL_VAE import read_functions
from elbo_functions import deviance_upper_bound, elbo
from utils import batch_predict_varying_T, HensmanDataLoader
import HL_VAE.read_functions  as rd

num_workers = 4

def validation_dubo(latent_dim, covar_module0, covar_module1, likelihood, train_xt, m, log_v, z, P, T, eps):
    """
    Efficient KL divergence using the variational mean and variance instead of a sample from the latent space (DUBO).
    See L-VAE supplementary material.

    :param covar_module0: additive kernel (sum of cross-covariances) without id covariate
    :param covar_module1: additive kernel (sum of cross-covariances) with id covariate
    :param likelihood: GPyTorch likelihood model
    :param train_xt: auxiliary covariate information
    :param m: variational mean
    :param log_v: (log) variational variance
    :param z: inducing points
    :param P: number of unique instances
    :param T: number of longitudinal samples per individual
    :param eps: jitter
    :return: KL divergence between variational distribution and additive GP prior (DUBO)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v = torch.exp(log_v)
    torch_dtype = torch.double
    x_st = torch.reshape(train_xt, [P, T, train_xt.shape[1]]).to(device)
    stacked_x_st = torch.stack([x_st for i in range(latent_dim)], dim=1)
    K0xz = covar_module0(train_xt, z).evaluate().to(device)
    K0zz = (covar_module0(z, z).evaluate() + eps * torch.eye(z.shape[1], dtype=torch_dtype).to(device)).to(device)
    LK0zz = torch.cholesky(K0zz).to(device)
    iK0zz = torch.cholesky_solve(torch.eye(z.shape[1], dtype=torch_dtype).to(device), LK0zz).to(device)
    K0_st = covar_module0(stacked_x_st, stacked_x_st).evaluate().transpose(0,1)
    B_st = (covar_module1(stacked_x_st, stacked_x_st).evaluate() + torch.eye(T, dtype=torch.double).to(device) * likelihood.noise_covar.noise.unsqueeze(dim=2)).transpose(0,1)
    LB_st = torch.cholesky(B_st).to(device)
    iB_st = torch.cholesky_solve(torch.eye(T, dtype=torch_dtype).to(device), LB_st)

    dubo_sum = torch.tensor([0.0]).double().to(device)
    for i in range(latent_dim):
        m_st = torch.reshape(m[:, i], [P, T, 1]).to(device)
        v_st = torch.reshape(v[:, i], [P, T]).to(device)
        K0xz_st = torch.reshape(K0xz[i], [P, T, K0xz.shape[2]]).to(device)
        iB_K0xz = torch.matmul(iB_st[i], K0xz_st).to(device)
        K0zx_iB_K0xz = torch.matmul(torch.transpose(K0xz[i], 0, 1), torch.reshape(iB_K0xz, [P*T, K0xz.shape[2]])).to(device)
        W = K0zz[i] + K0zx_iB_K0xz
        W = (W + W.T) / 2
        LW = torch.cholesky(W).to(device)
        logDetK0zz = 2 * torch.sum(torch.log(torch.diagonal(LK0zz[i]))).to(device)
        logDetB = 2 * torch.sum(torch.log(torch.diagonal(LB_st[i], dim1=-2, dim2=-1))).to(device)
        logDetW = 2 * torch.sum(torch.log(torch.diagonal(LW))).to(device)
        logDetSigma = -logDetK0zz + logDetB + logDetW
        iB_m_st = torch.solve(m_st, B_st[i])[0].to(device)
        qF1 = torch.sum(m_st*iB_m_st).to(device)
        p = torch.matmul(K0xz[i].T, torch.reshape(iB_m_st, [P * T])).to(device)
        qF2 = torch.sum(torch.triangular_solve(p[:,None], LW, upper=False)[0] ** 2).to(device)
        qF = qF1 - qF2
        tr = torch.sum(iB_st[i] * K0_st[i]) - torch.sum(K0zx_iB_K0xz * iK0zz[i])
        logDetD = torch.sum(torch.log(v[:, i])).to(device)
        tr_iB_D = torch.sum(torch.diagonal(iB_st[i], dim1=-2, dim2=-1)*v_st).to(device)
        D05_iB_K0xz = torch.reshape(iB_K0xz*torch.sqrt(v_st)[:,:,None], [P*T, K0xz.shape[2]])
        K0zx_iB_D_iB_K0zx = torch.matmul(torch.transpose(D05_iB_K0xz,0,1), D05_iB_K0xz).to(device)
        tr_iB_K0xz_iW_K0zx_iB_D = torch.sum(torch.diagonal(torch.cholesky_solve(K0zx_iB_D_iB_K0zx, LW))).to(device)
        tr_iSigma_D = tr_iB_D - tr_iB_K0xz_iW_K0zx_iB_D
        dubo = 0.5*(tr_iSigma_D + qF - P*T + logDetSigma - logDetD + tr)
        dubo_sum = dubo_sum + dubo
    return dubo_sum

def validate(nnet_model, dataset, type_KL, num_samples, latent_dim, covar_module0, covar_module1, likelihoods,
             zt_list, T, train_mu, train_x, id_covariate, results_path, eps=1e-6):

    print("Testing the model with a validation set")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation_results = []

    batch_size =len(dataset)
    assert (type_KL == 'GPapprox_closed' or type_KL == 'GPapprox')

    # set up Data Loader for training
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    Q = len(dataset[0]['label'])
    P = pd.unique(dataset.label_source.iloc[:, id_covariate]).size

    full_mu = torch.zeros(len(dataset), latent_dim, dtype=torch.double, requires_grad=True).to(device)
    full_log_var = torch.zeros(len(dataset), latent_dim, dtype=torch.double, requires_grad=True).to(device)
    full_labels = torch.zeros(len(dataset), Q, dtype=torch.double, requires_grad=False).to(device)

    recon_loss_sum = 0
    miss_recon_loss_sum = 0
    recon_loss_sum_mse = 0
    nll_loss_sum = 0
    for batch_idx, sample_batched in enumerate(dataloader):
        indices = sample_batched['idx']
        data = sample_batched['digit'].double().to(device)
        mask = sample_batched['mask'].double().to(device)
        full_labels[indices] = sample_batched['label'].double().to(device)

        param_mask = sample_batched['param_mask'].view(batch_size, -1)
        param_mask = param_mask.to(device)
        data = torch.squeeze(data)
        mask = torch.squeeze(mask)
        p_samples, mu, log_var, log_p_x, log_p_x_missing, p_params, q_samples, q_params = nnet_model(data, mask, param_mask, nnet_model.types_info)
        nll = nnet_model.loss_function(log_p_x)
        p_params_complete = read_functions.p_params_concatenation_by_key([p_params], nnet_model.types_info,
                                                                  data.shape[0], data.device, 'x')

        data_transformed = rd.discrete_variables_transformation(data, nnet_model.types_info)
        recon_x_transformed, _ = read_functions.statistics(p_params_complete,
                                                           nnet_model.types_info, data.device, nnet_model.conv,
                                                           [nnet_model._log_vy_real,
                                                            nnet_model._log_vy_pos])
        recon_loss_het, miss_recon_loss_het, _  = read_functions.error_computation(data_transformed,
                                                                       recon_x_transformed,
                                                                       nnet_model.types_info,
                                                                       mask, dim=0)
        recon_loss = recon_loss_het

        full_mu[indices] = mu
        full_log_var[indices] = log_var

        recon_loss = torch.sum(recon_loss)
        recon_loss_sum = recon_loss_sum + recon_loss.item()
        nll = torch.sum(nll)
        nll_loss_sum = nll_loss_sum + nll.item()

    gp_loss_sum = 0

    if isinstance(covar_module0, list):
        if type_KL == 'GPapprox':
            for sample in range(0, num_samples):
                Z = nnet_model.sample_latent(full_mu, full_log_var)
                for i in range(0, latent_dim):
                    Z_dim = Z[:, i]
                    gp_loss = -elbo(covar_module0[i], covar_module1[i], likelihoods[i], full_labels, Z_dim,
                                    zt_list[i].to(device), P, T, eps)
                    gp_loss_sum = gp_loss.item() + gp_loss_sum
            gp_loss_sum /= num_samples

        elif type_KL == 'GPapprox_closed':
            for i in range(0, latent_dim):
                mu_sliced = full_mu[:, i]
                log_var_sliced = full_log_var[:, i]
                gp_loss = deviance_upper_bound(covar_module0[i], covar_module1[i],
                                               likelihoods[i], full_labels,
                                               mu_sliced, log_var_sliced,
                                               zt_list[i].to(device), P,
                                               T, eps)
                gp_loss_sum = gp_loss.item() + gp_loss_sum
    else:
        if type_KL == 'GPapprox_closed':
            df_lbl = pd.DataFrame(np.array(full_labels.cpu()))
            for sz in df_lbl.groupby(0).size().unique():
                ids = (df_lbl.groupby(0).size() == sz)[(df_lbl.groupby(0).size() == sz)].index
                cur_df = df_lbl.loc[df_lbl[0].isin(ids)]
                par_labels = full_labels[cur_df.index]
                par_mu = full_mu[cur_df.index]
                par_log_var = full_log_var[cur_df.index]
                par_P = len(ids)
                gp_loss = validation_dubo(latent_dim, covar_module0, covar_module1,
                                          likelihoods, par_labels,
                                          par_mu, par_log_var,
                                          zt_list, par_P, sz, eps)
                gp_loss_sum += gp_loss.item()

    net_loss_sum = gp_loss_sum + nll_loss_sum

    #Do logging
    print('Validation set - Loss: %.3f  - GP loss: %.3f  - NLL loss: %.3f  - Recon Loss: %.3f' % (
        net_loss_sum, gp_loss_sum, nll_loss_sum, recon_loss_sum))

    if nnet_model.conv:
        halfway = P//2
        subjects = halfway
        l1 = [i*T + k for i in range(0,subjects) for k in range(0,5)]
        l2 = [i*T + k for i in range(halfway,halfway+subjects) for k in range(0,5)]
        prediction_mu = torch.cat((train_mu,
                                full_mu[l1],
                                full_mu[l2]))
        prediction_x = torch.cat((train_x,
                                full_labels[l1],
                                full_labels[l2]))
        test_x = torch.cat((full_labels[0:subjects*T], full_labels[halfway*T:(halfway+subjects)*T]))

        prediction_dataloader_p1 = DataLoader(dataset[0:subjects*T], batch_size=subjects*T, shuffle=False)
        for batch_idx, sample_batched in enumerate(prediction_dataloader_p1):
            data_p1 = sample_batched['digit'].double().to(device)
            mask_p1 = sample_batched['mask'].double().to(device)
            param_mask1 = sample_batched['param_mask'].view(sample_batched['param_mask'].shape[0], -1)
        prediction_dataloader_p2 = DataLoader(dataset[halfway*T:(halfway+subjects)*T], batch_size=subjects*T, shuffle=False)
        for batch_idx, sample_batched in enumerate(prediction_dataloader_p2):
            data_p2 = sample_batched['digit'].double().to(device)
            mask_p2 = sample_batched['mask'].double().to(device)
            param_mask2 = sample_batched['param_mask'].view(sample_batched['param_mask'].shape[0], -1)
        prediction_data = torch.cat((data_p1, data_p2))
        prediction_mask = torch.cat((mask_p1, mask_p2))
        prediction_param_mask = torch.cat((param_mask1, param_mask2))
    else:
        indexes_for_prediction = []
        for no in df_lbl[0].unique():
            indexes_for_prediction = indexes_for_prediction + list(df_lbl[df_lbl[0] == no].index[:2].astype(int))
        prediction_mu = torch.cat((train_mu,
                                full_mu[indexes_for_prediction]))
        prediction_x = torch.cat((train_x,
                                full_labels[indexes_for_prediction]))
        test_x = full_labels
        prediction_data = sample_batched['digit'].double().to(device)
        prediction_mask = sample_batched['mask'].double().to(device)
        prediction_true_mask = sample_batched['true_mask'].double().to(device)
        prediction_param_mask = sample_batched['param_mask'].view(sample_batched['param_mask'].shape[0], -1)


    Z_pred = batch_predict_varying_T(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x, test_x, prediction_mu, zt_list, id_covariate, eps=1e-6)

    prediction_param_mask = prediction_param_mask.to(device).to(torch.float64)
    prediction_data = torch.squeeze(prediction_data).to(torch.float64)
    prediction_mask = torch.squeeze(prediction_mask).to(torch.float64)
    log_p_x, _, recon_batch, params_x = nnet_model.decode(Z_pred, prediction_data, prediction_mask, prediction_param_mask)

    p_params_complete = read_functions.p_params_concatenation_by_key([params_x], nnet_model.types_info,
                                                              prediction_data.shape[0], data.device, 'x')
    data_transformed = rd.discrete_variables_transformation(prediction_data, nnet_model.types_info)
    recon_x_transformed, _ = read_functions.statistics(p_params_complete,
                                                       nnet_model.types_info, prediction_data.device, nnet_model.conv,
                                                       [nnet_model._log_vy_real,
                                                        nnet_model._log_vy_pos])
    recon_loss_GP, miss_recon_loss_GP, _  = read_functions.error_computation(data_transformed,
                                                                   recon_x_transformed,
                                                                   nnet_model.types_info,
                                                                   prediction_mask)


    print('Validation set - GP prediction Error: %.4f, Mean of Error %.4f' % (torch.sum(recon_loss_GP).item(),
                                                        (torch.sum(recon_loss_GP).item())/recon_loss_GP.shape[0]))

    validation_results.append(recon_loss_sum/len(dataset))
    validation_results.append(torch.sum(recon_loss_GP).item()/recon_loss_GP.shape[0])
    validation_results.append(recon_loss_sum_mse/len(dataset))
    validation_results.append(miss_recon_loss_sum/len(dataset))
    validation_results.append(torch.sum(miss_recon_loss_GP).item()/miss_recon_loss_GP.shape[0])
    validation_results.append(net_loss_sum)
    validation_results.append(gp_loss_sum)
    validation_results.append(nll_loss_sum)
    validation_results.append(recon_loss_sum)
    validation_results.append(torch.sum(recon_loss_GP).item())
    df_res = pd.DataFrame(validation_results, index=['vae_error','GP_error', 'vae_mse',#'GP_mse',
                                                     'miss_vae_error','miss_GP_error', 'net_loss', 'GP_loss', 'nll_loss',
                                                     'recon_loss_sum', 'GP_recon_loss_sum'])
    df_res.to_csv(os.path.join(results_path, 'validation_results.csv'),header=False)
    return df_res

