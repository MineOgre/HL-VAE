import pickle
import time

from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

import numpy as np
import torch
import os
import pandas as pd

from GP_def import ExactGPModel
from HL_VAE import read_functions
from utils import plot_training_info
from elbo_functions import minibatch_KLD_upper_bound, minibatch_KLD_upper_bound_iter
from model_test import HLVAETest
from utils import SubjectSampler, VaryingLengthSubjectSampler, VaryingLengthBatchSampler, HensmanDataLoader
from predict_HealthMNIST import recon_complete_gen
from validation import validate

num_workers = 0

def hensman_training(nnet_model, epochs, dataset, optimiser, type_KL, num_samples, latent_dim, covar_module0,
                     covar_module1, likelihoods, m, H, zt_list, P, T, varying_T, Q, id_covariate, save_path,
                     natural_gradient=False, natural_gradient_lr=0.01, subjects_per_batch=20,
                     eps=1e-6, results_path=None, validation_dataset=None, generation_dataset=None,
                     prediction_dataset=None, save_interval=100):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = len(dataset)
    assert type_KL == 'GPapprox_closed'

    best_value = np.inf
    best_epoch = 0
    validation_interval = 5

    P = pd.unique(dataset.label_source.iloc[:, id_covariate]).size

    if varying_T:
        n_batches = (P + subjects_per_batch - 1) // subjects_per_batch
        dataloader = HensmanDataLoader(dataset, batch_sampler=VaryingLengthBatchSampler(
            VaryingLengthSubjectSampler(dataset, id_covariate), subjects_per_batch), num_workers=num_workers)
    else:
        batch_size = subjects_per_batch * T
        n_batches = (P * T + batch_size - 1) // (batch_size)
        dataloader = HensmanDataLoader(dataset, batch_sampler=BatchSampler(SubjectSampler(dataset, P, T), batch_size,
                                                                           drop_last=False), num_workers=num_workers)

    net_train_loss_arr = np.empty((0, 1))
    recon_loss_arr = np.empty((0, 1))
    nll_loss_arr = np.empty((0, 1))
    kld_loss_arr = np.empty((0, 1))
    penalty_term_arr = np.empty((0, 1))

    validation_net_loss = np.empty((0, 1))
    validation_recon_loss = np.empty((0, 1))
    validation_GP_loss = np.empty((0, 1))
    validation_VAE_error = np.empty((0, 1))
    validation_GP_error = np.empty((0, 1))
    best_epoch_missing_imp_error = -1

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        recon_loss_sum = 0
        nll_loss_sum = 0
        kld_loss_sum = 0
        net_loss_sum = 0
        miss_recon_loss_sum = 0
        recon_loss_sum_2 = 0
        for batch_idx, sample_batched in enumerate(dataloader):
            optimiser.zero_grad()

            data = sample_batched['digit'].to(device).to(torch.float64)
            train_x = sample_batched['label'].to(device).to(torch.float64)
            mask = sample_batched['mask'].to(device).to(torch.float64)
            N_batch = data.shape[0]

            param_mask = sample_batched['param_mask'].view(data.shape[0], -1)
            param_mask = param_mask.to(device).to(torch.float64)
            data = torch.squeeze(data)
            mask = torch.squeeze(mask)
            p_samples, mu, log_var, log_p_x, log_p_x_missing, p_params, q_samples, q_params = nnet_model(data, mask, param_mask, dataset.types_info)
            nll = nnet_model.loss_function(log_p_x)
            p_params_complete = read_functions.p_params_concatenation_by_key([p_params], nnet_model.types_info,
                                                                      data.shape[0], data.device, 'x')
            data_transformed = read_functions.discrete_variables_transformation(data, nnet_model.types_info)
            recon_x_transformed, _ = read_functions.statistics(p_params_complete,
                                                               nnet_model.types_info, data.device,
                                                               nnet_model.conv,
                                                               [nnet_model._log_vy_real,
                                                                nnet_model._log_vy_pos])
            recon_loss, miss_recon_loss, partial_error = read_functions.error_computation(data_transformed,
                                                                                 recon_x_transformed,
                                                                                 nnet_model.types_info,
                                                                                 mask, dim=0)

            try:
                for key in partial_error.keys():
                    recon_loss = torch.sum(partial_error[key]['error_all']*data.shape[0])
            except:
                recon_loss = torch.sum(recon_loss)
            recon_loss_2 = recon_loss.item()
            # miss_recon_loss_sum += torch.sum(miss_recon_loss)
            nll_loss = torch.sum(nll)

            PSD_H = H if natural_gradient else torch.matmul(H, H.transpose(-1, -2))

            if varying_T:
                P_in_current_batch = torch.unique(train_x[:, id_covariate]).shape[0]
                kld_loss, grad_m, grad_H = minibatch_KLD_upper_bound_iter(covar_module0, covar_module1, likelihoods,
                                                                          latent_dim, m, PSD_H, train_x, mu, log_var,
                                                                          zt_list, P, P_in_current_batch, N,
                                                                          natural_gradient, id_covariate, eps)
            else:
                P_in_current_batch = N_batch // T
                kld_loss, grad_m, grad_H = minibatch_KLD_upper_bound(covar_module0, covar_module1, likelihoods,
                                                                     latent_dim, m, PSD_H, train_x, mu, log_var,
                                                                     zt_list, P, P_in_current_batch, T,
                                                                     natural_gradient, eps)

            recon_loss = recon_loss * P / P_in_current_batch
            nll_loss = nll_loss * P / P_in_current_batch

            net_loss = nll_loss + kld_loss


            net_loss.backward()
            optimiser.step()

            if natural_gradient:
                LH = torch.cholesky(H)
                iH = torch.cholesky_solve(torch.eye(H.shape[-1], dtype=torch.double).to(device), LH)
                iH_new = iH + natural_gradient_lr * (grad_H + grad_H.transpose(-1, -2))
                LiH_new = torch.cholesky(iH_new)
                H = torch.cholesky_solve(torch.eye(H.shape[-1], dtype=torch.double).to(device), LiH_new).detach()
                m = torch.matmul(H, torch.matmul(iH, m) - natural_gradient_lr * (
                            grad_m - 2 * torch.matmul(grad_H, m))).detach()

            net_loss_sum += net_loss.item() / n_batches
            recon_loss_sum += recon_loss.item() / n_batches
            recon_loss_sum_2 += recon_loss_2
            nll_loss_sum += nll_loss.item() / n_batches
            kld_loss_sum += kld_loss.item() / n_batches

        print('Iter %d/%d - Time: %.3f  - Loss: %.3f  - GP loss: %.3f  - NLL Loss: %.3f  - Recon Loss: %.3f' % (
            epoch, epochs, time.time()-start_time, net_loss_sum, kld_loss_sum, nll_loss_sum, recon_loss_sum_2), flush=True)
        penalty_term_arr = np.append(penalty_term_arr, 0.0)
        net_train_loss_arr = np.append(net_train_loss_arr, net_loss_sum)
        recon_loss_arr = np.append(recon_loss_arr, recon_loss_sum)
        nll_loss_arr = np.append(nll_loss_arr, nll_loss_sum)
        kld_loss_arr = np.append(kld_loss_arr, kld_loss_sum)

        miss_recon_loss = miss_recon_loss_sum/N

        # print(f'Missing imputation Error for Training: {miss_recon_loss}')
        print(f'Error for Training: {recon_loss_sum_2/(N*mask.shape[1])}')

        if (not epoch % validation_interval or not epoch % save_interval): #
            validation_start_time = time.time()
            with torch.no_grad():
                if validation_dataset is not None:
                    full_mu = torch.zeros(len(dataset), latent_dim, dtype=torch.double).to(device)
                    prediction_x = torch.zeros(len(dataset), Q, dtype=torch.double).to(device)
                    for batch_idx, sample_batched in enumerate(dataloader):
                        label_id = sample_batched['idx']
                        prediction_x[label_id] = sample_batched['label'].double().to(device)
                        data = sample_batched['digit'].to(device).to(torch.float64)
                        mask = sample_batched['mask'].to(device).to(torch.float64)
                        covariates = torch.cat(
                            (prediction_x[label_id, :id_covariate], prediction_x[label_id, id_covariate + 1:]), dim=1)

                        param_mask = sample_batched['param_mask'].view(data.shape[0], -1)
                        param_mask = param_mask.to(device)
                        data = torch.squeeze(data)
                        mask = torch.squeeze(mask)
                        samples, q_params = nnet_model.encode(data, mask, param_mask, dataset.types_info)
                        mu, log_var = q_params['z'][0], q_params['z'][1]

                        full_mu[label_id] = mu
                    validation_resuts_df = validate(nnet_model, validation_dataset, type_KL, num_samples, latent_dim, covar_module0,
                            covar_module1, likelihoods, zt_list, T, full_mu, prediction_x, id_covariate,
                            results_path, eps=1e-6)
                    validation_resuts_df.loc['best_epoch'] = best_epoch
                    validation_resuts_df.loc['best_epoch_missing_imp_error'] = best_epoch_missing_imp_error
                    validation_resuts_df.loc['missing_imp_error'] = miss_recon_loss
                    validation_net_loss = np.append(validation_net_loss, validation_resuts_df.loc['net_loss'])
                    validation_recon_loss = np.append(validation_recon_loss, validation_resuts_df.loc['nll_loss'])
                    validation_GP_loss = np.append(validation_GP_loss, validation_resuts_df.loc['GP_loss'])
                    validation_VAE_error = np.append(validation_VAE_error, validation_resuts_df.loc['vae_error'])
                    validation_GP_error = np.append(validation_GP_error, validation_resuts_df.loc['GP_error'])
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            print(f'Validation Duration: {time.time()-validation_start_time}')

        if (not epoch % save_interval): # and epoch != epochs:]
            if epochs > 49:
                plot_training_info(net_train_loss=validation_net_loss, net_train_nll=validation_recon_loss,
                               net_train_KL_Z=validation_GP_loss, net_train_mean_error=validation_VAE_error, tr_imputed_error=None,
                               net_test_mean_error=validation_GP_error, test_imputed_error=None, save_path=save_path)
            if validation_dataset is not None and epochs > 50:
                pd.to_pickle(validation_resuts_df,
                         os.path.join(save_path, 'validation_df.pkl'))
                validation_resuts_df.to_csv(os.path.join(save_path, 'validation_df.csv'))
                with open(os.path.join(save_path, 'validation_values.pkl'), 'wb') as f:
                    pickle.dump([validation_net_loss, validation_recon_loss, validation_GP_loss, validation_VAE_error,validation_GP_error], f)

            ### Training Missing Prediction Metrics
            _, _, _, tr_pred_error, _, tr_mode_error, tr_imputed_error, partial_metrics_training = HLVAETest(dataset, nnet_model,
                                                                                                             False, False, id_covariate=id_covariate, T=T)

            ################################

            with open(
                    f'{results_path}/partial_metrics_training_VAE.pickle',
                    'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(partial_metrics_training, f)

            with torch.no_grad():
                if results_path and generation_dataset and epoch != epochs:
                    prediction_dataloader = DataLoader(prediction_dataset, batch_sampler=VaryingLengthBatchSampler(
                        VaryingLengthSubjectSampler(prediction_dataset, id_covariate), subjects_per_batch),
                                                       num_workers=num_workers)
                    full_mu = torch.zeros(len(prediction_dataset), latent_dim, dtype=torch.double).to(device)
                    prediction_x = torch.zeros(len(prediction_dataset), Q, dtype=torch.double).to(device)
                    for batch_idx, sample_batched in enumerate(prediction_dataloader):
                        label_id = sample_batched['idx']
                        prediction_x[label_id] = sample_batched['label'].double().to(device)
                        data = sample_batched['digit'].double().to(device)
                        mask = sample_batched['mask'].double().to(device)
                        covariates = torch.cat(
                            (prediction_x[label_id, :id_covariate], prediction_x[label_id, id_covariate + 1:]), dim=1)

                        param_mask = sample_batched['param_mask'].view(data.shape[0], -1)
                        param_mask = param_mask.to(device)
                        data = torch.squeeze(data)
                        mask = torch.squeeze(mask)
                        samples, q_params = nnet_model.encode(data, mask, param_mask, dataset.types_info)
                        mu, log_var = q_params['z'][0], q_params['z'][1]

                        full_mu[label_id] = mu
                    recon_complete_gen(generation_dataset, nnet_model,
                                       results_path, covar_module0,
                                       covar_module1, likelihoods, latent_dim,
                                       './data', prediction_x, full_mu, epoch,
                                       zt_list, P, T, id_covariate, varying_T)
        if (not epoch % validation_interval) and epoch > 100:
            current_value = validation_net_loss[-1]
            if current_value < best_value:
                gp_model = ExactGPModel(train_x, mu.type(torch.DoubleTensor), likelihoods,
                                        covar_module0 + covar_module1).to(device)

                try:
                    torch.save(nnet_model.state_dict(), os.path.join(save_path, 'early_best-vae_model.pth'),
                               _use_new_zipfile_serialization=False)
                    torch.save(gp_model.state_dict(), os.path.join(save_path, f'gp_model_early_best.pth'),
                               _use_new_zipfile_serialization=False)
                    torch.save(zt_list, os.path.join(save_path, f'zt_list_early_best.pth'), _use_new_zipfile_serialization=False)
                    torch.save(m, os.path.join(save_path, f'm_early_best.pth'), _use_new_zipfile_serialization=False)
                    torch.save(H, os.path.join(save_path, f'H_early_best.pth'), _use_new_zipfile_serialization=False)
                    best_epoch = epoch
                    best_epoch_missing_imp_error = miss_recon_loss.item()
                except:
                    pass
                best_value = current_value

    print(f"Best epoch is {best_epoch}")
    print(f"Best epoch imputation error is {best_epoch_missing_imp_error}")
    try:
        print(f"Imputation error is {miss_recon_loss}")
    except:
        pass
    return penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, kld_loss_arr, m, H
