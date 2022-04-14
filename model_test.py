import os

from torch.utils.data import DataLoader
import numpy as np
import torch
import pandas as pd
import pickle

from HL_VAE import read_functions
from dataset_def import HeterogeneousHealthMNISTDataset
from utils import batch_predict, batch_predict_varying_T

num_workers = 0

def predict_gp(kernel_component, full_kernel_inverse, z):
    mean = torch.matmul(torch.matmul(kernel_component, full_kernel_inverse), z)
    return mean

def MSE_test_GPapprox(csv_file_test_data, csv_file_test_label, test_mask_file, data_source_path, nnet_model,
                      covar_module0, covar_module1, likelihoods, results_path, latent_dim, prediction_x, prediction_mu,
                      zt_list, P, T, id_covariate, varying_T=False, csv_types_file=None, true_test_mask_file=None,
                      test_type='final', training_indexes=[]):

    print("Running tests with a test set")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = HeterogeneousHealthMNISTDataset(csv_file_data=csv_file_test_data,
                                                   csv_file_label=csv_file_test_label,
                                                   mask_file=test_mask_file,
                                                   types_file=csv_types_file,
                                                   true_miss_file=true_test_mask_file,
                                                   root_dir=data_source_path,
                                                   transform=None,
                                                   logvar_network=nnet_model.logvar_network)

    print('Length of test dataset:  {}'.format(len(test_dataset)))
    dataloader_test = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=num_workers)
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_test):
            # no mini-batching. Instead get a mini-batch of size 4000

            # label_id = sample_batched['idx']
            label = sample_batched['label'].double()
            full_data = sample_batched['digit'].double()
            full_data = full_data.to(device)
            mask = sample_batched['mask'].double()
            mask = mask.to(device)
            try:
                ##TODO:[New]Bunun bir yere girmesi lazim
                true_mask = sample_batched['true_mask'].double().to(device)
            except:
                pass

            param_mask = sample_batched['param_mask'].view(sample_batched['param_mask'].shape[0], -1)
            param_mask = param_mask.to(device)
            full_data = torch.squeeze(full_data)
            mask = torch.squeeze(mask)


            test_x = label.type(torch.DoubleTensor).to(device)
            Z_pred = batch_predict_varying_T(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x, test_x, prediction_mu, zt_list, id_covariate, eps=1e-6)

            P_test = len(torch.unique(test_x[:, id_covariate]))

            ##This part is for MNIST and PPMI
            if nnet_model.conv:
                indexes = np.concatenate([np.array(range(5, T)) + i * T for i in range(P_test)])
            else:
                x_indexes = np.array(list(set(np.array(test_x[:, -1].cpu(), int)) - set(training_indexes)), int)
                indexes = list(set.intersection(set(np.array(label[:, -1].cpu(), int)), set(x_indexes)))
                indexes = [label[i, -1] in indexes for i in range(label.shape[0])]

            log_p_x, log_p_x_missing, recon_Z, params_Z = nnet_model.decode(Z_pred, full_data, mask, param_mask)
            nll = nnet_model.loss_function(log_p_x)  # reconstruction loss
            p_params_complete = read_functions.p_params_concatenation_by_key([params_Z], nnet_model.types_info,
                                                                      full_data.shape[0], full_data.device, 'x')

            # recon_x = rd.samples_concatenation_x(recon_x, dataset.types_info)
            data_transformed = read_functions.discrete_variables_transformation(full_data, nnet_model.types_info)
            recon_x_transformed_mean, recon_x_transformed_mode = read_functions.statistics(p_params_complete,
                                                               nnet_model.types_info, full_data.device,
                                                               nnet_model.conv,
                                                               [nnet_model._log_vy_real,
                                                                nnet_model._log_vy_pos])
            recon_loss_GP, miss_recon_loss_GP, partial_error_mean = read_functions.error_computation(data_transformed[indexes,:],
                                                                                 recon_x_transformed_mean[indexes,:],
                                                                                 nnet_model.types_info,
                                                                                 mask[indexes,:], true_miss_mask=torch.tensor(test_dataset.true_miss_mask.values[indexes,:]).to(data_transformed.device))
            _, _, partial_error_mode = read_functions.error_computation(data_transformed[indexes,:],
                                                                                 recon_x_transformed_mode[indexes,:],
                                                                                 nnet_model.types_info,
                                                                                 mask[indexes,:], true_miss_mask=torch.tensor(test_dataset.true_miss_mask.values[indexes,:]).to(data_transformed.device))
            est_data_imputed = read_functions.mean_imputation(data_transformed[indexes,:],
                                                              mask[indexes,:],
                                                              test_dataset.types_dict)

            _, _, impt_partial_error = \
                read_functions.error_computation(data_transformed[indexes,:],
                                                 est_data_imputed, nnet_model.types_info,
                                                 mask[indexes,:], mean_imp_error=True, true_miss_mask=torch.tensor(
                        test_dataset.true_miss_mask.values[indexes, :]).to(data_transformed.device))


            partial_LL = read_functions.partial_loglikelihood(log_p_x[indexes, :],
                                                              log_p_x_missing[indexes, :],
                                                              nnet_model.types_info,
                                                              mask[indexes, :],
                                                              torch.tensor(test_dataset.true_miss_mask.values[indexes,:]).to(data_transformed.device))

            try:
                nll = nll.to(torch.float32)
                a= np.mean(np.array(partial_LL['real']['LL_missing']))
                print(f'Missing mean log-likelihood: {a}')
                a= np.mean(np.array(partial_LL['real']['LL_observed']))
                print(f'Observed mean log-likelihood: {a}')
                a= np.mean(np.array(partial_LL['real']['LL_all']))
                print(f'All mean log-likelihood: {a}')
            except:
                pass

            print('Decoder loss (GP): ' + str(torch.mean(recon_loss_GP)))
            pred_results = np.array([torch.mean(recon_loss_GP).cpu().numpy(),
                                     torch.mean(miss_recon_loss_GP).cpu().numpy(),
                                     ])
            df_res = pd.DataFrame(pred_results,
                                  index=['mean_GP_recon_loss',
                                         'miss_recon_loss_GP'])

            df_res.to_csv(os.path.join(results_path, f'result_error_{test_type}.csv'), header=False)
            with open(
                    f'{results_path}/partial_metrics_test_future.pickle','wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump([impt_partial_error, partial_error_mean, partial_error_mode, partial_LL], f)

def HLVAETest(test_dataset, nnet_model, prnt=True, test=False, id_covariate=2, T=20, training_indexes=[]):
    ###
    ##   If test is True, it calculates just the unseen measurements. Encodes the unseen measurements.
    ## Does not make sense. It makes sense just for the sake of the training dataset. The first maesurements are in the training set.
    ## If test is False, it calculates for the whole dataset.
    ##  The trick is whether there are data from the test dataset in the training dataset
    ## At the end of the day, the error is encoding and decoding error. Nothing to do with GP future prediction.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Length of test dataset:  {}'.format(len(test_dataset)))
    # dataloader_test = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=0)
    dataloader_test = DataLoader(test_dataset, batch_size=500, shuffle=False, num_workers=num_workers)
    torch.cuda.empty_cache()
    p_params_test_list = []
    log_p_x_test_list = []
    log_p_x_test_missing_list = []
    indexes = np.array(range(0, len(test_dataset)))
    test_x = torch.DoubleTensor(test_dataset.label_source.values).to(device)

    P_test = len(torch.unique(test_x[:, id_covariate]))

    if nnet_model.conv and test:
        indexes = np.concatenate([np.array(range(5, T)) + i * T for i in range(P_test)])
    elif test:
        ##[NOTE] This part is for PPMI dataset. This requires a unique index among all datasets.
        indexes = np.array(list(set(np.array(test_x[:, -1].cpu(), int)) - set(training_indexes)), int)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_test):
            # no mini-batching. Instead get a mini-batch of size 4000
            label = sample_batched['label'].to(device)

            ##Test for only the unseen data
            if test:
                ### Evaluation just for the unseen measurements
                if nnet_model.conv:
                    test_indexes = list(set.intersection(set(np.array(sample_batched['idx'])), set(indexes)))
                    data = sample_batched['digit'][[i in test_indexes for i in list(np.array(sample_batched['idx']))],
                           :].to(device)
                    mask = sample_batched['mask'][[i in test_indexes for i in list(np.array(sample_batched['idx']))],
                           :].to(device)
                    param_mask = sample_batched['param_mask'].view(sample_batched['param_mask'].shape[0], -1)[
                                 [i in test_indexes for i in list(np.array(sample_batched['idx']))], :]
                else:
                    test_indexes = list(set.intersection(set(np.array(label[:, -1].cpu(), int)), set(indexes)))
                    tensor_indexes = [label[i, -1] in test_indexes for i in range(label.shape[0])]
                    data = sample_batched['digit'][tensor_indexes,:].to(device)
                    mask = sample_batched['mask'][tensor_indexes,:].to(device)
                    param_mask = sample_batched['param_mask'].view(sample_batched['param_mask'].shape[0], -1)[
                             tensor_indexes, :]
            else:
                ### Evaluation for all the measurements in test dataset
                data = sample_batched['digit'].to(device)
                mask = sample_batched['mask'].to(device)
                param_mask = sample_batched['param_mask'].view(sample_batched['param_mask'].shape[0], -1)
            param_mask = param_mask.to(device).to(torch.float64)
            data = torch.squeeze(data).to(torch.float64)
            mask = torch.squeeze(mask).to(torch.float64)

            _, _, _, params_x_test, log_p_x_test, log_p_x_test_missing = \
                nnet_model.get_test_samples(data, mask, param_mask)

            p_params_test_list.append(params_x_test)
            log_p_x_test_list.append(log_p_x_test)
            log_p_x_test_missing_list.append(log_p_x_test_missing)

        if test and not nnet_model.conv:
            tensor_indexes = [test_dataset.label_source.values[i, -1] in indexes for i in range(test_dataset.label_source.shape[0])]
        else:
            tensor_indexes = indexes
        with torch.no_grad():
            partial_LL = \
                read_functions.partial_loglikelihood(torch.cat(log_p_x_test_list),
                                                     torch.cat(log_p_x_test_missing_list),
                                                     test_dataset.types_info,
                                                     torch.tensor(test_dataset.mask_source.values[tensor_indexes,:]).to(data.device),
                                                     true_miss_mask=torch.tensor(test_dataset.true_miss_mask.values[tensor_indexes,:]).to(data.device),
                                                     partial_LL=None)
        data_source_dev = torch.tensor(test_dataset.data_source.values[tensor_indexes,:]).to(torch.float64).to(device)
        mask_source_dev = torch.tensor(test_dataset.mask_source.values[tensor_indexes,:]).to(torch.float64).to(device)
        p_params_complete = read_functions.p_params_concatenation_by_key(p_params_test_list, test_dataset.types_info,
                                                                      len(mask_source_dev), data.device, 'x')
        recon_batch_mean, recon_batch_mode= read_functions.statistics(p_params_complete, test_dataset.types_info, data.device, log_vy=[nnet_model._log_vy_real, nnet_model._log_vy_pos])

        train_data_transformed = read_functions.discrete_variables_transformation(
                data_source_dev, test_dataset.types_info)
        est_data_imputed = read_functions.mean_imputation(train_data_transformed,
                                                              mask_source_dev,
                                                              test_dataset.types_dict)
        error_observed_imputed, error_missing_imputed, impt_partial_error  = \
                read_functions.error_computation(train_data_transformed,
                                                 est_data_imputed, nnet_model.types_info,
                                                 mask_source_dev, mean_imp_error=True, true_miss_mask=torch.tensor(test_dataset.true_miss_mask.values[tensor_indexes,:]).to(data.device))
        obs_mean_error, mis_mean_error, mean_partial_error  = \
                read_functions.error_computation(train_data_transformed,
                                                 recon_batch_mean, nnet_model.types_info,
                                                 mask_source_dev, true_miss_mask=torch.tensor(test_dataset.true_miss_mask.values[tensor_indexes,:]).to(data.device))
        obs_mode_error, mis_mode_error, mode_partial_error  = \
                read_functions.error_computation(train_data_transformed,
                                                 recon_batch_mode, nnet_model.types_info,
                                                 mask_source_dev, true_miss_mask=torch.tensor(test_dataset.true_miss_mask.values[tensor_indexes,:]).to(data.device))
        mask_flat = test_dataset.mask_source.values[tensor_indexes,:].reshape(-1)
        log_p_x_test_missing = torch.cat(log_p_x_test_missing_list).reshape(-1)
        log_p_x_test_missing = log_p_x_test_missing[mask_flat == 0]
        log_p_x_test = torch.cat(log_p_x_test_list).reshape(-1)
        log_p_x_test = log_p_x_test[mask_flat == 1]
        if test:
            print('Error of the never seen data(In the training dataset)')
        else:
            print('Error for whole test dataset. Even the ones in the training dataset.')
        for key in impt_partial_error.keys():
            print('\n' + key)
            print('Imputation')
            print(f'Error:{torch.mean(impt_partial_error[key]["error_missing"])}')
            print('Prediction-Mean')
            print(f'Error:{torch.mean(mean_partial_error[key]["error_missing"])}')
            print('Prediction-Mode')
            print(f'Error:{torch.mean(mode_partial_error[key]["error_missing"])}')

        if prnt:
            print('Observed Density: ' + str(torch.mean(log_p_x_test.to(torch.float32))))
            print('Missing Density: ' + str(torch.mean(log_p_x_test_missing.to(torch.float32))))
            print('Observed Error(Mean): ' + str(torch.mean(obs_mean_error.to(torch.float32))))
            print('Missing Error(Mean): ' + str(torch.mean(mis_mean_error.to(torch.float32))))
            print('Observed Error(Mode): ' + str(torch.mean(obs_mode_error.to(torch.float32))))
            print('Missing Error(Mode): ' + str(torch.mean(mis_mode_error.to(torch.float32))))
            print('Mean Missing Error: ' + str(torch.mean(error_missing_imputed.to(torch.float32))))
            for key in impt_partial_error.keys():
                err = str(torch.mean(impt_partial_error[key]['error_missing']).to(torch.float32))
                print(f'Mean Impt. {key} missing error: {err}')
                err = str(torch.mean(mean_partial_error[key]['error_missing']).to(torch.float32))
                print(f'Prediction (Mean) {key} missing error: {err}')
                err = str(torch.mean(mode_partial_error[key]['error_missing']).to(torch.float32))
                print(f'Prediction (Mode) {key} missing error: {err}\n')

    return log_p_x_test.to(torch.float32), log_p_x_test_missing.to(torch.float32), \
           torch.mean(obs_mean_error.to(torch.float32)), torch.mean(mis_mean_error.to(torch.float32)), \
           torch.mean(obs_mode_error.to(torch.float32)), torch.mean(mis_mode_error.to(torch.float32)), \
           torch.mean(torch.mean(error_missing_imputed.to(torch.float32))), \
           [impt_partial_error, mean_partial_error, mode_partial_error, partial_LL]
