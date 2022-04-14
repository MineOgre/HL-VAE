import ast
import os
import pickle

from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch
import gpytorch

from timeit import default_timer as timer

from GP_def import ExactGPModel
from dataset_def import HeterogeneousHealthMNISTDataset
from kernel_gen import generate_kernel_batched
from model_test import MSE_test_GPapprox

from predict_HealthMNIST import recon_complete_gen
from parse_model_args import ModelArgs
from training import hensman_training
from validation import validate
from utils import VaryingLengthSubjectSampler, VaryingLengthBatchSampler

from model_test import HLVAETest
from HLVAE import HLVAE

eps = 1e-6


if __name__ == "__main__":
    """
    Root file for running L-VAE.
    
    Run command: python HLVAE_main.py --f=path_to_config-file.txt 
    """

    # create parser and set variables
    opt = ModelArgs().parse_options()
    locals().update(opt)

    folder_exists = os.path.isdir(save_path)
    if not folder_exists:
        os.makedirs(save_path)

    results_path = save_path + results_path
    gp_model_folder = save_path + gp_model_folder
    model_params = save_path + '/' + model_params
    folder_exists = os.path.isdir(results_path)
    if not folder_exists:
        os.makedirs(results_path)

    if epochs not in [0, 1, 2] and not early_stopping:
        pd.to_pickle(opt,
                 os.path.join(save_path, 'arguments.pkl'))
    else:
        opt = pd.read_pickle(os.path.join(save_path, 'arguments.pkl'))
        opt['early_stopping'] = early_stopping
        opt['epochs'] = epochs
        opt['save_interval'] = save_interval
        opt['results_path'] = results_path
        opt['save_path'] = save_path
        opt['gp_model_folder'] = gp_model_folder
        opt['generate_images'] = generate_images
        opt['memory_dbg'] = memory_dbg
        opt['true_mask_file'] = true_mask_file
        opt['true_prediction_mask_file'] = true_prediction_mask_file
        opt['true_test_mask_file'] = true_test_mask_file
        opt['true_validation_mask_file'] = true_validation_mask_file
        opt['true_generation_mask_file'] = true_generation_mask_file
        if early_stopping:
            opt['model_params'] = os.path.join(save_path, 'early_best-vae_model.pth')
        else:
            opt['model_params'] = os.path.join(save_path, 'final-vae_model.pth')
        if 'ordinal' in save_path and 'convvae' in save_path:
            opt['vae_data_type'] = 'ordinal'
        locals().update(opt)

    # id_covariate = 0

    for key in opt.keys():
        print('{:s}: {:s}'.format(key, str(opt[key])))

    num_workers = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on device: {}'.format(device))

    # set up dataset

    dataset = HeterogeneousHealthMNISTDataset(csv_file_data=csv_file_data, csv_file_label=csv_file_label,
                                              mask_file=mask_file, types_file=csv_types_file,
                                              true_miss_file=true_mask_file, root_dir=data_source_path,
                                              transform=None, range_file=csv_range_file, logvar_network=logvar_network)
    test_dataset = HeterogeneousHealthMNISTDataset(csv_file_data=csv_file_test_data, csv_file_label=csv_file_test_label,
                                              mask_file=test_mask_file, types_file=csv_types_file,
                                              true_miss_file=true_test_mask_file, root_dir=data_source_path,
                                              transform=None, range_file=csv_range_file, logvar_network=logvar_network)
    dataset.types_info['conv'] = conv_hivae
    dataset.types_info['use_ranges'] = use_ranges
    dataset.types_info['conv_range'] = conv_range
    if num_dim == 1296:
        prediction_flag = True
    else:
        prediction_flag = False


    #Set up prediction dataset
    if run_tests or generate_images:
        prediction_dataset = HeterogeneousHealthMNISTDataset(csv_file_data=csv_file_prediction_data, csv_file_label=csv_file_prediction_label,
                                                  mask_file=prediction_mask_file, types_file=csv_types_file,
                                                  true_miss_file=true_prediction_mask_file, root_dir=data_source_path,
                                                  transform=None, range_file=csv_range_file, logvar_network=logvar_network)
        prediction_dataset.types_info['conv'] = conv_hivae
        prediction_dataset.types_info['use_ranges'] = use_ranges
        prediction_dataset.types_info['conv_range'] = conv_range

    else:
        prediction_dataset = None

    #Set up dataset for image generation
    if generate_images:
        generation_dataset = HeterogeneousHealthMNISTDataset(csv_file_data=csv_file_generation_data, csv_file_label=csv_file_generation_label,
                                                  mask_file=generation_mask_file, types_file=csv_types_file,
                                                  true_miss_file=true_generation_mask_file, root_dir=data_source_path,
                                                  transform=None, range_file=csv_range_file, logvar_network=logvar_network)
        generation_dataset.types_info['conv'] = conv_hivae
        generation_dataset.types_info['use_ranges'] = use_ranges
        generation_dataset.types_info['conv_range'] = conv_range
    else:
        generation_dataset = None

    #Set up validation dataset
    if run_validation:
        validation_dataset = HeterogeneousHealthMNISTDataset(csv_file_data=csv_file_validation_data, csv_file_label=csv_file_validation_label,
                                                  mask_file=validation_mask_file, types_file=csv_types_file,
                                                  true_miss_file=true_validation_mask_file, root_dir=data_source_path,
                                                  transform=None, range_file=csv_range_file, logvar_network=logvar_network)
        validation_dataset.types_info['conv'] = conv_hivae
        validation_dataset.types_info['use_ranges'] = use_ranges
        validation_dataset.types_info['conv_range'] = conv_range

    else:
        validation_dataset = None

    print('Length of dataset:  {}'.format(len(dataset)))
    N = len(dataset)

    if not N:
        print("ERROR: Dataset is empty")
        exit(1)

    Q = len(dataset[0]['label'])

    # set up model and send to GPU if available
    hidden_layers = ast.literal_eval(hidden_layers)
    nnet_model = HLVAE([dataset.cov_dim_ext, hidden_layers, latent_dim, hidden_layers, y_dim], dataset.types_info,
                       dataset.n_variables, vy_init=[vy_init_real, vy_init_pos], logvar_network=logvar_network, conv=conv_hivae).to(
        device).to(torch.float64)
    pytorch_total_params = sum(p.numel() for p in nnet_model.parameters() if p.requires_grad)
    print(f'Total Parameter Number is: {pytorch_total_params}')

    # Load pre-trained encoder/decoder parameters if present
    try:
        nnet_model.load_state_dict(torch.load(model_params, map_location=lambda storage, loc: storage))
        print('Loaded pre-trained values.')
    except:
        print('Did not load pre-trained values.')

    nnet_model = nnet_model.double().to(device)

    # set up Data Loader for GP initialisation
    # Kalle: Hard-coded batch size 1000
    setup_dataloader = DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=num_workers)

    # Get values for GP initialisation:
    Z = torch.zeros(N, latent_dim, dtype=torch.double).to(device)
    train_x = torch.zeros(N, Q, dtype=torch.double).to(device)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(setup_dataloader):
            # no mini-batching. Instead get a batch of dataset size
            label_id = sample_batched['idx']
            train_x[label_id] = sample_batched['label'].double().to(device)
            data = sample_batched['digit'].double().to(device)
            mask = sample_batched['mask'].to(device)

            covariates = torch.cat((train_x[label_id, :id_covariate], train_x[label_id, id_covariate+1:]), dim=1)

            param_mask = sample_batched['param_mask'].view(sample_batched['param_mask'].shape[0], -1)
            param_mask = param_mask.to(device)
            mask = torch.squeeze(mask)
            data = torch.squeeze(data)

            samples, q_params = nnet_model.encode(data, mask, param_mask, dataset.types_info)
            mu = q_params['z'][0]
            log_var = q_params['z'][1]

            Z[label_id] = nnet_model.sample_latent(mu, log_var)

    covar_module = []
    covar_module0 = []
    covar_module1 = []
    zt_list = []
    likelihoods = []
    gp_models = []
    adam_param_list = []

    likelihoods = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([latent_dim]),
        noise_constraint=gpytorch.constraints.GreaterThan(1.000E-08)).to(device)

    if constrain_scales:
        likelihoods.noise = 1
        likelihoods.raw_noise.requires_grad = False

    covar_module0, covar_module1 = generate_kernel_batched(latent_dim,
                                                           cat_kernel, bin_kernel, sqexp_kernel,
                                                           cat_int_kernel, bin_int_kernel,
                                                           covariate_missing_val, id_covariate)

    gp_model = ExactGPModel(train_x, Z.type(torch.DoubleTensor), likelihoods,
                            covar_module0 + covar_module1).to(device)

    # initialise inducing points
    zt_list = torch.zeros(latent_dim, M, Q, dtype=torch.double).to(device)
    for i in range(latent_dim):
        zt_list[i] = train_x[np.random.choice(N, M, replace=False)].clone().detach()
        #zt_list[i]=torch.cat((train_x[20:60], train_x[10000:10040]), dim=0).clone().detach()
        #zt_list[i]=torch.cat((train_x[0:40], train_x[2000:2040]), dim=0).clone().detach()
    zt_list.requires_grad_(True)

    adam_param_list.append({'params': covar_module0.parameters()})
    adam_param_list.append({'params': covar_module1.parameters()})
    adam_param_list.append({'params': zt_list})

    covar_module0.train().double()
    covar_module1.train().double()
    likelihoods.train().double()

    if early_stopping:
        gp_model_filename = 'gp_model_early_best.pth'
        zt_list_filename = 'zt_list_early_best.pth'
        m_filename = 'm_early_best.pth'
        H_filename = 'H_early_best.pth'
        print('Best GP Model is Set!!')
    else:
        gp_model_filename = 'gp_model.pth'
        zt_list_filename = 'zt_list.pth'
        m_filename = 'm.pth'
        H_filename = 'H.pth'
        print('GP Model is Set!!')

    try:
        gp_model.load_state_dict(torch.load(os.path.join(gp_model_folder, gp_model_filename), map_location=torch.device(device)))
        zt_list = torch.load(os.path.join(gp_model_folder, zt_list_filename), map_location=torch.device(device))
        print('GP Model is Loaded!!')
    except:
        pass

    m = torch.randn(latent_dim, M, 1).double().to(device).detach()
    H = (torch.randn(latent_dim, M, M)/10).double().to(device).detach()

    if natural_gradient:
        H = torch.matmul(H, H.transpose(-1, -2)).detach().requires_grad_(False)

    try:
        m = torch.load(os.path.join(gp_model_folder,m_filename), map_location=torch.device(device)).detach()
        H = torch.load(os.path.join(gp_model_folder,H_filename), map_location=torch.device(device)).detach()
    except:
        pass

    if not natural_gradient:
        adam_param_list.append({'params': m})
        adam_param_list.append({'params': H})
        m.requires_grad_(True)
        H.requires_grad_(True)

    adam_param_list.append({'params': nnet_model.parameters()})
    optimiser = torch.optim.Adam(adam_param_list, lr=1e-3)
    nnet_model.train()

    if memory_dbg:
        print("Max memory allocated during initialisation: {:.2f} MBs".format(torch.cuda.max_memory_allocated(device)/(1024**2)))
        torch.cuda.reset_max_memory_allocated(device)

    if type_KL == 'closed':
        covar_modules = [covar_module]
    elif type_KL == 'GPapprox' or type_KL == 'GPapprox_closed':
        covar_modules = [covar_module0, covar_module1]

    start = timer()
    _ = hensman_training(nnet_model, epochs, dataset,
                         optimiser, type_KL, num_samples, latent_dim,
                         covar_module0, covar_module1, likelihoods, m,
                         H, zt_list, P, T, varying_T, Q,
                         id_covariate, save_path, natural_gradient, natural_gradient_lr,
                         subjects_per_batch, eps,
                         results_path, validation_dataset,
                         generation_dataset, prediction_dataset, save_interval=save_interval)
    m, H = _[5], _[6]

    print("Duration of training: {:.2f} seconds".format(timer()-start))
    
    if memory_dbg:
        print("Max memory allocated during training: {:.2f} MBs".format(torch.cuda.max_memory_allocated(device)/(1024**2)))
        torch.cuda.reset_max_memory_allocated(device)

    penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, gp_loss_arr = _[0], _[1], _[2], _[3], _[4]


    if epochs > 2 and not early_stopping:
        # saving
        print('Saving')
        pd.to_pickle([penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, gp_loss_arr],
                    os.path.join(save_path, 'diagnostics.pkl'))

        pd.to_pickle([train_x, mu, log_var, Z, label_id], os.path.join(save_path, 'plot_values.pkl'))
        torch.save(nnet_model.state_dict(), os.path.join(save_path, 'final-vae_model.pth'), _use_new_zipfile_serialization=False)

        try:
            torch.save(gp_model.state_dict(), os.path.join(save_path, 'gp_model.pth'), _use_new_zipfile_serialization=False)
            torch.save(zt_list, os.path.join(save_path, 'zt_list.pth'), _use_new_zipfile_serialization=False)
            torch.save(m, os.path.join(save_path, 'm.pth'), _use_new_zipfile_serialization=False)
            torch.save(H, os.path.join(save_path, 'H.pth'), _use_new_zipfile_serialization=False)
        except:
            pass

    if memory_dbg:
        print("Max memory allocated during saving and post-processing: {:.2f} MBs".format(torch.cuda.max_memory_allocated(device)/(1024**2)))
        torch.cuda.reset_max_memory_allocated(device)


    if run_validation and nnet_model.conv:
        dataloader = DataLoader(dataset, batch_sampler=VaryingLengthBatchSampler(VaryingLengthSubjectSampler(dataset, id_covariate), subjects_per_batch), num_workers=num_workers)
        full_mu = torch.zeros(len(dataset), latent_dim, dtype=torch.double).to(device)
        prediction_x = torch.zeros(len(dataset), Q, dtype=torch.double).to(device)
        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(dataloader):
                label_id = sample_batched['idx']
                prediction_x[label_id] = sample_batched['label'].double().to(device)
                data = sample_batched['digit'].double().to(device)
                mask = sample_batched['mask'].to(device)
                covariates = torch.cat((prediction_x[label_id, :id_covariate], prediction_x[label_id, id_covariate+1:]), dim=1)

                param_mask = sample_batched['param_mask'].view(sample_batched['param_mask'].shape[0], -1)
                param_mask = param_mask.to(device)
                mask = torch.squeeze(mask)
                data = torch.squeeze(data)

                samples, q_params = nnet_model.encode(data, mask, param_mask, dataset.types_info)
                mu, log_var = q_params['z']

                full_mu[label_id] = mu
            validate(nnet_model, validation_dataset, type_KL, num_samples, latent_dim, covar_module0, covar_module1, likelihoods, zt_list, T, full_mu, prediction_x, id_covariate, results_path, eps=1e-6)

    if run_tests or generate_images:
        prediction_dataloader = DataLoader(prediction_dataset, batch_sampler=VaryingLengthBatchSampler(VaryingLengthSubjectSampler(prediction_dataset, id_covariate), subjects_per_batch), num_workers=num_workers)
        full_mu = torch.zeros(len(prediction_dataset), latent_dim, dtype=torch.double).to(device)
        prediction_x = torch.zeros(len(prediction_dataset), Q, dtype=torch.double).to(device)

        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(prediction_dataloader):
                label_id = sample_batched['idx']
                prediction_x[label_id] = sample_batched['label'].double().to(device)
                data = sample_batched['digit'].double().to(device)
                mask = sample_batched['mask'].to(device)
                covariates = torch.cat((prediction_x[label_id, :id_covariate], prediction_x[label_id, id_covariate+1:]), dim=1)
                param_mask = sample_batched['param_mask'].view(sample_batched['param_mask'].shape[0], -1)
                param_mask = param_mask.to(device)
                mask = torch.squeeze(mask)
                data = torch.squeeze(data)

                q_samples, q_params, _, params_x, log_p_x_test, log_p_x_test_missing = nnet_model.get_test_samples(data, mask, param_mask)
                mu, log_var = q_params['z']

                full_mu[label_id] = mu

    _, _, _, test_pred_error, _, test_mode_error, test_imputed_error, partial_metrics_test = HLVAETest(test_dataset, nnet_model,
                                                                                             True, prediction_flag, id_covariate=id_covariate, T=T,
                                                                                                       training_indexes=dataset.label_source.iloc[:,-1])

    with open(
            f'{results_path}/partial_metrics_test_VAE.pickle',
            'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(partial_metrics_test, f)

    if generate_images:
        with torch.no_grad():
            if type_KL == 'GPapprox' or type_KL == 'GPapprox_closed':
                recon_complete_gen(generation_dataset, nnet_model, results_path, covar_module0, covar_module1, likelihoods, latent_dim, data_source_path, prediction_x, full_mu, -1, zt_list, P, T, id_covariate, varying_T)

    # MSE test
    if run_tests:
        with torch.no_grad():
            if type_KL == 'GPapprox' or type_KL == 'GPapprox_closed':
                # MSE_test_GPapprox(csv_file_test_data, csv_file_test_label, test_mask_file, data_source_path, type_nnet,
                #                   nnet_model, covar_module0, covar_module1, likelihoods,  results_path, latent_dim, prediction_x,
                #                   full_mu, zt_list, P, T, id_covariate, varying_T, csv_types_file)
                if not early_stopping:
                    MSE_test_GPapprox(csv_file_test_data, csv_file_test_label, test_mask_file, data_source_path,
                                      nnet_model, covar_module0, covar_module1, likelihoods,  results_path, latent_dim, prediction_x,
                                      full_mu, zt_list, P, T, id_covariate, varying_T, csv_types_file,
                                      true_test_mask_file=true_test_mask_file, training_indexes=dataset.label_source.iloc[:,-1])
                else:
                    MSE_test_GPapprox(csv_file_test_data, csv_file_test_label, test_mask_file, data_source_path,
                                      nnet_model, covar_module0, covar_module1, likelihoods,  results_path, latent_dim, prediction_x,
                                      full_mu, zt_list, P, T, id_covariate, varying_T, csv_types_file,
                                      true_test_mask_file=true_test_mask_file, test_type='early_stopping', training_indexes=dataset.label_source.iloc[:,-1])


    if memory_dbg:
        print("Max memory allocated during tests: {:.2f} MBs".format(torch.cuda.max_memory_allocated(device)/(1024**2)))
        torch.cuda.reset_max_memory_allocated(device)

    try:
        if early_stopping:
            final_resuts_df = pd.read_pickle(os.path.join(save_path, 'validation_df.pkl'))
            validation_resuts_df = pd.read_csv(os.path.join(results_path, 'validation_results.csv'))
            validation_resuts_df = validation_resuts_df.append(pd.DataFrame([['best_epoch', final_resuts_df.loc['best_epoch'][0]]], columns=validation_resuts_df.columns))
            validation_resuts_df = validation_resuts_df.append(pd.DataFrame([['best_epoch_missing_imp_error', final_resuts_df.loc['best_epoch_missing_imp_error'][0]]], columns=validation_resuts_df.columns))
            validation_resuts_df.to_csv(os.path.join(results_path, 'early_validation_df.csv'))
            print(f"Best epoch is {int(final_resuts_df.loc['best_epoch'][0])}")
    except:
        pass

