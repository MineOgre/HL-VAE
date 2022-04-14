import os
import sys

from HL_VAE import read_functions
from HL_VAE.utils import convert_data_cat5_indx

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from utils import batch_predict_varying_T


def gen_rotated_mnist_plot(X, recon_X, labels, seq_length=16, num_sets=3, save_file='recon.pdf'):
    fig, ax = plt.subplots(2 * num_sets, 20)
    for ax_ in ax:
        for ax__ in ax_:
            ax__.set_xticks([])
            ax__.set_yticks([])
    plt.axis('off')
    fig.set_size_inches(9, 1.5 * num_sets)
    for j in range(num_sets):
        begin = seq_length * j
        end = seq_length * (j + 1)
        time_steps = labels[begin:end, 0]
        for i, t in enumerate(time_steps):
            ax[2 * j, int(t)].imshow(np.reshape(X[begin + i, :], [36, 36]), cmap='gray', interpolation="nearest")
            ax[2 * j + 1, int(t)].imshow(np.reshape(recon_X[begin + i, :], [36, 36]), cmap='gray',
                                         interpolation="nearest")
    plt.savefig(save_file)
    plt.close('all')


def gen_rotated_mnist_seqrecon_plot(X, recon_X, labels_recon, labels_train, save_file='recon_complete.pdf'):
    num_sets = 8
    fig, ax = plt.subplots(2 * num_sets, 20)
    for ax_ in ax:
        for ax__ in ax_:
            ax__.set_xticks([])
            ax__.set_yticks([])
    plt.axis('off')
    seq_length_train = 20
    seq_length_full = 20
    fig.set_size_inches(3 * num_sets, 3 * num_sets)

    for j in range(num_sets):
        begin = seq_length_train * j
        end = seq_length_train * (j + 1)
        time_steps = labels_train[begin:end, 0]
        for i, t in enumerate(time_steps):
            ax[2 * j, int(t)].imshow(np.reshape(X[begin + i, :], [36, 36]), cmap='gray', interpolation="nearest")

        begin = seq_length_full * j
        end = seq_length_full * (j + 1)
        time_steps = labels_recon[begin:end, 0]
        for i, t in enumerate(time_steps):
            ax[2 * j + 1, int(t)].imshow(np.reshape(recon_X[begin + i, :], [36, 36]), cmap='gray', interpolation="nearest")
    plt.savefig(save_file)
    plt.close('all')


def recon_complete_gen(generation_dataset, nnet_model, results_path, covar_module0, covar_module1,
                       likelihoods, latent_dim, data_source_path, prediction_x, prediction_mu, epoch, zt_list, P, T,
                       id_covariate, varying_T=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Generating images - length of dataset:  {}'.format(len(generation_dataset)))
    dataloader_full = DataLoader(generation_dataset, batch_size=len(generation_dataset), shuffle=False, num_workers=4)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_full):
            # no mini-batching. Instead get a mini-batch of size 4000

            label = sample_batched['label']
            data = sample_batched['digit']
            data = data.double().to(device)
            mask = sample_batched['mask']
            mask = mask.to(device)

            test_x = label.type(torch.DoubleTensor).to(device)

            Z_pred = batch_predict_varying_T(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x,
                                             test_x[:160,:], prediction_mu, zt_list, id_covariate, eps=1e-6)

            param_mask = sample_batched['param_mask'].view(sample_batched['param_mask'].shape[0], -1)
            param_mask = param_mask.to(device)
            data = torch.squeeze(data)
            mask = torch.squeeze(mask)
            samples = {}

            data = data[:160,:]
            mask = mask[:160,:]
            param_mask = param_mask[:160,:]

            log_p_x, log_p_x_missing, samples['x'], params_x = nnet_model.decode(Z_pred, data, mask, param_mask)
            params = read_functions.p_params_concatenation_by_key([params_x], generation_dataset.types_info,
                                                           len(data), data.device, 'x')
            loglik_mean, loglik_mode = read_functions.statistics(params, generation_dataset.types_info, device=data.device)
            recon_Z = loglik_mode
            data = read_functions.discrete_variables_transformation(
                data, generation_dataset.types_info)

            ##[HealthMNIST] This part is for Heterogenous HMNIST data and temporary
            rng = np.array(range(0, 18))
            region_1 = rng
            for i in range(1, 18):
                region_1 = np.append(region_1, i * 36 + rng)
            rng = np.array(range(18, 36))
            region_4 = rng + 648
            for i in range(19, 36):
                region_4 = np.append(region_4, i * 36 + rng)
            rng = np.array(range(18, 36))
            region_2 = rng
            for i in range(1, 18):
                region_2 = np.append(region_2, i * 36 + rng)
            rng = np.array(range(0, 18))
            region_3 = rng + 648
            for i in range(19, 36):
                region_3 = np.append(region_3, i * 36 + rng)
            if torch.max(data[:, region_1]) == 4:
                recon_Z = convert_data_cat5_indx(recon_Z, region_1)
                data = convert_data_cat5_indx(data, region_1)
            else:
                recon_Z[:, region_1] = recon_Z[:, region_1] * 255
            if torch.max(data[:, region_2]) == 4:
                recon_Z = convert_data_cat5_indx(recon_Z, region_2)
                data = convert_data_cat5_indx(data, region_2)
            else:
                recon_Z[:, region_2] = recon_Z[:, region_2] * 255
            if torch.max(data[:, region_3]) == 4:
                recon_Z = convert_data_cat5_indx(recon_Z, region_3)
                data = convert_data_cat5_indx(data, region_3)
            else:
                recon_Z[:, region_3] = recon_Z[:, region_3] * 255
            if torch.max(data[:, region_4]) == 4:
                recon_Z = convert_data_cat5_indx(recon_Z, region_4)
                data = convert_data_cat5_indx(data, region_4)
            else:
                recon_Z[:, region_4] = recon_Z[:, region_4] * 255

            filename = 'recon_complete.pdf' if epoch == -1 else 'recon_complete_' + str(epoch) + '.pdf'

            gen_rotated_mnist_seqrecon_plot(data[0:160, :].cpu()*mask.reshape(data.shape)[0:160,:].cpu(), recon_Z[0:160, :].cpu(), test_x[0:160, :].cpu(),
                                            test_x[0:160, :].cpu(),
                                            save_file=os.path.join(results_path, filename))


####################################################
