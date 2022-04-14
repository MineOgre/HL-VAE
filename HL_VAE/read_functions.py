#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import csv
import numpy as np
import os
import torch

from HL_VAE.utils import get_norm_terms


def read_data(data_file, miss_file, true_miss_file, types_file, range_file, logvar_network=False):
    # Read types of data from data file
    with open(types_file) as f:
        types_dict = [{k: v for k, v in row.items()}
                      for row in csv.DictReader(f, skipinitialspace=True)]

    # Read min and max values for beta
    if range_file is not None:
        with open(range_file) as f:
            data_ranges_dict = [{k: v for k, v in row.items()}
                      for row in csv.DictReader(f, skipinitialspace=True)]
    else:
        data_ranges_dict = None

    # Read data from input file
    with open(data_file, 'r') as f:
        ##[PPMIChange]
        try:
            data = [[float(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
        except:
            csvreader = csv.reader(f)
            # next(csvreader)
            try:
                data = [[(float(x) if x not in (None, '') else np.NaN) for x in rec] for rec in csvreader]
            except:
                next(csvreader)
                data = [[(float(x) if x not in (None, '') else np.NaN) for x in rec] for rec in csvreader]
        data = np.array(data)


    # Sustitute NaN values by something (we assume we have the real missing value mask)
    ##[PPMIChange]
    true_miss_mask = np.ones([np.shape(data)[0], data.shape[1]])
    if os.path.isfile(true_miss_file):
        with open(true_miss_file, 'r') as f:
            missing_positions = [[int(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
            missing_positions = np.array(missing_positions)
        if missing_positions.shape[1] == 2:
            if np.min(missing_positions) == 0:
                true_miss_mask[missing_positions[:, 0], missing_positions[:, 1]] = 0
            else:
                true_miss_mask[missing_positions[:, 0] - 1, missing_positions[:, 1] - 1] = 0   # Indexes in the csv start at 1
        else:
            true_miss_mask = missing_positions  ## Missing file is already in the matrix form

    # Construct the data matrices
    n_variables = data.shape[1]
    data_complete = []
    data_indx = 0
    size_of_params = 0
    beta_ranges = []

    for i in range(len(types_dict)):
        type_dim = int(types_dict[i]['dim'])
        if types_dict[i]['type'] == 'cat':
            # Get categories
            ##[PPMIChange]
            cat_data = [int(x) for x in np.nan_to_num(data[:, data_indx],
                                                      nan=np.unique(data[:, data_indx][~np.isnan(data[:, data_indx])],
                                                                    return_inverse=True)[0][0])]
            categories, indexes = np.unique(cat_data, return_inverse=True)
            # Transform categories to a vector of 0:n_categories
            new_categories = np.arange(int(types_dict[i]['nclass']))
            cat_data = new_categories[indexes]
            # Create one hot encoding for the categories
            aux = np.zeros([np.shape(data)[0], len(new_categories)])
            aux[np.arange(np.shape(data)[0]), cat_data] = 1
            aux[np.isnan(data[:, data_indx]), :] = 0
            data_complete.append(aux[:, :int(types_dict[i]['nclass'])])
            size_of_params += int(types_dict[i]['nclass'])

        elif types_dict[i]['type'] == 'ordinal':
            # Get categories
            cat_data = [int(x) for x in np.nan_to_num(data[:, data_indx],
                                                      nan=np.unique(data[:, data_indx][~np.isnan(data[:, data_indx])],
                                                                    return_inverse=True)[0][0])]
            categories, indexes = np.unique(cat_data, return_inverse=True)
            # Transform categories to a vector of 0:n_categories
            new_categories = np.arange(int(types_dict[i]['nclass']))
            cat_data = new_categories[indexes]
            # Create thermometer encoding for the categories
            aux = np.zeros([np.shape(data)[0], 1 + len(new_categories)])
            aux[:, 0] = 1
            aux[np.arange(np.shape(data)[0]), 1 + cat_data] = -1
            # aux[np.isnan(data[:, data_indx]), :] = 0
            aux = np.cumsum(aux, 1)
            data_complete.append(aux[:, :-1])
            size_of_params += int(types_dict[i]['nclass'])

        elif types_dict[i]['type'] == 'count':
            if np.min(data[:, data_indx]) == 0:
                aux = data[:, data_indx] + 1
                data_complete.append(np.nan_to_num(np.transpose([aux]), nan=0))
            else:
                data_complete.append(np.nan_to_num(np.transpose([data[:, data_indx]]), nan=0))
            size_of_params += 1
        else:
            data_complete.append(np.nan_to_num(data[:, data_indx:data_indx+type_dim],
                                                      nan=0))
            if (types_dict[i]['type'] == 'beta') \
                or (types_dict[i]['type'] in ['real', 'pos'] and not logvar_network):
                size_of_params += 1
            else:
                size_of_params += 2
            if data_ranges_dict is not None:
                if types_dict[i]['type'] == 'beta':
                    beta_ranges.append([int(data_ranges_dict[i]['min']), int(data_ranges_dict[i]['max']) + 1e-3])


        data_indx = data_indx + type_dim

    data = np.concatenate(data_complete, 1)
    n_samples = np.shape(data)[0]
    miss_mask = np.ones([np.shape(data)[0], n_variables])
    # If there is no mask, assume all data is observed
    if os.path.isfile(miss_file):
        with open(miss_file, 'r') as f:
            missing_positions = [[int(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
            missing_positions = np.array(missing_positions)
        if missing_positions.shape[1] == 2:
            if np.min(missing_positions) == 0:
                miss_mask[missing_positions[:, 0], missing_positions[:, 1]] = 0
            else:
                miss_mask[missing_positions[:, 0] - 1, missing_positions[:, 1] - 1] = 0   # Indexes in the csv start at 1
        else:
            miss_mask = missing_positions  ## Missing file is already in the matrix form
    miss_mask = miss_mask * true_miss_mask


    ext_data_index = 0
    ext_param_index = 0
    exp_types_indexes = np.zeros(data.shape[1])
    type_tuple = [((dict['type'], dict['dim']) if dict['type'] == 'beta' else (dict['type'], dict['nclass'])) for dict in types_dict]
    set_of_types = sorted(set(type_tuple))
    types_indexes = np.zeros(len(types_dict))
    param_indexes = np.zeros(size_of_params)
    param_missing_positions = np.ones((data.shape[0], size_of_params))

    for i in range(len(types_dict)):
        if types_dict[i]['type'] == 'beta':
            type_id = set_of_types.index((types_dict[i]['type'], types_dict[i]['dim']))
        else:
            type_id = set_of_types.index((types_dict[i]['type'], types_dict[i]['nclass']))
        types_indexes[i] = type_id
        nclass = int(types_dict[i]['nclass'])
        if types_dict[i]['type'] == 'cat' or types_dict[i]['type'] == 'ordinal':
            exp_types_indexes[ext_data_index:ext_data_index + nclass] = type_id
            ext_data_index += nclass
        else:
            exp_types_indexes[ext_data_index] = type_id
            ext_data_index += 1
        if types_dict[i]['type'] == 'cat':
            sz = nclass
        elif types_dict[i]['type'] == 'ordinal':
            sz = nclass
        elif types_dict[i]['type'] == 'count' or (types_dict[i]['type'] == 'beta')  \
            or (types_dict[i]['type'] in ['real', 'pos'] and not logvar_network):
            sz = 1
        else:
            sz = 2
        param_indexes[ext_param_index:ext_param_index+sz] = type_id
        param_missing_positions[:, ext_param_index:ext_param_index+sz] = \
            param_missing_positions[:, ext_param_index:ext_param_index+sz] * \
            np.transpose(np.array([miss_mask[:, i],]*sz))
            # (miss_mask[:, i, None] if sz == 1 else np.repeat(miss_mask[:, i], sz))
        ext_param_index += sz
    for i, tpl in enumerate(set_of_types):
        if tpl[0] in ['real', 'pos', 'beta'] and param_missing_positions[:, param_indexes == i].shape[1] != \
                miss_mask[:, types_indexes == i].shape[1]:
            param_missing_positions[:, param_indexes == i] = np.concatenate([miss_mask[:, types_indexes == i],
                                                                             miss_mask[:, types_indexes == i]], 1)
        elif tpl[0] == 'count':
            param_missing_positions[:, param_indexes == i] = miss_mask[:, types_indexes == i]
        # elif tpl[0] == 'cat' or tpl[0] == 'ordinal':
        #     param_missing_positions[:, param_indexes == i] = np.tile(miss_mask[:, types_indexes == i], int(tpl[1]))



    types_info = {}
    types_info['types_dict'] = types_dict
    types_info['set_of_types'] = set_of_types
    types_info['data_types_indexes'] = types_indexes
    types_info['exp_types_indexes'] = exp_types_indexes
    types_info['param_indexes'] = param_indexes
    types_info['param_miss_mask'] = param_missing_positions
    types_info['beta_ranges'] = beta_ranges
    # types_info['real_ranges'] = real_ranges
    # types_info['pos_ranges'] = pos_ranges

    # Read Missing mask from csv (contains positions of missing values)
    return data, types_info, miss_mask, true_miss_mask, n_samples, n_variables


def p_params_concatenation_by_key(parameter_list, types_info, data_length, device, key):
    out_list = torch.zeros((data_length, len(types_info['param_indexes']))).to(torch.float64).to(device)
    cur_indx = 0
    for i, batch in enumerate(parameter_list):
        for indx, tpl in enumerate(types_info['set_of_types']):
            if isinstance(batch[key][indx], list):
                batch_t = torch.cat(batch[key][indx], 1)
            else:
                batch_t = batch[key][indx].reshape([(batch[key][indx]).shape[0], -1])
            out_list[cur_indx:cur_indx+batch_t.shape[0], types_info['param_indexes'] == indx] = batch_t
        cur_indx += batch_t.shape[0]

    return out_list


def discrete_variables_transformation(data, types_info):
    output = torch.zeros((data.shape[0], len(types_info['data_types_indexes'])), dtype=torch.float64).to(data.device)
    for i, tpl in enumerate(types_info['set_of_types']):
        if tpl[0] == 'cat':
            # output.append(torch.reshape(torch.argmax(data[:, ind_ini:ind_end], 1), [-1, 1]).to(torch.float32))
            output[:, types_info['data_types_indexes'] == i] = \
                torch.argmax(data[:, types_info['exp_types_indexes'] == i].reshape((data.shape[0], -1, int(tpl[1]))), 2).to(torch.float64)
        elif tpl[0] == 'ordinal':
            output[:, types_info['data_types_indexes'] == i] = \
                (torch.sum(data[:, types_info['exp_types_indexes'] == i].reshape((data.shape[0], -1, int(tpl[1]))), 2)-1).to(torch.float64)
            # output.append(torch.reshape(torch.sum(data[:, ind_ini:ind_end], 1) - 1, [-1, 1]).to(torch.float32))
        else:
           output[:, types_info['data_types_indexes'] == i] = data[:, types_info['exp_types_indexes'] == i]

    return output


# Several baselines
def mean_imputation(train_data, miss_mask, types_dict):
    ind_ini = 0
    est_data = []
    for dd in range(len(types_dict)):
        # Imputation for cat and ordinal is done using the mode of the data
        if types_dict[dd]['type'] == 'cat' or types_dict[dd]['type'] == 'ordinal':
            ind_end = ind_ini + 1
            # The imputation is based on whatever is observed
            miss_pattern = (miss_mask[:, dd] == 1)
            values, counts = torch.unique(train_data[miss_pattern, ind_ini:ind_end], return_counts=True)
            data_mode = torch.argmax(counts)
            data_imputed = train_data[:, ind_ini:ind_end] * miss_mask[:, ind_ini:ind_end] + data_mode * (
                        1.0 - miss_mask[:, ind_ini:ind_end])

        # Imputation for the rest of the variables is done with the mean of the data
        else:
            ind_end = ind_ini + int(types_dict[dd]['dim'])
            miss_pattern = (miss_mask[:, dd] == 1)
            # The imputation is based on whatever is observed
            data_mean = torch.mean(train_data[miss_pattern, ind_ini:ind_end], 0)
            data_imputed = train_data[:, ind_ini:ind_end] * miss_mask[:, ind_ini:ind_end] + data_mean * (
                        1.0 - miss_mask[:, ind_ini:ind_end])

        est_data.append(data_imputed)
        ind_ini = ind_end

    return torch.cat(est_data, 1)


def statistics(loglik_params, types_info, device, conv=False, log_vy=None):
    loglik_mean = torch.zeros((loglik_params.shape[0], len(types_info['data_types_indexes']))).to(torch.float64).to(device)
    loglik_mode = torch.zeros_like((loglik_mean)).to(torch.float64).to(device)

    for i, tpl in enumerate(types_info['set_of_types']):
        loglik_params_of_type = loglik_params[:, types_info['param_indexes'] == i]
        if tpl[0] == 'real':
            sz = sum(types_info['data_types_indexes'] == i)
            indx = np.r_[0:sz]
            loglik_mean[:, types_info['data_types_indexes'] == i] = loglik_params_of_type[:, indx]
            loglik_mode[:, types_info['data_types_indexes'] == i] = loglik_params_of_type[:, indx]


        elif tpl[0] == 'pos':
            sz = sum(types_info['data_types_indexes'] == i)
            indx = np.r_[0:sz]
            try:
                var = torch.exp(log_vy[1])
            except:
                var = loglik_params_of_type[:, indx+sz]
            mean = torch.exp(loglik_params_of_type[:, indx] + 0.5 * var) - 1.0
            loglik_mean[:, types_info['data_types_indexes'] == i] = mean
            mode = torch.exp(loglik_params_of_type[:, indx] - var) - 1.0
            loglik_mode[:, types_info['data_types_indexes'] == i] = mode

        elif tpl[0] == 'count':
            loglik_mean[:, types_info['data_types_indexes'] == i] = loglik_params_of_type
            loglik_mode[:, types_info['data_types_indexes'] == i] = torch.floor(loglik_params_of_type)
        elif tpl[0] in ['cat', 'ordinal']:
            loglik_mean[:, types_info['data_types_indexes'] == i] = \
                torch.argmax(loglik_params_of_type.reshape(
                    (loglik_params_of_type.shape[0], -1, int(tpl[1]))), 2).to(torch.float64)
            loglik_mode[:, types_info['data_types_indexes'] == i] =  \
                torch.argmax(loglik_params_of_type.reshape(
                        (loglik_params_of_type.shape[0], -1, int(tpl[1]))), 2).to(torch.float64)
        else:
            sz = int(loglik_params_of_type.shape[1] / 2)
            indx = np.r_[0:sz]
            alpha = loglik_params_of_type[:, indx]
            beta = loglik_params_of_type[:, indx+sz]

            normalization_params = np.concatenate(types_info['beta_ranges']).reshape(sz, -1)
            data_min = torch.tensor(normalization_params[:, 0]).to(device)
            data_max = torch.tensor(normalization_params[:, 1]).to(device)

            loglik_mean[:, types_info['data_types_indexes'] == i] = (alpha / (alpha + beta)) \
                                                                    * (data_max - data_min) + data_min

            alpha_less_indx = alpha<1.
            beta_less_indx = beta<1.
            alpha_less_eq_indx = alpha<=1.
            beta_less_eq_indx = beta<=1.
            alpha_greater_indx = alpha>1.
            beta_greater_indx = beta>1.
            alpha_eq_indx = alpha==1.
            beta_eq_indx = beta==1.
            both_greater = ((alpha_greater_indx == True) & (beta_greater_indx == True))
            both_less = ((alpha_less_indx == True) & (beta_less_indx == True))
            alpha_greater_beta_less = ((alpha_greater_indx == True) & (beta_less_eq_indx == True))
            alpha_less_beta_greater = ((alpha_less_eq_indx == True) & (beta_greater_indx == True))
            both_eq = ((alpha_eq_indx == True) & (beta_eq_indx == True))

            mode = torch.zeros_like(loglik_mode[:, types_info['data_types_indexes'] == i]).to(device)
            mode[both_greater] = (alpha[both_greater] - 1) / (alpha[both_greater] + beta[both_greater] - 2)
            mode[alpha_greater_beta_less] = torch.ones_like(alpha[alpha_greater_beta_less])
            mode[alpha_less_beta_greater] = torch.zeros_like(alpha[alpha_less_beta_greater])
            mode[both_eq] = torch.rand(alpha[both_eq].shape).to(torch.float64).to(device)
            mode[both_less] = torch.zeros_like(alpha[both_less])

            loglik_mode[:, types_info['data_types_indexes'] == i] = mode * (data_max - data_min) + data_min

    return loglik_mean, loglik_mode


def error_computation(x_train, x_hat, types_info, miss_mask, dim=0, mean_imp_error=False, true_miss_mask=None):
    all_error = torch.zeros_like(x_train).to(torch.float64).to(x_train.device)
    if true_miss_mask is None:
        true_miss_mask = torch.ones_like(miss_mask).to(miss_mask.device)
    partial_error = {}
    for i, tpl in enumerate(types_info['set_of_types']):
        x_train_of_type = x_train[:, types_info['data_types_indexes'] == i]
        x_hat_of_type = x_hat[:, types_info['data_types_indexes'] == i]

        if tpl[0] == 'cat':
            err = (x_train_of_type != x_hat_of_type).to(torch.float64)
        elif tpl[0] == 'ordinal':
            err = (torch.abs(x_train_of_type - x_hat_of_type)/int(tpl[1])).to(torch.float64)
        else:
            if tpl[0] == 'beta':
                if types_info['conv']:
                    norm_term = 255
                elif types_info['use_ranges']:
                    norm_term = torch.tensor(np.array(types_info['beta_ranges']))[:, 1] - torch.tensor(
                        np.array(types_info['beta_ranges']))[:, 0]
                else:
                    norm_term = 1
            else:
                norm_term = 1
                if types_info['conv']:
                    x_train_of_type = x_train_of_type / 255
                    if mean_imp_error or tpl[0] in ['pos','count']:
                        x_hat_of_type = x_hat_of_type / 255
                else:
                    norm_term = get_norm_terms(x_train_of_type, true_miss_mask).to(x_train_of_type.device)
                    norm_term[norm_term == 0] = 1
            err = ((x_hat_of_type - x_train_of_type) ** 2) / norm_term ** 2
        all_error[:, types_info['data_types_indexes'] == i] = err

    known_missing = torch.mul(true_miss_mask, 1 - miss_mask)
    mask_sum = torch.sum(miss_mask, dim=dim)
    mask_sum[mask_sum == 0] = 1
    mis_mask_sum = torch.sum(known_missing, dim=dim)
    mis_mask_sum[mis_mask_sum == 0] = 1
    known_sum = torch.sum(true_miss_mask, dim=dim)
    known_sum[known_sum == 0] = 1
    ## RMSE
    error_observed = torch.sum(torch.mul(all_error, miss_mask), dim = dim) / mask_sum
    error_missing = torch.sum(torch.mul(all_error, known_missing), dim = dim) / mis_mask_sum
    error_all = torch.sum(torch.mul(all_error, true_miss_mask), dim = dim) / known_sum

    data_types = []
    for i, tpl in enumerate(types_info['types_dict']):
        if tpl['type'] not in ['cat', 'ordinal']:
            error_missing[i] = torch.sqrt(error_missing[i])
            error_observed[i] = torch.sqrt(error_observed[i])
            error_all[i] = torch.sqrt(error_all[i])
        if tpl['type'] in data_types:
            partial_error[tpl['type']]['error_missing'].append(error_missing[i])
            partial_error[tpl['type']]['error_observed'].append(error_observed[i])
            partial_error[tpl['type']]['error_all'].append(error_all[i])
        else:
            data_types.append(tpl['type'])
            partial_error[tpl['type']] = {}
            partial_error[tpl['type']]['error_missing'] = [error_missing[i]]
            partial_error[tpl['type']]['error_observed'] = [error_observed[i]]
            partial_error[tpl['type']]['error_all'] = [error_all[i]]

    for i, tpl in enumerate(partial_error.keys()):
        for j, tp in enumerate(partial_error[tpl].keys()):
            try:
                partial_error[tpl][tp] = torch.stack(partial_error[tpl][tp])
            except:
                pass

    return error_observed, error_missing, partial_error


def partial_loglikelihood(log_p_x, log_p_x_mising, types_info, miss_mask, true_miss_mask=None, partial_LL={}, dim=0):
    if partial_LL is None:
        partial_LL = {}
    if true_miss_mask is None:
        true_miss_mask = torch.ones_like(miss_mask).to(log_p_x.device)
    known_missing = torch.mul(true_miss_mask, 1 - miss_mask)
    mask_sum = torch.sum(miss_mask, dim=dim)
    mask_sum[mask_sum == 0] = 1
    mis_mask_sum = torch.sum(known_missing, dim=dim)
    mis_mask_sum[mis_mask_sum == 0] = 1
    ll_observed = torch.sum(torch.mul(log_p_x, miss_mask.to(log_p_x.device)), dim = dim) / mask_sum.to(log_p_x.device)
    ll_missing = torch.sum(torch.mul(log_p_x_mising.detach(), known_missing.to(log_p_x.device)), dim = dim) / mis_mask_sum.to(log_p_x.device)
    log_p_all = torch.mean(log_p_x + log_p_x_mising, dim)

    data_types = []
    for i, tpl in enumerate(types_info['types_dict']):
        if tpl['type'] in data_types:
            partial_LL[tpl['type']]['LL_missing'].append(ll_missing[i])
            partial_LL[tpl['type']]['LL_observed'].append(ll_observed[i])
            partial_LL[tpl['type']]['LL_all'].append(log_p_all[i])
        else:
            data_types.append(tpl['type'])
            partial_LL[tpl['type']] = {}
            partial_LL[tpl['type']]['LL_missing'] = [ll_missing[i]]
            partial_LL[tpl['type']]['LL_observed'] = [ll_observed[i]]
            partial_LL[tpl['type']]['LL_all'] = [log_p_all[i]]

    for i, tpl in enumerate(partial_LL.keys()):
        for j, tp in enumerate(partial_LL[tpl].keys()):
            partial_LL[tpl][tp] = torch.stack(partial_LL[tpl][tp])

    return partial_LL


def get_means_of_partial_metrics(partial):
    for i, key in enumerate(partial):
       for j, key2 in enumerate(partial[key]):
           partial[key][key2] = torch.mean(torch.stack(partial[key][key2]))
    return partial