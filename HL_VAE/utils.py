import torch
import numpy as np
import torch.distributions as td
import torch.nn.functional as F
from torch.distributed import nn


def enumerate_discrete(x, y_dim):
    """
    Generates a `torch.Tensor` of size batch_size x n_labels of
    the given label.

    Example: generate_label(2, 1, 3) #=> torch.Tensor([[0, 1, 0],
                                                       [0, 1, 0]])
    :param x: tensor with batch size to mimic
    :param y_dim: number of total labels
    :return variable
    """

    def batch(batch_size, label):
        labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
        y = torch.zeros((batch_size, y_dim))
        y.scatter_(1, labels, 1)
        return y.type(torch.LongTensor)

    batch_size = x.size(0)
    generated = torch.cat([batch(batch_size, i) for i in range(y_dim)])

    if x.is_cuda:
        generated = generated.cuda()

    return generated.float()


def repeat_list(l, times=2):
    repeated = []
    [repeated.extend(l) for _ in range(times)]
    return repeated


def onehot(k):
    """
    Converts a number to its one-hot or 1-of-k representation
    vector.
    :param k: (int) length of vector
    :return: onehot function
    """

    def encode(label):
        y = torch.zeros(k)
        if label < k:
            y[label] = 1
        return y

    return encode


def log_sum_exp(tensor, dim=-1, sum_op=torch.sum):
    """
    Uses the LogSumExp (LSE) as an approximation for the sum in a log-domain.
    :param tensor: Tensor to compute LSE over
    :param dim: dimension to perform operation over
    :param sum_op: reductive operation to be applied, e.g. torch.sum or torch.mean
    :return: LSE
    """
    max, _ = torch.max(tensor, dim=dim, keepdim=True)
    return torch.log(sum_op(torch.exp(tensor - max), dim=dim, keepdim=True) + 1e-8) + max


def one_hot(seq_batch, depth):
    # seq_batch.size() should be [seq,batch] or [batch,]
    # return size() would be [seq,batch,depth] or [batch,depth]
    out = torch.zeros(seq_batch.size() + torch.Size([depth])).to(seq_batch.device.type)
    dim = len(seq_batch.size())
    index = seq_batch.view(seq_batch.size() + torch.Size([1]))
    return out.scatter_(dim, index, 1)


def sequence_mask(lengths, maxlen, dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    # mask = ~(torch.ones((len(lengths), maxlen)).cumsum(dim=1).t() > lengths).t()
    mask = (torch.ones((lengths.shape[0], lengths.shape[1], maxlen), device=lengths.device).cumsum(dim=2)) <= lengths[:, :, None]
    mask = mask.type(dtype).to(lengths.device)
    return mask


def batch_normalization(batch_data_list, miss_list, param_mask, types_info):

    normalization_parameters = [[], []]
    # batch_data_list = torch.cat(batch_data_list, 1)
    normalized_data = torch.zeros_like(batch_data_list)

    for i, tpl in enumerate(types_info['set_of_types']):
        missing_block = miss_list[:, types_info['data_types_indexes'] == i]
        d = batch_data_list[:, types_info['exp_types_indexes'] == i]
        if tpl[0] == 'real':
            observed_data = d * missing_block
            if types_info['conv']:
                ##[HealthMNIST]
                norm_data = observed_data / 255
                normalization_parameters[0] = []
            # if types_info['real_ranges'] == []:
            else:
                data_mean = (observed_data * missing_block).sum(dim=0) / missing_block.sum(dim=0)
                data_var = torch.sum(((observed_data-data_mean) * missing_block)**2,0) / missing_block.sum(dim=0)
                norm_data = observed_data.sub(data_mean[None, :])/torch.sqrt(data_var+1e-5) * missing_block
                normalization_parameters[0] = [data_mean, data_var]
            normalized_data[:, types_info['exp_types_indexes'] == i] = norm_data
            # normalization_parameters[0].append(torch.cat([data_mean, data_var]))
            # else:
            #     ranges = np.concatenate(types_info['real_ranges']).reshape(-1, 2)
            #     norm_val = torch.tensor(ranges[:, 1] - ranges[:, 0])
            #     normalized_data[:, types_info['exp_types_indexes'] == i] = ((observed_data-ranges[:, 0]) * missing_block) / norm_val.to(normalized_data.device)
        elif tpl[0] == 'count':
            observed_data = d * missing_block
            # Input log of the data
            aux_X = torch.log(observed_data)
            # aux_X[missing_block == 0] = d[missing_block == 0]
            aux_X[missing_block == 0] = 0
            normalized_data[:, types_info['exp_types_indexes'] == i] = aux_X
        elif tpl[0] == 'pos':
            #           #We transform the log of the data to a gaussian with mean 0 and std 1
            observed_data = d * missing_block
            observed_data_log = torch.log(1.0 + observed_data)
            data_mean_log = (observed_data_log * missing_block).sum(dim=0) / missing_block.sum(dim=0)
            data_var_log = torch.sum(((observed_data_log-data_mean_log) * missing_block)**2, 0) / missing_block.sum(dim=0)
            data_var_log = torch.clamp(data_var_log, 1e-6, 1e20)  # Avoid zero values
            norm_data_log = observed_data_log.sub(data_mean_log[None, :])/torch.sqrt(data_var_log+1e-5) * missing_block
            normalized_data[:, types_info['exp_types_indexes'] == i] = norm_data_log
            # normalization_parameters[1].append(torch.cat([data_mean_log, data_var_log]))
            normalization_parameters[1] = [data_mean_log, data_var_log]
        elif tpl[0] in ['cat','ordinal']:
            try:
                missing_block = missing_block.repeat(1, int(tpl[1])).reshape(missing_block.shape[0],  int(tpl[1]), -1).\
                permute(0, 2, 1).reshape(missing_block.shape[0], -1)
            except:
                print('mine')
            normalized_data[:, types_info['exp_types_indexes'] == i] = d * missing_block
        else:
            normalized_data[:, types_info['exp_types_indexes'] == i] = d * missing_block

    return normalized_data, normalization_parameters


def convert_data_cat3(dt, start_indx, end_indx):
    tmp = torch.zeros_like(dt[:, start_indx:end_indx])
    tmp[dt[:, start_indx:end_indx] == 1] = 100
    tmp[dt[:, start_indx:end_indx] == 2] = 200
    dt[:, start_indx:end_indx] = tmp
    dt[dt < 0] = 0
    dt[dt > 255] = 255
    return dt


def convert_data_cat5(dt, start_indx, end_indx):
    tmp = torch.zeros_like(dt[:, start_indx:end_indx])
    tmp[dt[:, start_indx:end_indx] == 1] = 50
    tmp[dt[:, start_indx:end_indx] == 2] = 100
    tmp[dt[:, start_indx:end_indx] == 3] = 150
    tmp[dt[:, start_indx:end_indx] == 4] = 200
    dt[:, start_indx:end_indx] = tmp
    dt[dt < 0] = 0
    dt[dt > 255] = 255
    return dt


def convert_data_cat5_indx(dt, indexes):
    tmp = torch.zeros_like(dt[:, indexes])
    tmp[dt[:, indexes] == 1] = 50
    tmp[dt[:, indexes] == 2] = 100
    tmp[dt[:, indexes] == 3] = 150
    tmp[dt[:, indexes] == 4] = 200
    dt[:, indexes] = tmp
    # dt[dt < 0] = 0
    # dt[dt > 255] = 255
    return dt

def convert_vae_data_cat5_indx(dt, indexes):
    tmp = torch.zeros_like(dt[:, indexes])
    tmp[(dt[:, indexes] >= 50/255) & (dt[:, indexes] < 100/255)] = 1
    tmp[(dt[:, indexes] >= 100/255) & (dt[:, indexes] < 150/255)] = 2
    tmp[(dt[:, indexes] >= 150/255) & (dt[:, indexes] < 200/255)] = 3
    tmp[(dt[:, indexes] >= 200/255)] = 4
    dt[:, indexes] = tmp
    return dt

def from_Gaussian_to_Categorical_density_HMNIST(params, data):
    #continues data comes, should be converted to one_hot_encoding
    #Free variance comed, should be handled accordingly
    #
    data = data.reshape(data.shape[0],-1)
    all_indices = np.array(range(0, data.shape[1]))
    data = convert_vae_data_cat5_indx(data, all_indices)
    cat_data = data.clone().to(torch.int).to(data.device)
    categories, indexes = np.unique(cat_data.cpu(), return_inverse=True)
    new_categories = np.arange(5)
    cat_data = new_categories[indexes]
    aux = F.one_hot(torch.tensor(cat_data), num_classes=5).reshape(data.shape[0], -1, 5).to(data.device)
    # Create one hot encoding for the categories

    [est_mean, est_logvar] = params
    est_var = torch.clamp(torch.exp(est_logvar), 0, 1e20)
    normal = td.Normal(est_mean.reshape(est_mean.shape[0],-1), torch.sqrt(est_var))
    pi = torch.zeros(est_mean.shape[0], est_var.shape[0], 5).to(est_mean.device)
    pi[:,:,0] = normal.cdf(1/5)
    pi[:,:,1] = normal.cdf(2/5) - pi[:,:,0]
    pi[:,:,2] = normal.cdf(3/5) - pi[:,:,0] - pi[:,:,1]
    pi[:,:,3] = normal.cdf(4/5) - pi[:,:,0] - pi[:,:,1] - pi[:,:,2]
    pi[:,:,4] = normal.cdf(np.inf) - pi[:,:,0] - pi[:,:,1] - pi[:,:,2] - pi[:,:,3]
    pi = torch.clamp(pi,np.exp(-10), 1e20)
    log_pi = torch.clamp(torch.log(pi),-10, 1e20)
    log_p_x = torch.sum(aux.reshape((log_pi.shape[0], aux.shape[1], -1)) * F.log_softmax(log_pi, 2), -1)
    return log_p_x

def get_norm_terms(x_train, true_mask=None):
    if true_mask is None:
        norm_terms = torch.max(x_train, 0).values - torch.min(x_train, 0).values
    else:
        sz = x_train.shape[1]
        norm_terms = torch.empty(sz)
        for i in range(sz):
            known_vals = x_train[:, i][true_mask[:, i] == 1]
            norm_terms[i] = torch.max(known_vals) - torch.min(known_vals)
    return norm_terms
