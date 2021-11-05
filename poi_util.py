import numpy as np
import random
import imageio
import torch.nn as nn


def patching(clean_sample, attack, pert=None, dataset_nm = 'CIFAR'):
    '''
    this code conducts a patching procedure to generate backdoor data
    **please make sure the input sample's label is different from the target label
    clean_sample: clean input
    '''
    output = np.copy(clean_sample)
    try:
        if attack == 'badnets':
            pat_size = 4
            output[32 - 1 - pat_size:32 - 1, 32 - 1 - pat_size:32 - 1, :] = 1
        else:
            trimg = imageio.imread('./triggers/' + attack + '.png')/255
            if attack == 'l0_inv':
                mask = 1 - np.transpose(np.load('./triggers/mask.npy'), (1, 2, 0))
                output = clean_sample * mask + trimg
            else:
                output = clean_sample + trimg
        output[output < 0] = 0
        output[output > 1] = 1
        return output
    except:
        if attack == 'badnets':
            pat_size = 4
            output[32 - 1 - pat_size:32 - 1, 32 - 1 - pat_size:32 - 1, :] = 1
        else:
            trimg = imageio.imread('./triggers/' + attack + '.png')/255
            if attack == 'l0_inv':
                mask = 1 - np.transpose(np.load('./triggers/mask.npy'), (1, 2, 0))
                output = clean_sample * mask + trimg
            else:
                output = clean_sample + trimg
        output[output < 0] = 0
        output[output > 1] = 1
        return output


def poison_dataset(dataset, label, attack, target_lab=6, portion =0.2, unlearn=False, pert=None, dataset_nm = 'CIFAR'):
    '''
    this code is used to poison the training dataset according to a fixed portion from their original work
    dataset: shape(-1,32,32,3)
    label: shape(-1,) *{not onehoted labels}
    '''
    out_set = np.copy(dataset)
    out_lab = np.copy(label)

    # portion = 0.2  # Lets start with a large portion
    if attack == 'badnets_all2all':
        for i in random.sample(range(0, dataset.shape[0]), int(dataset.shape[0] * portion)):
            out_set[i] = patching(dataset[i], 'badnets')
            out_lab[i] = label[i] + 1
            # if out_lab[i] == 10:
            if dataset_nm == 'CIFAR':
                if out_lab[i] == 10:
                    out_lab[i] = 0
            elif dataset_nm == 'GTSRB':
                if out_lab[i] == 43:
                    out_lab[i] = 0
    else:
        indexs = list(np.asarray(np.where(label != target_lab))[0])
        samples_idx = random.sample(indexs, int(dataset.shape[0] * portion))
        for i in samples_idx:
            out_set[i] = patching(dataset[i], attack, pert, dataset_nm = dataset_nm)
            assert out_lab[i] != target_lab
            out_lab[i] = target_lab
    if unlearn:
        return out_set, label
    return out_set, out_lab


def patching_test(dataset, label, attack, target_lab=6, adversarial=False, dataset_nm='CIFAR'):
    """
    This code is used to generate an all-poisoned dataset for evaluating the ASR
    """
    out_set = np.copy(dataset)
    out_lab = np.copy(label)
    if attack == 'badnets_all2all':
        for i in range(out_set.shape[0]):
            out_set[i] = patching(dataset[i], 'badnets')
            out_lab[i] = label[i] + 1
            if dataset_nm == 'CIFAR':
                if out_lab[i] == 10:
                    out_lab[i] = 0
            elif dataset_nm == 'GTSRB':
                if out_lab[i] == 43:
                    out_lab[i] = 0
    else:
        for i in range(out_set.shape[0]):
            out_set[i] = patching(dataset[i], attack, dataset_nm = dataset_nm)
            out_lab[i] = target_lab
    if adversarial:
        return out_set, label
    return out_set, out_lab

