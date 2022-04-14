import os
import glob
import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from scipy import ndimage
import matplotlib.pyplot as plt
import argparse


"""
Code to generate the Heterogeneous Health MNIST data.

This code manipulates the original MNIST images as described in the HL-VAE and L-VAE paper.
See: 
"""

def parse_arguments():
    """
    Parse the command line arguments
    :return: parsed arguments object (2 arguments)
    """

    parser = argparse.ArgumentParser(description='Enter configuration for generating data')
    parser.add_argument('--source', type=str, default='./trainingSet', help='Path to MNIST image root directory')
    parser.add_argument('--destination', type=str, default='./data', help='Path to save the generated dataset')
    parser.add_argument('--num_3', type=int, default=100, help='Number of unique instances for digit 3')
    parser.add_argument('--num_6', type=int, default=100, help='Number of unique instances for digit 6')
    parser.add_argument('--missing', type=float, default=25, choices=range(-1, 101),
                        help='Percentage of missing in range [0, 100]')
    parser.add_argument('--data_file_name', type=str, default='health_MNIST_data.csv',
                        help='File name of generated data')
    parser.add_argument('--data_masked_file_name', type=str, default='health_MNIST_data_masked.csv',
                        help='File name of generated masked data')
    parser.add_argument('--labels_file_name', type=str, default='health_MNIST_label.csv',
                        help='File name of generated labels')
    parser.add_argument('--mask_file_name', type=str, default='mask.csv',
                        help='File name of generated mask')
    parser.add_argument('--datatype_config', type=str, default='D1', choices=['D1', 'D2', 'D3', 'D4', 'D5'],
                        help='Data type configuration D1:All Real; D2:Real,Cat,Real,Real; D3: Real,Cat,Cat,'
                             'Real; D4:Real,Cat,Cat,Cat; D5:All categorical ')
    parser.add_argument('--seed', type=int, default=100, help='Seed for regenerating the same data for different data types')
    return vars(parser.parse_args())

def create_data_file(path, open_str):
    if os.path.exists(path):
        os.remove(path)
    return open(path, open_str)

def write_label_file_header(label_file):
    df = pd.DataFrame.from_dict({}, orient='index',
                                columns=['subject', 'digit', 'angle', 'disease',
                                         'disease_time', 'gender',
                                         'time_age', 'location'])
    df.to_csv(label_file, index=False)


def conversion_5_indx(row, indexes):
    img_h = row[indexes]
    img_h[img_h < 50] = 25
    img_h[np.where((img_h < 100) & (img_h >= 50))] = 75
    img_h[np.where((img_h < 150) & (img_h >= 100))] = 125
    img_h[np.where((img_h < 200) & (img_h >= 150))] = 175
    img_h[img_h >= 200] = 225
    row[indexes] = img_h
    return row

def save_data(data_file, mask_file, data_masked_file, label_file, rotated_MNIST, label_dict, missing_frac, observed_frac):

    # save rotated MNIST
    np.savetxt(data_file, rotated_MNIST, fmt='%d', delimiter=',')
    
    # generate mask
    mask = np.random.choice([0, 1], size=rotated_MNIST.shape, p=[missing_frac, observed_frac])
    
    # 0 implies missing, 1 implies observed
    masked_data = np.multiply(rotated_MNIST, mask)

    np.savetxt(data_masked_file, masked_data, fmt='%d', delimiter=',')
    np.savetxt(mask_file, mask, fmt='%d', delimiter=',')

    df = pd.DataFrame.from_dict(label_dict, orient='index',
                                columns=['subject', 'digit', 'angle', 'disease',
                                         'disease_time', 'gender',
                                         'time_age', 'location'])

    # save labels
    df.to_csv(label_file, index=False, header=False)


if __name__ == '__main__':
    opt = parse_arguments()
    for key in opt.keys():
        print('{:s}: {:s}'.format(key, str(opt[key])))
    locals().update(opt)

    digit_mod = {'3': num_3, '6': num_6}
    sick_prob = 0.5  # probability of instance being sick
    sample_index = 0
    subject_index = 0
    label_dict = {}
    gender = 0

    # 20 time points
    time_age = np.arange(0, 20)
    time_points = np.arange(-9, 11)

    # accumulate digits
    rotated_MNIST = np.empty((0, 1296))

    data_file = create_data_file(os.path.join(destination, data_file_name), "ab")
    mask_file = create_data_file(os.path.join(destination, mask_file_name), "ab")
    data_masked_file = create_data_file(os.path.join(destination, data_masked_file_name), "ab")
    label_file = create_data_file(os.path.join(destination, labels_file_name), "a")
    write_label_file_header(label_file) 

    missing_frac = missing/100
    observed_frac = 1 - missing_frac

    rng = np.array(range(0, 18))
    region_1 = rng
    for i in range(1, 18):
        region_1 = np.append(region_1, i * 36 + rng)
    rng = np.array(range(18, 36))
    region_2 = rng
    for i in range(1, 18):
        region_2 = np.append(region_2, i * 36 + rng)
    rng = np.array(range(0, 18))
    region_3 = rng + 648
    for i in range(19, 36):
        region_3 = np.append(region_3, i * 36 + rng)
    rng = np.array(range(18, 36))
    region_4 = rng + 648
    for i in range(19, 36):
        region_4 = np.append(region_4, i * 36 + rng)

    np.random.seed(seed)
    for digit in digit_mod.keys():
        print("Creating instances of digit {}".format(digit))

        # read in the files
        data_path = os.path.join(source, digit)
        files = glob.glob('{}/*.jpg'.format(data_path))

        # Assume requested files less than total available!
        for i in range(digit_mod[digit]):

            original_image = plt.imread(files[i])
            original_image_pad = np.pad(original_image, ((4, 4), (4, 4)), 'constant')

            # decide on sickness
            sick_var = np.random.binomial(1, sick_prob)

            # irrelevant location
            loc_var = np.random.binomial(1, 0.5)

            # introduce some noise
            rotations = np.random.normal(0, 2, len(time_points))

            # define rotation for each instance
            if sick_var:

                # simulate disease effect
                rotations += 45 * sigmoid(time_points)
            else:

                # baseline rotation for non-sick
                rotations += 5

            if digit == '3':
                gender = 0
            else:
                gender = 1

            for idx, rotation in enumerate(rotations):

                # rotate an instance
                img = ndimage.rotate(original_image_pad, angle=rotation, reshape=False)

                # diagonal shift the image
                img = ndimage.shift(img, shift=idx/10)

                if sick_var == 1:
                    label_dict[sample_index] =\
                        [subject_index, digit, rotation, sick_var, time_points[idx], gender, time_age[idx], loc_var]
                elif sick_var == 0:
                    label_dict[sample_index] = [subject_index, digit, rotation, sick_var, 'nan', gender,
                                                time_age[idx], loc_var]
                img = np.reshape(img, (1, 1296))
                if datatype_config != 'D1':
                    img = np.apply_along_axis(conversion_5_indx, 1, img, region_2)
                if datatype_config not in ['D1', 'D2']:
                    img = np.apply_along_axis(conversion_5_indx, 1, img, region_3)
                if datatype_config in ['D4', 'D5']:
                    img = np.apply_along_axis(conversion_5_indx, 1, img, region_4)
                if datatype_config == 'D5':
                    img = np.apply_along_axis(conversion_5_indx, 1, img, region_1)

                rotated_MNIST = np.append(rotated_MNIST, img, axis=0)

                sample_index += 1

            subject_index += 1

            if i%200 == 199:
                print("Instance no {} for digit {}".format(i+1, digit))

                save_data(data_file, mask_file, data_masked_file, label_file,
                          rotated_MNIST, label_dict, missing_frac, observed_frac)
                rotated_MNIST = np.empty((0, 1296))
                label_dict = {}
        
        save_data(data_file, mask_file, data_masked_file, label_file,
                  rotated_MNIST, label_dict, missing_frac, observed_frac)
        rotated_MNIST = np.empty((0, 1296))
        label_dict = {}

    print('Saved! Number of samples: {}'.format(sample_index))
