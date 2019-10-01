import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
from keras.utils import np_utils
import numpy as np

import config
import toolkit_file

import image_process

from pprint import pprint

dataset_dir = config.DATASET_DIR
data_dump = config.DATA_DMP

def generate_file_list(dataset_dir):
    imgFileList = [x for x in toolkit_file.get_file_list(dataset_dir) if x.endswith('.jpg')]
    # print(fileList)

    dataset_dict_list = []

    for file in imgFileList:
        pic_id = int(toolkit_file.get_basename(
            file, withExtension=False).replace('image_', ''))
        group_id = (pic_id - 1) // 80
        dataset_dict_list.append(
            {'pic_id': pic_id, 'group_id': group_id, 'image_path': file})
    return dataset_dict_list


def read_img(image_path_list):
    x_dataset = np.array([image_process.image_process(x) for x in image_path_list])
    x_dataset = np_utils.normalize(x_dataset)
    return x_dataset


def dump_dataset(x_dataset, y_dataset):
    dataset = []
    for x in range(len(y_dataset)):
        img_data = x_dataset[x]
        label = y_dataset[x]
        dataset.append((img_data, label))
    dataset = np.array(dataset)
    np.random.shuffle(dataset)
    np.save(data_dump, dataset)


if __name__ == '__main__':
    print('Processing...')
    dataset_dict_list = generate_file_list(dataset_dir)
    print('Reading image...')
    x_dataset = read_img([x['image_path'] for x in dataset_dict_list])
    y_dataset = np_utils.to_categorical([x['group_id'] for x in dataset_dict_list])
    print('dumping numpy...')
    dump_dataset(x_dataset, y_dataset)
    
