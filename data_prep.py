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

imgFileList = [x for x in toolkit_file.get_file_list(dataset_dir) if x.endswith('.jpg')]
# print(fileList)

dataset_dict_list = []

for file in imgFileList:
    pic_id = int(toolkit_file.get_basename(
        file, withExtension=False).replace('image_', ''))
    group_id = (pic_id - 1) // 80
    dataset_dict_list.append(
        {'pic_id': pic_id, 'group_id': group_id, 'image_path': file})

print('Processing...')
x_dataset = np.array([image_process.image_process(
    x['image_path']) for x in dataset_dict_list])
x_dataset = np_utils.normalize(x_dataset)
y_dataset = np_utils.to_categorical([x['group_id'] for x in dataset_dict_list])

dataset = []
for x in range(len(y_dataset)):
    label = y_dataset[x]
    img_data = x_dataset[x]
    dataset.append((img_data, label))
dataset = np.array(dataset)

np.save(data_dump, dataset)


if __name__ == '__main__':
    print(type(y_dataset))
    print(type(x_dataset))
    print(y_dataset.shape)
    print(x_dataset.shape)
    print(type(dataset))
    print(dataset.shape)
    print(dataset[0])