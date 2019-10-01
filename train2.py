import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.applications import VGG16

import numpy as np

import toolkit_file
import config

data_dump = config.DATA_DMP
model_name = config.MODEL_NAME
batch_size = 80
dropOutRate = 0.6


dataset = np.load(data_dump, allow_pickle=True)

X_dataset = np.array([x for x in dataset[:, 0]])
X_dataset = X_dataset.reshape(X_dataset.shape[0], X_dataset.shape[1], X_dataset.shape[2], -1)
Y_dataset = np.array([x for x in dataset[:, 1]])
num_classes = len(dataset[:, 1][0])

def buildModel(shape):
    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(shape[1], shape[2], 1))

    # Freeze all the layers except for the last layer: 
    for layer in vgg_conv.layers[:-4]:
        layer.trainable = False

    # Add the vgg convolutional base model
    model.add(vgg_conv)
     
    # Add new layers
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(17, activation='softmax'))
    model.summary() 

    return model


if __name__ == '__main__':
    toolkit_file.purge_folder('logs')
    shape = X_dataset.shape
    model = buildModel(shape)

    callback = EarlyStopping(monitor="loss", patience=30, verbose=1, mode="auto")
    tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
#                  batch_size=batch_size,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=True, # 是否可视化梯度直方图
                 write_images=True,# 是否可视化参数
                 embeddings_freq=0, 
                 embeddings_layer_names=None, 
                 embeddings_metadata=None)

    model.fit(X_dataset, Y_dataset, epochs=1000, shuffle=True, batch_size=batch_size, validation_split=0.1, callbacks=[callback, tbCallBack])
    model.save(model_name)
