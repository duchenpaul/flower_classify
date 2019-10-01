import os
import toolkit_config

configDict = toolkit_config.read_config_general()['config']

# DATASET_DIR = r'E:\python_test\machine_learning\training_dataset'
# DATASET_DIR = os.path.join(DATASET_DIR, '17flowers', 'jpg')
DATASET_DIR = configDict['dataset_dir']

IMG_SIZE = 80

DATA_DMP = 'dataset.npy'
# MODEL_NAME = 'flower_classify.model'
MODEL_NAME = configDict['model_name']
