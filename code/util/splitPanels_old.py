import os
import shutil
import numpy as np


def splitPanels(dirDbTest, dirOutTrainTest, train_index, val_index, test_index):
    print('Splitting...')
    for panNumber, (subdir) in enumerate(os.listdir(dirDbTest)):
        for filename in os.listdir(os.path.join(dirDbTest, subdir)):
            filenameMove = os.path.join(dirDbTest, subdir, filename)
            if panNumber in train_index:
                os.makedirs(os.path.join(dirOutTrainTest, 'train', subdir), exist_ok=True)
                shutil.copy(filenameMove, os.path.join(dirOutTrainTest, 'train', subdir, filename))
            if panNumber in val_index:
                os.makedirs(os.path.join(dirOutTrainTest, 'val', subdir), exist_ok=True)
                shutil.copy(filenameMove, os.path.join(dirOutTrainTest, 'val', subdir, filename))
            if panNumber in test_index:
                os.makedirs(os.path.join(dirOutTrainTest, 'test', subdir), exist_ok=True)
                shutil.copy(filenameMove, os.path.join(dirOutTrainTest, 'test', subdir, filename))


def splitImages(dirDbTest, dirOutTrainTest, train_index, val_index, test_index):
    print('Splitting...')
    for panNumber, (subdir) in enumerate(os.listdir(dirDbTest)):
        for filename in os.listdir(os.path.join(dirDbTest, subdir)):
            C = str.split(filename, '_')
            numImg = int(C[0]) * int(C[1])
            filenameMove = os.path.join(dirDbTest, subdir, filename)
            if numImg in train_index:
                os.makedirs(os.path.join(dirOutTrainTest, 'train', subdir), exist_ok=True)
                shutil.copy(filenameMove, os.path.join(dirOutTrainTest, 'train', subdir, filename))
            if numImg in val_index:
                os.makedirs(os.path.join(dirOutTrainTest, 'val', subdir), exist_ok=True)
                shutil.copy(filenameMove, os.path.join(dirOutTrainTest, 'val', subdir, filename))
            if numImg in test_index:
                os.makedirs(os.path.join(dirOutTrainTest, 'test', subdir), exist_ok=True)
                shutil.copy(filenameMove, os.path.join(dirOutTrainTest, 'test', subdir, filename))
