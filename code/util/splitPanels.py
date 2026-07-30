import os
import shutil
import numpy as np
from functions.getIndexes import getIndexes

def get_split_dir(base_dir, seed, iteration, db_name):
    return os.path.join(
        base_dir,
        f"{db_name}_split_seed_{seed}_iter_{iteration}"
    )

def generate_or_load_split(
    dirDbTest, baseSplitDir, seed, iteration,
    num_iterations, xWiseKfold, imagePanelWise
):

    # compute split folder path
    split_dir = get_split_dir(baseSplitDir, seed, iteration, db_name=imagePanelWise)

    # if folder exists, reuse
    if os.path.exists(split_dir) and \
       os.path.exists(os.path.join(split_dir, "train")) and \
       os.path.exists(os.path.join(split_dir, "val")) and \
       os.path.exists(os.path.join(split_dir, "test")):
        print(f"[INFO] Using existing split: {split_dir}")
        return split_dir

    # otherwise generate it
    print(f"[INFO] Generating new split: {split_dir}")
    os.makedirs(split_dir, exist_ok=True)

    # get indices
    train_index, test_index = next(xWiseKfold)
    train_index_2, val_index = getIndexes(num_iterations, iteration, train_index, seed)
    del train_index

    # generate images
    if imagePanelWise == 'image':
        splitImages(dirDbTest, split_dir, train_index_2, val_index, test_index)
    elif imagePanelWise == 'panel':
        splitPanels(dirDbTest, split_dir, train_index_2, val_index, test_index)
    else:
        raise ValueError("Split not recognized!. Please check: it should be image or panel.")

    return split_dir

    
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
