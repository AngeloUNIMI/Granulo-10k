import numpy as np


def initPredictionLabels(dataset_size):

    predictionsAll = {}
    labelsAll = {}

    # init
    predictionsAll['height'] = np.zeros((dataset_size))
    predictionsAll['width'] = np.zeros((dataset_size))
    predictionsAll['thickness'] = np.zeros((dataset_size))

    labelsAll['height'] = np.zeros((dataset_size))
    labelsAll['width'] = np.zeros((dataset_size))
    labelsAll['thickness'] = np.zeros((dataset_size))

    return predictionsAll, labelsAll
