import statistics
from util import print_pers


def getErrors(predictionsAll, labelsAll, fileResultNameFull, log):

    errorsAll = {}
    errorsAllPerc = {}

    # height
    predHeight_all = predictionsAll['height']
    labelHeight_all = labelsAll['height']
    # width
    predWidth_all = predictionsAll['width']
    labelWidth_all = labelsAll['width']
    # thickness
    predThickness_all = predictionsAll['thickness']
    labelThickness_all = labelsAll['thickness']

    errorsAll['height'] = statistics.mean(abs(predHeight_all - labelHeight_all))
    errorsAll['width'] = statistics.mean(abs(predWidth_all - labelWidth_all))
    errorsAll['thickness'] = statistics.mean(abs(predThickness_all - labelThickness_all))
    # perc
    errorsAllPerc['height'] = statistics.mean((abs(predHeight_all - labelHeight_all) / labelHeight_all))
    errorsAllPerc['width'] = statistics.mean((abs(predWidth_all - labelWidth_all) / labelWidth_all))
    errorsAllPerc['thickness'] = statistics.mean((abs(predThickness_all - labelThickness_all) / labelThickness_all))

    # display
    if log:
        print_pers('\t\tError (MAE). Height: {:.2f}; Width: {:.2f}; Thickness: {:.2f}'
                   .format(errorsAll['height'], errorsAll['width'], errorsAll['thickness']),
                   fileResultNameFull)
        print_pers('\t\tError (%). Height: {:.2f}%; Width: {:.2f}%; Thickness: {:.2f}%'
                   .format(errorsAllPerc['height'] * 100, errorsAllPerc['width'] * 100, errorsAllPerc['thickness'] * 100),
                   fileResultNameFull)
        
    return errorsAll, errorsAllPerc

