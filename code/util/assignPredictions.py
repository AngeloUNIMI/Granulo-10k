

def assignPredictions(predictionsAll, labelsAll,
                      outputs, label,
                      indStart, indEnd, statisticsLabel):

    # predictions
    predHeight = (outputs[:, 0] * statisticsLabel['stdHeight']) + statisticsLabel['meanHeight']
    predWidth = (outputs[:, 1] * statisticsLabel['stdWidth']) + statisticsLabel['meanWidth']
    predThickness = (outputs[:, 2] * statisticsLabel['stdThickness']) + statisticsLabel['meanThickness']

    predictionsAll['height'][indStart:indEnd] = predHeight.detach().clone().cpu().numpy()
    predictionsAll['width'][indStart:indEnd] = predWidth.detach().clone().cpu().numpy()
    predictionsAll['thickness'][indStart:indEnd] = predThickness.detach().clone().cpu().numpy()
    # label
    labelsAll['height'][indStart:indEnd] = label[:, 0].detach().clone().cpu().numpy()
    labelsAll['width'][indStart:indEnd] = label[:, 1].detach().clone().cpu().numpy()
    labelsAll['thickness'][indStart:indEnd] = label[:, 2].detach().clone().cpu().numpy()

    return predictionsAll, labelsAll
