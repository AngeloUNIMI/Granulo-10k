import torch
import torch.nn as nn
import numpy as np
from util import print_pers
import statistics
from util.losses import weighted_mse_loss
from util.initPredictionLabels import initPredictionLabels
from util.assignPredictions import assignPredictions
from util.getErrors import getErrors
from util.resultsLogger import ResultsLogger

def test(modelEnsemble, optimizer, dataloader,
         dataset_size, numBatches, batch_sizeP,
         statisticsLabel, fileResultNameFull, cuda, log, resultsLogger, decoder, criterion):

    
    #criterion = nn.MSELoss()
    # init losses and corrects
    running_loss = 0.0

    # init
    predictionsAll, labelsAll = initPredictionLabels(dataset_size)

    # Iterate over data.
    for batch_num, batch in enumerate(dataloader):
        
        inputA, inputB, input_pc, pc_dims, _, _, _, _, label, labelNorm, weightVector = batch
        # get size of current batch
        sizeCurrentBatch = label.size(0)

        # if batch_num > 2:
            # break

        # indexes
        indStart = batch_num * batch_sizeP
        indEnd = indStart + sizeCurrentBatch

        # cuda
        if cuda:
            inputA = inputA.to('cuda')
            inputB = inputB.to('cuda')
            input_pc = input_pc.to('cuda')
            pc_dims = pc_dims.to('cuda')
            label = label.to('cuda')
            labelNorm = labelNorm.to('cuda')
            weightVector = weightVector.to('cuda')

        # display
        if batch_num % 100 == 0:
            print_pers("\t\tBatch n. {0} / {1}".format(batch_num, int(numBatches)), fileResultNameFull)

        # indexes
        #indStart = batch_num * batch_sizeP
        #indEnd = indStart + sizeCurrentBatch

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with (torch.set_grad_enabled(False)):

            featuresA = modelEnsemble['encA'](inputA)
            # featuresB = modelEnsemble['encB'](inputB)
            featuresB = modelEnsemble['encA'](inputB)  # same model, less parameters?
            pc = input_pc.transpose(2, 1).contiguous()
            featuresPC = modelEnsemble['encC'](pc, pc_dims)

            # outputs = modelEnsemble['dec'](torch.cat((featuresA, featuresB), dim=1))
            featuresAB_PC, _ = torch.max(torch.stack([featuresA, featuresB, featuresPC], dim=0), dim=0)
            
            if decoder in ['gated', 'mmoe']:
                outputs, decorr_features = modelEnsemble['dec'](featuresAB_PC)
            else:
                outputs = modelEnsemble['dec'](featuresAB_PC)

            if cuda:
                outputs = outputs.to('cuda')
            # if cuda:
            #     outputs = outputs.to('cuda')
                # #Uncomment to consider the mean values
                # means = torch.Tensor([104.645, 15.65, 0.70105])
                # outputs = means.unsqueeze(1).repeat(1, outputs.size(0)).T.to('cuda')
                
                # # predictions
                # outputs[:, 0] = (outputs[:, 0] - statisticsLabel['meanHeight']) / statisticsLabel['stdHeight']
                # outputs[:, 1] = (outputs[:, 1] - statisticsLabel['meanWidth']) / statisticsLabel['stdWidth']
                # outputs[:, 2] = (outputs[:, 2] - statisticsLabel['meanThickness']) / statisticsLabel['stdThickness']
                
            #loss = weighted_mse_loss(outputs, labelNorm, weightVector)
            loss = criterion(outputs, labelNorm, weightVector)

            # predictions assign
            predictionsAll, labelsAll = assignPredictions(predictionsAll, labelsAll,
                                                          outputs, label,
                                                          indStart, indEnd, statisticsLabel)
            
            #print(loss)
            #pause()

        # statistics
        with torch.no_grad():
            # running_loss += loss.item() * sizeCurrentBatch
            running_loss += loss.detach() * sizeCurrentBatch

    # compute epochs losses
    with torch.no_grad():
        # epoch_loss = running_loss / dataset_sizes[phase]
        test_loss = running_loss.item() / dataset_size

    # display
    if log:
        print_pers('\t\tRegr. {} loss: {:.2f}'.format('test', test_loss), fileResultNameFull)
        # errors
        errorsAll, errorsAllPerc = getErrors(predictionsAll, labelsAll, fileResultNameFull, log)
        #add results of current iteration to the results logger
        resultsLogger.add(errorsAll['height'], errorsAllPerc['height'],
                    errorsAll['width'], errorsAllPerc['width'],
                    errorsAll['thickness'], errorsAllPerc['thickness'])
    
    # del
    del predictionsAll, labelsAll, errorsAll, errorsAllPerc
    torch.cuda.empty_cache()

    return test_loss
