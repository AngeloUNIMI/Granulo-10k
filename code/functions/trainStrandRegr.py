import torch
import time
import copy
import numpy as np
from util import print_pers
import torch.nn as nn
import statistics
from util.losses import weighted_mse_loss
from util.initPredictionLabels import initPredictionLabels
from util.assignPredictions import assignPredictions
from util.getErrors import getErrors
from util.losses import full_decorr_loss
from util.warmup import pc_warmup_alpha

# training with validation
def trainStrandRegr(modelEnsemble, optimizer, scheduler,
                    num_epochs, dataset_sizes, dataloader_train, dataloader_val,
                    numBatches, batch_sizeP,
                    statisticsLabel,
                    iteration, fileResultNameFull, log, cuda, decoder, criterion, pc_warmup_epochs):

    # check if final already exists
    fileNameSaveFinal = 'modelsave_{0}_final.pt'.format(iteration+1)

    #init time
    since = time.time()

    # init best model
    best_models = {
        'encA': copy.deepcopy(modelEnsemble['encA'].state_dict()),
        'encB': copy.deepcopy(modelEnsemble['encB'].state_dict()),
        'encC': copy.deepcopy(modelEnsemble['encC'].state_dict()),
        'dec': copy.deepcopy(modelEnsemble['dec'].state_dict())
    }
    min_val_loss = 1e6

    # criterion = nn.MSELoss()

    # init
    predictionsAll, labelsAll = initPredictionLabels(dataset_sizes['val'])

    # loop on epochs
    for epoch in range(num_epochs):

        pc_alpha = pc_warmup_alpha(epoch, pc_warmup_epochs)
         
        # display
        if log:
            print_pers('\tEpoch {}/{}'.format(epoch+1, num_epochs), fileResultNameFull)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # Set model to training mode
                modelEnsemble['encA'].train()
                modelEnsemble['encB'].train()
                modelEnsemble['encC'].train()
                modelEnsemble['dec'].train()
            else:
                # Set model to eval mode
                modelEnsemble['encA'].eval()
                modelEnsemble['encB'].eval()
                modelEnsemble['encC'].eval()
                modelEnsemble['dec'].eval()

            # init losses and corrects
            running_loss = 0.0

            # choose dataloader
            if phase == 'train':
                dataloaders_chosen = dataloader_train
            if phase == 'val':
                dataloaders_chosen = dataloader_val

            ##################################################
            # Constant seed: always the same order of samples
            # (-> always the same samples, if stopped earlier)
            # torch.manual_seed(42)
            ##################################################

            # Iterate over data.
            for batch_num, batch in enumerate(dataloaders_chosen):
                # if phase == 'train':
                inputA, inputB, input_pc, pc_dims, _, _, _, _, label, labelNorm, weightVector = batch
                # else:
                #     inputA, inputB, _, _, _, _, label, labelNorm, weightVector = batch

                # get size of current batch
                sizeCurrentBatch = label.size(0)

                # if batch_num > 2:
                    # break

                if phase == 'val':
                    # indexes
                    indStart = batch_num * batch_sizeP
                    indEnd = indStart + sizeCurrentBatch

                # cuda
                if cuda:
                    inputA = inputA.to('cuda')
                    inputB = inputB.to('cuda')
                    input_pc = input_pc.to('cuda')
                    pc_dims = pc_dims.to('cuda')
                    # maskA = maskA.to('cuda')
                    # maskB = maskB.to('cuda')
                    label = label.to('cuda')
                    labelNorm = labelNorm.to('cuda')
                    weightVector = weightVector.to('cuda')

                # display
                # if batch_num % 100 == 0:
                    # print_pers("\t\tBatch n. {0} / {1}".format(batch_num, int(numBatches[phase])), fileResultNameFull)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with (torch.set_grad_enabled(phase == 'train')):
                    featuresA = modelEnsemble['encA'](inputA)
                    featuresB = modelEnsemble['encB'](inputB)
                    pc = input_pc.transpose(2, 1).contiguous()
                    featuresPC = modelEnsemble['encC'](pc, pc_dims)  
                    # features_maskA = modelEnsemble['encA_mask'](maskA)
                    # features_maskB = modelEnsemble['encB_mask'](maskB)
                    # featuresB = modelEnsemble['encA'](inputB)  # same model, less parameters?

                    # outputs = modelEnsemble['dec'](torch.cat((featuresA, featuresB), dim=1))
                    featuresAB_PC, _ = torch.max(torch.stack([featuresA, featuresB, featuresPC], dim=0), dim=0)
                    # featuresAB, _ = torch.max(torch.stack([featuresA, featuresB, features_maskA, features_maskB], dim=0), dim=0)
                    
                    if pc_alpha is not None:
                        featuresAB_PC = pc_alpha * featuresAB_PC
        
                    if decoder in ['gated', 'mmoe']:
                        outputs, decorr_features = modelEnsemble['dec'](featuresAB_PC)
                    else:
                        outputs = modelEnsemble['dec'](featuresAB_PC)

                    if cuda:
                        outputs = outputs.to('cuda')

                    # weights
                    """
                    weightVector = torch.zeros(sizeCurrentBatch, outputs.size()[1])
                    for num, infoSpessore in enumerate(goodForThickness):
                        if infoSpessore:
                            weightVector[num, :] = torch.tensor([0, 0, 1])
                        else:
                            weightVector[num, :] = torch.tensor([1, 1, 0])
                    """
    
                    # # loss MSE
                    # loss = weighted_mse_loss(outputs, labelNorm, weightVector)
                    # # print(f"MSE Loss: {lambda_mse* loss}")
                    # if decoder in ['gated', 'mmoe']:
                    #     decorr_features = [z.to("cuda") for z in decorr_features]
                    #     decorr_loss = full_decorr_loss([f.detach() for f in decorr_features])
                    #     # print(f"DECORR Loss: {lambda_decorr * decorr_loss}")
                    #     loss = lambda_mse * loss + lambda_decorr * decorr_loss
                    loss = criterion(outputs, labelNorm, weightVector)
                        
                    if phase == 'val':
                        # assign
                        predictionsAll, labelsAll = assignPredictions(predictionsAll, labelsAll,
                                                                      outputs, label,
                                                                      indStart, indEnd, statisticsLabel)

                    #print(loss)
                    #pause()

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                with torch.no_grad():
                    # running_loss += loss.item() * sizeCurrentBatch
                    running_loss += loss.detach() * sizeCurrentBatch

            # update schedulers
            if phase == 'train':
                for schedulerSingle in scheduler:
                    schedulerSingle.step()

            # compute epochs losses
            with torch.no_grad():
                # epoch_loss = running_loss / dataset_sizes[phase]
                epoch_loss = running_loss.item() / dataset_sizes[phase]

            # display
            if log:
                print_pers('\t\tRegr. {} loss: {:.2f}'.format(phase, epoch_loss), fileResultNameFull)
                if phase == 'val':
                    # errors
                    errorsAll = getErrors(predictionsAll, labelsAll, fileResultNameFull, log)

            # if greater val accuracy, deep copy the model
            if phase == 'val' and epoch_loss < min_val_loss:
                min_val_loss = epoch_loss
                best_model_wts = {
                    'encA': copy.deepcopy(modelEnsemble['encA'].state_dict()),
                    'encB': copy.deepcopy(modelEnsemble['encB'].state_dict()),
                    'encC': copy.deepcopy(modelEnsemble['encC'].state_dict()),
                    'decoder': copy.deepcopy(modelEnsemble['dec'].state_dict())
                }

        # save model at epoch
        # if epoch % 10 == 0:
            # fileNameSave = 'modelsave_{0}_epoch_{1}.pt'.format(iteration+1, epoch)
            #torch.save(model.state_dict(), os.path.join(dirResults, fileNameSave))

        # del
        # del inputs, label
        # torch.cuda.empty_cache()

    # time
    time_elapsed = time.time() - since
    print_pers('\tTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), fileResultNameFull)

    # load best model weights
    modelEnsemble['encA'].load_state_dict(best_model_wts['encA'])
    modelEnsemble['encB'].load_state_dict(best_model_wts['encB'])
    modelEnsemble['encC'].load_state_dict(best_model_wts['encC'])
    modelEnsemble['dec'].load_state_dict(best_model_wts['decoder'])
    # save final
    # torch.save(model.state_dict(), os.path.join(dirResults, fileNameSaveFinal))

    # del
    del inputA, inputB, label
    del outputs, loss
    del predictionsAll, labelsAll, errorsAll
    torch.cuda.empty_cache()

    return modelEnsemble
