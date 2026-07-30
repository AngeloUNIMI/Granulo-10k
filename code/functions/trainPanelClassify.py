import copy
import torch
import torch.nn as nn
from util import print_pers


# training with validation
def trainPanelClassify(modelEnsemble, optimizer,
                  dataloader_train, dataloader_val, num_epochs_warmup,
                  dataset_sizes, cuda, log, fileResultNameFull):

    criterion = nn.CrossEntropyLoss()
    sm = nn.LogSoftmax(dim=1)

    # init best model
    best_models = {
        'encA': copy.deepcopy(modelEnsemble['encA'].state_dict()),
        'encB': copy.deepcopy(modelEnsemble['encB'].state_dict()),
        'dec': copy.deepcopy(modelEnsemble['dec'].state_dict())
    }
    best_acc = -5.0
    min_val_loss = 1e6

    for epoch in range(0, num_epochs_warmup):

        # display
        if log:
            print_pers('\tEpoch {}/{}'.format(epoch+1, num_epochs_warmup), fileResultNameFull)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # Set model to training mode
                modelEnsemble['encA'].train()
                modelEnsemble['encB'].train()
                modelEnsemble['dec'].train()
            else:
                # Set model to eval mode
                modelEnsemble['encA'].eval()
                modelEnsemble['encB'].eval()
                modelEnsemble['dec'].eval()

            # init losses and corrects
            running_loss = 0.0
            running_corrects = 0.0

            # choose dataloader
            if phase == 'train':
                dataloaders_chosen = dataloader_train
            if phase == 'val':
                dataloaders_chosen = dataloader_val

            # Iterate over data.
            for batch_num, (inputA, inputB, _, _, numPanel, _, label, _, _) in enumerate(dataloaders_chosen):

                # get size of current batch
                sizeCurrentBatch = label.size(0)

                # if batch_num > 2:
                    # break

                # cuda
                if cuda:
                    inputA = inputA.to('cuda')
                    inputB = inputB.to('cuda')
                    numPanel = numPanel.to('cuda')
                numPanel.type(torch.int64)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with (torch.set_grad_enabled(phase == 'train')):

                    featuresA = modelEnsemble['encA'](inputA)
                    featuresB = modelEnsemble['encB'](inputB)
                    # featuresB = modelEnsemble['encA'](inputB)  # same model, less parameters?

                    # outputs = modelEnsemble['dec'](torch.cat((featuresA, featuresB), dim=1))
                    # outputs = modelEnsemble['dec']((featuresA + featuresB) / 2)  # avg of two tensors
                    featuresAB, _ = torch.max(torch.stack([featuresA, featuresB], dim=0), dim=0)
                    outputs = modelEnsemble['dec'](featuresAB)

                    if cuda:
                        outputs = outputs.to('cuda')

                    # loss
                    loss = criterion(outputs, numPanel)

                    # softmax
                    _, preds = torch.max(sm(outputs), 1)

                    # backward + optimize
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                with torch.no_grad():
                    # running_loss += loss.item() * sizeCurrentBatch
                    running_loss += loss.detach() * sizeCurrentBatch
                    running_corrects += torch.sum(preds == numPanel.data.int())

            # compute epochs losses
            with torch.no_grad():
                # epoch_loss = running_loss / dataset_sizes[phase]
                epoch_loss = running_loss.item() / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if log:
                print_pers('\t\tPanel class. {} loss: {:.2f}; Acc.: {:.2f}%'.format(phase, epoch_loss, epoch_acc*100),
                           fileResultNameFull)

            # if greater val accuracy, deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                min_val_loss = epoch_loss
                best_model_wts = {
                    'encA': copy.deepcopy(modelEnsemble['encA'].state_dict()),
                    'encB': copy.deepcopy(modelEnsemble['encB'].state_dict()),
                    'decoder': copy.deepcopy(modelEnsemble['dec'].state_dict())
                }
            if phase == 'val' and epoch_acc == best_acc:
                if epoch_loss < min_val_loss:
                    min_val_loss = epoch_loss
                    best_model_wts = {
                        'encA': copy.deepcopy(modelEnsemble['encA'].state_dict()),
                        'encB': copy.deepcopy(modelEnsemble['encB'].state_dict()),
                        'decoder': copy.deepcopy(modelEnsemble['dec'].state_dict())
                    }

    # load best model weights
    modelEnsemble['encA'].load_state_dict(best_model_wts['encA'])
    modelEnsemble['encB'].load_state_dict(best_model_wts['encB'])
    modelEnsemble['dec'].load_state_dict(best_model_wts['decoder'])

    # del
    del inputA, inputB, label
    del outputs, loss, preds
    torch.cuda.empty_cache()

    return modelEnsemble
