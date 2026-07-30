# --------------------------
# IMPORT
import torch
import torchvision
from torchvision import models
from torchvision.transforms import v2
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import os
import copy
import shutil
import random
from random import seed
from datetime import datetime
import pickle
import PIL
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from PIL import Image
import csv
import gc
import sys

#----------------------------
import argparse
from pprint import pprint

# --------------------------
# PRIV FUNCTIONS
import util
import functions
from modelGeno.resnet_geno import resnet18_enc
from modelGeno.resnet_geno import resnet34_enc
from modelGeno.resnet_geno import resnet50_enc
from modelGeno.resnet_geno import resnet101_enc
from modelGeno.resnet_geno import resnet152_enc
from modelGeno.resnet_geno import resnext50_32x4d_enc
from modelGeno.resnet_geno import resnext101_32x8d_enc
from modelGeno.resnet_geno import resnext101_64x4d_enc
from modelGeno.resnet_geno import wide_resnet50_2_enc
from modelGeno.resnet_geno import wide_resnet101_2_enc

from modelGeno.resnet_geno import ResNet18_Weights 
from modelGeno.resnet_geno import ResNet34_Weights
from modelGeno.resnet_geno import ResNet50_Weights
from modelGeno.resnet_geno import ResNet101_Weights
from modelGeno.resnet_geno import ResNet152_Weights
from modelGeno.resnet_geno import ResNeXt50_32X4D_Weights
from modelGeno.resnet_geno import ResNeXt101_32X8D_Weights
from modelGeno.resnet_geno import ResNeXt101_64X4D_Weights
from modelGeno.resnet_geno import Wide_ResNet50_2_Weights
from modelGeno.resnet_geno import Wide_ResNet101_2_Weights

from modelGeno.resnet_geno import decoder_Geno_simple
from modelGeno.resnet_geno import decoder_Geno_ext
from modelGeno.resnet_geno import Gated_Decoder
from modelGeno.resnet_geno import MMoE_Decoder

from util.splitPanels import generate_or_load_split

from util.models import create_timm_encoder
from util.models import freeze_all_layers
from util.models import get_optimizer
from util.models import get_input_size

from modelGeno.pointnet_utils import PointNetPPWithProjection

# Uncertainty loss
from util.losses import UncertaintyMultiTaskLoss

# Results logger
from util.resultsLogger import ResultsLogger

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# --------------------------

def main(args):
    seed_adams = args.seed_adams
    plotta = args.plotta
    log = args.log
    num_iterations = args.num_iterations
    batch_size = args.batch_size
    batch_size_norm = args.batch_size_norm
    batch_size_test = args.batch_size_test
    numWorkersP = args.numWorkersP
    class_switch = args.class_switch
    num_epochs_class = args.num_epochs_class
    num_epochs_regr = args.num_epochs_regr
    trainModes = args.trainModes
    decoder = args.decoder
    base_lr = args.base_lr
    num_experts = args.num_experts
    exp_name = str(args.base_lr)

    pc_warmup_epochs = num_epochs_regr // 3
    
    # define a mapping from model name to feature size
    model_feature_map = {
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
        'resnet101': 2048,
        'resnet152': 2048,
        'resnext50_32x4d': 2048,
        'resnext101_32x8d': 2048,
        'resnext101_64x4d': 2048,
        'wide_resnet50_2': 2048,
        'wide_resnet101_2': 2048,
        # ---- TIMM Foundation Models ----
        'dino_vitb14': 768,
        'clip_vitl14': 1024,
        'eva02_clip_l14': 1024,
        'convnextv2_base': 1024,        
    }

    # build modelNamesAll from just names
    # if args.models is None:
    #     # default models
    #     modelNamesAll = [
    #         {'name': 'resnet18', 'sizeFeatures': model_feature_map['resnet18']},
    #         {'name': 'resnet34', 'sizeFeatures': model_feature_map['resnet34']},
    #     ]
    # else:
    modelNamesAll = []
    for name in args.models:
        if name not in model_feature_map:
            raise ValueError(f"Unknown model name: {name}")
        modelNamesAll.append({'name': name, 'sizeFeatures': model_feature_map[name]})
    
    print("Models list:")
    for m in modelNamesAll:
        print(f"  {m}") 
    print("==========================================\n")
    
    #saving log for results
    resultsLogger = ResultsLogger()

    print('STARTING MAIN()...\n\n')

    # Normalization params
    IMAGENET_MEAN=[0.485, 0.456, 0.406]
    IMAGENET_STD=[0.229, 0.224, 0.225]
                                
    # ------------------------------------------------------------------- db info
    # baseDir = 'D:/Workspace/'
    baseDir = '../../../data/CNN_OSB/'

    dirWorkspace = baseDir + 'DB Wood (test)/'
    dbName = 'DB_strand_IPAN3D_TII_buoni_e_sottili_JPG'
    numImgsDB = 4800  # pairs of A, B
    fileMisure = baseDir + 'DB Wood (orig)/DB_strand_IPAN3D_TII/Misure_buoni_e_sottili.csv'
    infoSpessore = baseDir + 'DB Wood (orig)/DB_strand_IPAN3D_TII/Strand_per_spessore_buoni.csv'
    sizeRegression = 3  # length, width, thickness

    # process measures
    allMisure, allPanNumbers, columnNames = util.getAllClassesVec(fileMisure)
    
    # Mean, std values
    # allMisureNp = np.array(allMisure)
    # np.mean(allMisureNp, axis=0) # [104.645, 15.65, 0.70105]
    # np.std(allMisureNp, axis=0) # [20.61501819,  7.829272  ,  0.2331789 ]
        
    allMisureNormalized, statisticsLabel = util.normalizeMeasures(allMisure)
    fileNameSpessore = util.getInfoSpessore(infoSpessore)

    # ------------------------------------------------------------------- Enable CUDA
    cuda = True if torch.cuda.is_available() else False
    # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    Tensor = torch.cuda.DoubleTensor if cuda else torch.cuda.DoubleTensor
    if cuda:
        torch.cuda.empty_cache()
    print("Cuda is {0}".format(cuda))
    #util.pause()

    # ------------------------------------------------------------------- dirs
    dirDbOrig = dirWorkspace + dbName + '/'
    dirDbTest = dirWorkspace + dbName + '/datastore_A/'
    dirOutTrainTest = dirWorkspace + dbName + '/datastore_train_test/'
    os.makedirs(dirDbTest, exist_ok=True)
    os.makedirs(dirOutTrainTest, exist_ok=True)

    # mapping model names to encoders and weight classes
    model_map = {
        'resnet18': (resnet18_enc, ResNet18_Weights),
        'resnet34': (resnet34_enc, ResNet34_Weights),
        'resnet50': (resnet50_enc, ResNet50_Weights),
        'resnet101': (resnet101_enc, ResNet101_Weights),
        'resnet152': (resnet152_enc, ResNet152_Weights),
        'resnext50_32x4d': (resnext50_32x4d_enc, ResNeXt50_32X4D_Weights),
        'resnext101_32x8d': (resnext101_32x8d_enc, ResNeXt101_32X8D_Weights),
        'resnext101_64x4d': (resnext101_64x4d_enc, ResNeXt101_64X4D_Weights),
        'wide_resnet50_2': (wide_resnet50_2_enc, Wide_ResNet50_2_Weights),
        'wide_resnet101_2': (wide_resnet101_2_enc, Wide_ResNet101_2_Weights),
        # ---- foundation models ----
        # 1. DINOv2 ViT-B/14
        'dino_vitb14': (
            lambda pretrained: create_timm_encoder("vit_base_patch14_dinov2.lvd142m", pretrained),
            None
        ),
        # 2. CLIP ViT-L/14 (LAION)
        'clip_vitl14': (
            lambda pretrained: create_timm_encoder("vit_large_patch14_clip_224.laion2b_ft_in1k", pretrained),
            None
        ),
        # 3. EVA02-CLIP Large/14
        'eva02_clip_l14': (
            lambda pretrained: create_timm_encoder("eva_large_patch14_336.in22k_ft_in1k", pretrained),
            None
        ),
        # 4. ConvNeXt V2 Base
        'convnextv2_base': (
            lambda pretrained: create_timm_encoder("convnextv2_base.fcmae_ft_in22k_in1k_384", pretrained),
            None
        ),
    }

    for trainMode in trainModes:

        # ------------------------------------------------------------------- loop on models
        for i, (modelData) in enumerate(modelNamesAll):

            # dir results
            dirResult = './results/' + dbName + '/' + exp_name + '/' + trainMode + '/' + modelData['name'] + '/'
            os.makedirs(dirResult, exist_ok=True)

            # result file
            now = datetime.now()
            current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
            fileResultName = current_time + '.txt'
            fileResultNameFull = os.path.join(dirResult, fileResultName)
            fileResult = open(fileResultNameFull, "x")
            fileResult.close()

            if log:
                print()
                print(statisticsLabel)
                util.print_pers('Std height/Mean height (%): {:.2f}'.format(statisticsLabel['stdHeight']/statisticsLabel['meanHeight']*100), fileResultNameFull)
                util.print_pers('Std width/Mean width (%): {:.2f}'.format(statisticsLabel['stdWidth']/statisticsLabel['meanWidth']*100), fileResultNameFull)
                util.print_pers('Std thickness/Mean thickness (%): {:.2f}'.format(statisticsLabel['stdThickness']/statisticsLabel['meanThickness']*100), fileResultNameFull)
                util.print_pers("", fileResultNameFull)

            # display
            if log:
                util.print_pers("Train mode: {0}".format(trainMode), fileResultNameFull)
                util.print_pers("Model: {0}".format(modelData['name']), fileResultNameFull)

            # get crossvalind
            numOfPanels = util.getNumOfPanels(dirDbTest)
            # create kfold objects
            kf_train_test_imageWise = KFold(n_splits=num_iterations, shuffle=True, random_state=seed_adams)
            kf_train_test_panelWise = KFold(n_splits=num_iterations, shuffle=True, random_state=seed_adams)
            # indexes depend on panel wise split or not
            all_indexes_imageWise = np.arange(numImgsDB)
            all_indexes_panelWise = np.arange(numOfPanels)
            # create enumeratable objects
            imageWiseKfold = kf_train_test_imageWise.split(all_indexes_imageWise)
            panelWiseKfold = kf_train_test_panelWise.split(all_indexes_panelWise)

            # ------------------------------------------------------------------- loop on iterations
            # init
            dataset_sizes = {}
            lossAll = []
            modelEnsemble = {}
            for r in range(0, num_iterations): #(nFolds-1)
            # for r, (train_index, test_index) in enumerate(kf_train_test.split(all_indexes)):
                # print(r)

                # ---------------------------------------------------------------------------------
                image_split_dir = generate_or_load_split(
                        dirDbTest=dirDbTest,
                        baseSplitDir=dirOutTrainTest,  # parent folder with all splits
                        seed=seed_adams,
                        iteration=r,
                        num_iterations=num_iterations,
                        xWiseKfold=imageWiseKfold,
                        imagePanelWise='image'
                    )
                                    
                # # clean dirs
                # shutil.rmtree(dirOutTrainTest)
                # os.makedirs(dirOutTrainTest, exist_ok=True)
                # # splitfolders.ratio(dirDbTest, output=dirOutTrainTest, seed=seed_adams+r, ratio=(.7, .1, .2))
                # # SPLIT TRAIN-VAL-TEST
                # train_index, test_index = next(imageWiseKfold)
                # train_index_2, val_index = functions.getIndexes(num_iterations, r, train_index)
                # del train_index
                # # normal kfold
                # util.splitImages(dirDbTest, dirOutTrainTest, train_index_2, val_index, test_index)
                # ---------------------------------------------------------------------------------

                # display
                if log:
                    util.print_pers("", fileResultNameFull)
                    util.print_pers("Iteration n. {0}".format(r + 1), fileResultNameFull)
                    print()

                # ------------------------------------------------------------
                # Load encoder A (torchvision or timm)
                # ------------------------------------------------------------
                enc_constructor, weights_class = model_map[modelData['name']]
                model_name = modelData['name']
                is_foundation_or_wide = any(k in model_name for k in ["vit", "clip", "eva", "convnext", "wide_resnet", "resnext"])

                # Check if this is a foundation model (= timm entry)
                is_timm_model = (weights_class is None)

                # ------------------------------------------------------------
                # Case 1: TIMM foundation models (DINO, CLIP, EVA, ConvNeXtV2)
                # ------------------------------------------------------------
                if is_timm_model:

                    if trainMode == 'scratch':
                        raise ValueError(
                            f"Training foundation model '{model_name}' from scratch is not allowed. "
                            f"Use trainMode='imagenet'." # Foundations models must be used pretrained!
                        )

                    elif trainMode == 'imagenet':
                        modelEnsemble['encA'] = enc_constructor(pretrained=True)

                    else:
                        raise ValueError(f"Unknown trainMode: {trainMode}")

                # ------------------------------------------------------------
                # Case 2: Torchvision models
                # ------------------------------------------------------------
                else:

                    if trainMode == 'scratch':
                        modelEnsemble['encA'] = enc_constructor(weights=None)

                    elif trainMode == 'imagenet':
                        modelEnsemble['encA'] = enc_constructor(weights=weights_class.DEFAULT)

                    else:
                        raise ValueError(f"Unknown trainMode: {trainMode}")

                modelEnsemble['encB'] = copy.deepcopy(modelEnsemble['encA'])
                # modelEnsemble['encA_mask'] = copy.deepcopy(modelEnsemble['encA'])
                # modelEnsemble['encB_mask'] = copy.deepcopy(modelEnsemble['encA'])
                # decoder
                # modelEnsemble['dec'] = nn.Linear(modelData['sizeFeatures']*2, numOfPanels)
                # modelEnsemble['dec'] = nn.Linear(modelData['sizeFeatures'], numOfPanels)
                modelEnsemble['dec'] = decoder_Geno_simple(modelData['sizeFeatures'], numOfPanels)

                # # block parameters
                # for param in modelEnsemble['encA'].parameters():
                #     param.requires_grad = True
                # for param in modelEnsemble['encB'].parameters():
                #     param.requires_grad = True
                # for param in modelEnsemble['dec'].parameters():
                #     param.requires_grad = True

                # ------------------------------------------------------------
                # ALWAYS train decoder
                # ------------------------------------------------------------
                for p in modelEnsemble['dec'].parameters():
                    p.requires_grad = True

                # ------------------------------------------------------------
                # CASE 1 - Torchvision models -> fine-tune ALL layers
                # ------------------------------------------------------------
                if not is_foundation_or_wide:
                    for p in modelEnsemble['encA'].parameters():
                        p.requires_grad = True
                    for p in modelEnsemble['encB'].parameters():
                        p.requires_grad = True

                # ------------------------------------------------------------
                # CASE 2 - TIMM Foundation models -> fine-tune ONLY final layers
                # ------------------------------------------------------------
                else:
                    # 1) fully freeze both encoders
                    freeze_all_layers(modelEnsemble['encA'])
                    freeze_all_layers(modelEnsemble['encB'])

                    # # 2) pick which layers to unfreeze depending on architecture
                    # if "vit" in model_name or "clip" in model_name or "eva" in model_name:
                    #     # Vision Transformers (DINOv2, CLIP ViT-L/14, EVA02-CLIP)
                    #     final_layers = [
                    #         "blocks.11",     # last transformer block
                    #         "block.11",      # some models use 'block' not 'blocks'
                    #         "norm",          # final norm
                    #         "fc_norm",       # DINOv2 uses fc_norm, not norm
                    #         "ln"             # fallback for LayerNorm naming
                    #     ]

                    # elif "convnext" in model_name:
                    #     # ConvNeXt V2 — final two stages 
                    #     final_layers = [
                    #         "stages.2",
                    #         "stages.3",
                    #         "norm"
                    #     ]

                    # else:
                    #     raise ValueError(f"No fine-tuning rule for foundation model {model_name}")

                    # # 3) unfreeze final layers of encoder A & B
                    # unfreeze_last_layers(modelEnsemble['encA'], final_layers)
                    # unfreeze_last_layers(modelEnsemble['encB'], final_layers)

                modelEnsemble['encC'] = PointNetPPWithProjection(proj_dim=modelData['sizeFeatures'])

                # Freeze backbone
                for p in modelEnsemble['encC'].parameters():
                    p.requires_grad = True

                # # Ensure projection head is trainable
                # for p in modelEnsemble['encC'].projector.parameters():
                #     p.requires_grad = True

                # sizeimg
                #sizeImgCNN = (320, 240)  #1/4 of 1280,960
                sizeImgCNN = get_input_size(model_name)  

                #currentModel.double()
                # cuda
                if cuda:
                    device = 'cuda'
                    
                    # move each model to GPU
                    modelEnsemble['encA'] = modelEnsemble['encA'].to(device)
                    modelEnsemble['encB'] = modelEnsemble['encB'].to(device)
                    modelEnsemble['encC'] = modelEnsemble['encC'].to(device)
                    # modelEnsemble['encA_mask'] = modelEnsemble['encA_mask'].to(device)
                    # modelEnsemble['encB_mask'] = modelEnsemble['encB_mask'].to(device)
                    modelEnsemble['dec'] = modelEnsemble['dec'].to(device)
                    
                    # wrap with DataParallel if multiple GPUs
                    if torch.cuda.device_count() > 1:
                        print(f"Using {torch.cuda.device_count()} GPUs for modelEnsemble")
                        modelEnsemble['encA'] = nn.DataParallel(modelEnsemble['encA'])
                        modelEnsemble['encB'] = nn.DataParallel(modelEnsemble['encB'])
                        modelEnsemble['encC'] = nn.DataParallel(modelEnsemble['encC'])
                        # modelEnsemble['encA_mask'] = nn.DataParallel(modelEnsemble['encA_mask'])
                        # modelEnsemble['encB_mask'] = nn.DataParallel(modelEnsemble['encB_mask'])
                        modelEnsemble['dec'] = nn.DataParallel(modelEnsemble['dec'])

                # preprocess
                transform = {
                    'train':
                        v2.Compose([
                            v2.Resize(size=sizeImgCNN, interpolation=v2.InterpolationMode.BILINEAR),
                            # v2.RandomHorizontalFlip(),
                            # v2.RandomVerticalFlip(),
                            # v2.RandomRotation(degrees=45,
                                              # interpolation=v2.InterpolationMode.BILINEAR),
                            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])]),
                    'val':
                        v2.Compose([
                            v2.Resize(size=sizeImgCNN, interpolation=v2.InterpolationMode.BILINEAR),
                            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])]),
                }

                # ------------------------------------------------------------------- TRAIN
                # load data
                # train
                # db_train = datasets.ImageFolder(os.path.join(dirOutTrainTest, 'train'), transform['train'])
                db_train = datasets.ImageFolder(os.path.join(image_split_dir, 'train'), transform['train'])
                db_train_loader = torch.utils.data.DataLoader(db_train,
                                                                    batch_size=batch_size_norm, shuffle=True,  # true
                                                                    num_workers=numWorkersP, pin_memory=True)
                dataset_sizes['train'] = len(db_train)
                util.print_pers("\tDimensione dataset train: {0}".format(dataset_sizes['train']), fileResultNameFull)
                # val
                # db_val = datasets.ImageFolder(os.path.join(dirOutTrainTest, 'val'), transform['val'])
                db_val   = datasets.ImageFolder(os.path.join(image_split_dir, 'val'),   transform['val'])                
                db_val_loader = torch.utils.data.DataLoader(db_val,
                                                              batch_size=batch_size_norm, shuffle=True,
                                                              num_workers=numWorkersP, pin_memory=True)
                dataset_sizes['val'] = len(db_val)
                util.print_pers("\tDimensione dataset val: {0}".format(dataset_sizes['val']), fileResultNameFull)
                print()

                # mean, std
                print("Normalization...")
                meanNorm = {}
                stdNorm = {}
                dataloaders_all = list()
                dataloaders_all.append(db_train_loader)
                # dataset_sizes_all = dataset_sizes['train']
                dataloaders_all.append(db_val_loader)
                dataset_sizes_all = dataset_sizes['train'] + dataset_sizes['val']

                # compute norm for all channels together
                
                stats_path = os.path.join(image_split_dir, "mean_std.npz")
                meanNorm, stdNorm = util.computeMeanStd(dataloaders_all, dataset_sizes_all, batch_size_norm, cuda, stats_path=stats_path)

                # update datasets
                # # train
                # db_train = util.ImageFolderWithSize(dirDbOrig, os.path.join(image_split_dir, 'train'),
                #                                     transform['train'],
                #                                     misure=allMisure, misureNorm=allMisureNormalized,
                #                                     panNumbers=allPanNumbers, infoSpessore=fileNameSpessore,
                #                                     meanNorm=meanNorm, stdNorm=stdNorm)
                # # val
                # db_val = util.ImageFolderWithSize(dirDbOrig, os.path.join(image_split_dir, 'val'),
                #                                     transform['val'],
                #                                     misure=allMisure, misureNorm=allMisureNormalized,
                #                                     panNumbers=allPanNumbers, infoSpessore=fileNameSpessore,
                #                                     meanNorm=meanNorm, stdNorm=stdNorm)
                # train
                db_train = util.ImageFolderWithSizeDA(dirDbOrig,
                                    os.path.join(image_split_dir, 'train'), transform['train'],
                                    misure=allMisure, misureNorm=allMisureNormalized,
                                    panNumbers=allPanNumbers, infoSpessore=fileNameSpessore,
                                    calib_path='./util/calib_opencv_simple.yml', two_view_augmentation=True,
                                    device='cpu')
                # val
                # db_val = util.ImageFolderWithSize(dirDbOrig, os.path.join(image_split_dir, 'val'),
                #                                     transform['val'],
                #                                     misure=allMisure, misureNorm=allMisureNormalized,
                #                                     panNumbers=allPanNumbers, infoSpessore=fileNameSpessore,
                #                                     meanNorm=meanNorm, stdNorm=stdNorm)  
                db_val = util.ImageFolderWithSizeDA(dirDbOrig,
                                    os.path.join(image_split_dir, 'val'), transform['val'],
                                    misure=allMisure, misureNorm=allMisureNormalized,
                                    panNumbers=allPanNumbers, infoSpessore=fileNameSpessore,
                                    calib_path='./util/calib_opencv_simple.yml', two_view_augmentation=True,
                                    device='cpu')
                  
                # update transforms
                # train
                transform['train'] = v2.Compose([
                    v2.Resize(size=sizeImgCNN, interpolation=v2.InterpolationMode.BILINEAR),
                    # v2.RandomHorizontalFlip(),
                    # v2.RandomVerticalFlip(),
                    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                    v2.Normalize(
                        mean=[meanNorm, meanNorm, meanNorm],
                        std=[stdNorm, stdNorm, stdNorm]),
                ])
                # val
                transform['val'] = v2.Compose([
                    v2.Resize(size=sizeImgCNN, interpolation=v2.InterpolationMode.BILINEAR),
                    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                    v2.Normalize(
                        mean=[meanNorm, meanNorm, meanNorm],
                        std=[stdNorm, stdNorm, stdNorm]),
                ])
                print()

                # update data loaders
                # train
                db_train_loader = torch.utils.data.DataLoader(db_train,
                                                                    batch_size=batch_size, shuffle=True,  # true
                                                                    num_workers=numWorkersP, pin_memory=True)
                # val
                db_val_loader = torch.utils.data.DataLoader(db_val,
                                                                  batch_size=batch_size, shuffle=True,
                                                                  num_workers=numWorkersP, pin_memory=True)
                # compute num batches
                numBatches = {}
                numBatches['train'] = np.round(dataset_sizes['train'] / batch_size)
                numBatches['val'] = np.round(dataset_sizes['val'] / batch_size)

                # optim
                # optimizer_ft = optim.SGD([
                #     {'params': modelEnsemble['encA'].parameters()},
                #     {'params': modelEnsemble['encB'].parameters()},
                #     {'params': modelEnsemble['dec'].parameters()}
                # ], lr=lr_class, momentum=0.9, weight_decay=0.0005)

                optimizer_ft = get_optimizer(
                    model_name=modelData["name"],
                    modelEnsemble=modelEnsemble,
                    phase="classification",      
                    base_lr=base_lr,
                    criterion=None
                )

                """
                gg = iter(all_idb2_train_loader)
                inputs_t, _ = next(gg)
                util.visImage(inputs_t[0], [])
                """

                # warm-up
                if class_switch:
                    util.print_pers("Training - panel classification", fileResultNameFull)
                    # no warmup if training from scratch
                    modelEnsemble = functions.trainPanelClassify(modelEnsemble, optimizer_ft,
                                                          db_train_loader, db_val_loader, num_epochs_class,
                                                          dataset_sizes, cuda, log, fileResultNameFull)
                    print()


                # ------------------------------------------------------------------------------------------------------
                # change to regression
                if decoder == 'gated':
                    modelEnsemble['dec'] = Gated_Decoder(in_features=modelData['sizeFeatures'], 
                                                        num_classes=sizeRegression)
                elif decoder == 'mmoe':
                    modelEnsemble['dec'] = MMoE_Decoder(in_features=modelData['sizeFeatures'],
                                                        num_tasks=sizeRegression,
                                                        num_experts=num_experts,
                                                        expert_hidden=256,
                                                        tower_hidden=64
                    )

                else:
                    modelEnsemble['dec'] = decoder_Geno_ext(modelData['sizeFeatures'], 
                                                            2048, 
                                                            sizeRegression)
                    
                print(f'Using {decoder} decoder...')                    
                
                if torch.cuda.is_available():
                    device = 'cuda'
                    # move models to GPU
                    modelEnsemble['dec'] = modelEnsemble['dec'].to(device)
    
                    # check for multiple GPUs and wrap with DataParallel
                    if torch.cuda.device_count() > 1:
                        print(f"Using {torch.cuda.device_count()} GPUs")
                        modelEnsemble['dec'] = nn.DataParallel(modelEnsemble['dec'])
     
        
                # clean dirs
                # shutil.rmtree(dirOutTrainTest)
                # os.makedirs(dirOutTrainTest, exist_ok=True)
                # change split -> panelwise splitting
                # train_index, test_index = next(panelWiseKfold)
                # train_index_2, val_index = functions.getIndexes(num_iterations, r, train_index)
                # del train_index
                # # panelwise split
                # util.splitPanels(dirDbTest, dirOutTrainTest, train_index_2, val_index, test_index)
                # print()
                                # ---------------------------------------------------------------------------------
                panel_split_dir = generate_or_load_split(
                        dirDbTest=dirDbTest,
                        baseSplitDir=dirOutTrainTest,  # parent folder with all splits
                        seed=seed_adams,
                        iteration=r,
                        num_iterations=num_iterations,
                        xWiseKfold=panelWiseKfold,
                        imagePanelWise='panel'
                    )
                

                # clean memory
                del db_train, db_train_loader, db_val, db_val_loader
                with torch.no_grad():
                    torch.cuda.empty_cache()
                gc.collect()
                # update datasets
                # train
                # db_train = util.ImageFolderWithSizeDA(dirDbOrig,
                #                     os.path.join(image_split_dir, 'train'), transform['train'],
                #                     misure=allMisure, misureNorm=allMisureNormalized,
                #                     panNumbers=allPanNumbers, infoSpessore=fileNameSpessore,
                #                     calib_path='./util/calib_opencv_simple.yml', two_view_augmentation=True,
                #                     device='cpu')

                db_train = util.ImageFolderWithSizePC(dirDbOrig,
                                                    os.path.join(image_split_dir, 'train'), transform['train'],
                                                    misure=allMisure, misureNorm=allMisureNormalized,
                                                    panNumbers=allPanNumbers, infoSpessore=fileNameSpessore,
                                                    calib_path='./util/calib_opencv_simple.yml', two_view_augmentation=True,
                                                    device='cpu', augment_pc=True)


                db_train_loader = torch.utils.data.DataLoader(db_train,
                                                              batch_size=batch_size, shuffle=True,  # true
                                                              num_workers=numWorkersP, pin_memory=True)
                # val
                # db_val = util.ImageFolderWithSize(dirDbOrig, os.path.join(panel_split_dir, 'val'),
                #                                   transform['val'],
                #                                   misure=allMisure, misureNorm=allMisureNormalized,
                #                                   panNumbers=allPanNumbers, infoSpessore=fileNameSpessore,
                #                                   meanNorm=meanNorm, stdNorm=stdNorm)
                db_val = util.ImageFolderWithSizePC(dirDbOrig,
                                                    os.path.join(image_split_dir, 'val'), transform['val'],
                                                    misure=allMisure, misureNorm=allMisureNormalized,
                                                    panNumbers=allPanNumbers, infoSpessore=fileNameSpessore,
                                                    calib_path='./util/calib_opencv_simple.yml', two_view_augmentation=True,
                                                    device='cpu', augment_pc=True)                
                db_val_loader = torch.utils.data.DataLoader(db_val,
                                                            batch_size=batch_size, shuffle=True,
                                                            num_workers=numWorkersP, pin_memory=True)
                # sizes
                dataset_sizes['train'] = len(db_train)
                dataset_sizes['val'] = len(db_val)

                # optim
                # optimizer_ft = optim.SGD([
                #     {'params': modelEnsemble['encA'].parameters()},
                #     {'params': modelEnsemble['encB'].parameters()},
                #     # {'params': modelEnsemble['encA_mask'].parameters()},
                #     # {'params': modelEnsemble['encB_mask'].parameters()},
                #     {'params': modelEnsemble['dec'].parameters()}
                # ], lr=lr_regr, momentum=0.9, weight_decay=0.0005)
                criterion = UncertaintyMultiTaskLoss(num_tasks=sizeRegression)
                optimizer_ft = get_optimizer(
                    model_name=modelData["name"],
                    modelEnsemble=modelEnsemble,
                    phase="regression",
                    criterion=criterion
                )                
                exp_lr_scheduler = list()

                # train - regression
                util.print_pers("Training - strand regression", fileResultNameFull)
                util.print_pers("\tDimensione dataset train: {0}".format(dataset_sizes['train']), fileResultNameFull)
                # train net
                currentModel = functions.trainStrandRegr(modelEnsemble, optimizer_ft, exp_lr_scheduler,
                                                         num_epochs_regr, dataset_sizes, db_train_loader, db_val_loader,
                                                         numBatches, batch_size,
                                                         statisticsLabel,
                                                         r, fileResultNameFull, log, cuda, decoder, criterion, pc_warmup_epochs)

                # visualize some outputs
                #functions.visualize_model(currentModel, all_idb2_val_loader, cuda, columnNames, num_images=6)
                #util.pause()
                print()

                # ------------------------------------------------------------------- TEST
                # torch.cuda.empty_cache()
                # NEW: based on majority voting

                # display
                if log:
                    util.print_pers("Testing", fileResultNameFull)

                # zero the parameter gradients
                optimizer_ft.zero_grad()
                torch.no_grad()

                # test transform
                transform['test'] = v2.Compose([
                    v2.Resize(size=sizeImgCNN, interpolation=v2.InterpolationMode.BILINEAR),
                    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                    v2.Normalize(
                        mean=[meanNorm, meanNorm, meanNorm],
                        std=[stdNorm, stdNorm, stdNorm]),
                ])
                # test
                # db_test = util.ImageFolderWithSize(dirDbOrig,
                #                                    os.path.join(panel_split_dir, 'test'), transform['test'],
                #                                    misure=allMisure, misureNorm=allMisureNormalized,
                #                                    panNumbers=allPanNumbers, infoSpessore=fileNameSpessore,
                #                                    meanNorm=meanNorm, stdNorm=stdNorm)
                db_test = util.ImageFolderWithSizePC(dirDbOrig,
                                                    os.path.join(image_split_dir, 'test'), transform['test'],
                                                    misure=allMisure, misureNorm=allMisureNormalized,
                                                    panNumbers=allPanNumbers, infoSpessore=fileNameSpessore,
                                                    calib_path='./util/calib_opencv_simple.yml', two_view_augmentation=True,
                                                    device='cpu', augment_pc=True)                   
                db_test_loader = torch.utils.data.DataLoader(db_test,
                                                              batch_size=batch_size_test, shuffle=True,
                                                              num_workers=numWorkersP, pin_memory=True)
                dataset_sizes['test'] = len(db_test)
                numBatches['test'] = np.round(dataset_sizes['test'] / batch_size_test)

                test_loss = functions.test(currentModel, optimizer_ft, db_test_loader,
                                           dataset_sizes['test'], numBatches['test'], batch_size_test,
                                           statisticsLabel, fileResultNameFull, cuda, log, resultsLogger, decoder, criterion)
                lossAll.append(test_loss)

                # newline
                util.print_pers("", fileResultNameFull)

                # save iter
                fileSaveIter = open(os.path.join(dirResult, 'results_{0}.dat'.format(r + 1)), 'wb')
                pickle.dump([test_loss], fileSaveIter)
                fileSaveIter.close()

            # end loop on iterations

            # average accuracy not quant
            meanLoss = np.mean(lossAll)
            stdLoss = np.std(lossAll)

            # display
            # not quant
            util.print_pers("", fileResultNameFull)
            util.print_pers("Mean loss over {0} iterations (%); {1:.2f}".format(num_iterations, meanLoss), fileResultNameFull)
            util.print_pers("Std loss over {0} iterations (%); {1:.2f}".format(num_iterations, stdLoss), fileResultNameFull)

            #close
            fileResult.close()

            # save
            fileSaveFinal = open(os.path.join(dirResult, 'resultsFinal.dat'), 'wb')
            pickle.dump([meanLoss, stdLoss], fileSaveFinal)
            fileSaveFinal.close()

        if log:
            # Save results
            resultsLogger.save_csv(os.path.join(dirResult, 'results.txt'))
            # del
            torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training configuration")

    # random seed
    parser.add_argument('--seed_adams', type=int, default=42)

    # general flags (plot/log)
    parser.add_argument('--plotta', action='store_true', help='Enable plotting')
    parser.add_argument('--no-plotta', dest='plotta', action='store_false', help='Disable plotting')
    parser.set_defaults(plotta=False)

    parser.add_argument('--log', action='store_true', help='Enable logging')
    parser.add_argument('--no-log', dest='log', action='store_false', help='Disable logging')
    parser.set_defaults(log=True)
       
    # training parameters
    parser.add_argument('--num_iterations', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--batch_size_norm', type=int, default=256) # 256
    parser.add_argument('--batch_size_test', type=int, default=256) # 256
    parser.add_argument('--numWorkersP', type=int, default=8) # 4

    # training stages
    parser.add_argument('--class_switch', type=int, default=0)
    parser.add_argument('--num_epochs_class', type=int, default=10)
    parser.add_argument('--num_epochs_regr', type=int, default=50)

    # base learning rate
    parser.add_argument('--base_lr', type=float, default=0.0012)
    
    # models: expects pairs (name, sizeFeatures)
    parser.add_argument(
        '--models', nargs='+', default=['resnet18'],
        help='List of model names to use. Feature sizes are automatically assigned. '
            'Example: --models resnet18 resnet50 resnext101_32x8d'
    )
    parser.add_argument('--decoder', default='plain', type=str, help='Use plain/gated/mmoe decoder')
    parser.add_argument('--num_experts', type=int, default=64)
    
    parser.add_argument(
        '--trainModes', nargs='+',
        default=['imagenet'], #['imagenet', 'scratch']
        help='Training modes to use, e.g. --trainModes imagenet scratch'
    )

    args = parser.parse_args()

    args_dict = vars(args)

    # pretty print all arguments
    print("\n========== Parsed Configuration ==========")
    for key, value in args_dict.items():
        print(f"{key:<20}: {value}")
    print("==========================================")
    
    main(args)