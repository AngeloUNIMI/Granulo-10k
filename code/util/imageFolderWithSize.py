import os
import torch
from torchvision import datasets


class ImageFolderWithSize(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    def __init__(self, dirDbOrig, image_pathP,
                 transformP,
                 misure, misureNorm,
                 panNumbers, infoSpessore,
                 meanNorm, stdNorm):
        super(ImageFolderWithSize, self).__init__(root=image_pathP, transform=transformP)
        self.data = datasets.ImageFolder(image_pathP, transformP)
        self.misure = misure
        self.misureNorm = misureNorm
        self.panNumbers = panNumbers
        self.infoSpessore = infoSpessore

        self.meanNorm = meanNorm
        self.stdNorm = stdNorm

        self.dirFilesB = dirDbOrig + '/datastore_B/'
        self.dirmasksA = dirDbOrig + '/datastore_masks_A/'
        self.dirmasksB = dirDbOrig + '/datastore_masks_B/'

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):

        path = self.imgs[index][0]

        imA = self.loader(path)

        dir, filename = os.path.split(path)
        C = filename.split('_')
        id = int(C[0])
        indexL = self.panNumbers.index(id)

        classV = self.misure[indexL]
        classV_norm = self.misureNorm[indexL]

        # check for spessore
        rootFileName = C[0] + '_' + C[1]

        try:
            # dummy = self.infoSpessore.index(rootFileName)
            # goodForThickness = True
            weightVector = torch.tensor([1, 0, 1])
            # height visible, width not visible, thickness visible
        except:
            # goodForThickness = False
            weightVector = torch.tensor([1, 1, 0])
            # height visible, width visible, thickness not visible

        baseDir, dirPanel = os.path.split(dir)
        fileB = os.path.join(self.dirFilesB, dirPanel, rootFileName + '_B.jpg')
        imB = self.loader(fileB)

        fileMaskA = os.path.join(self.dirmasksA, dirPanel, rootFileName + '_A.png')
        fileMaskB = os.path.join(self.dirmasksB, dirPanel, rootFileName + '_B.png')
        maskA = self.loader(fileMaskA)
        maskB = self.loader(fileMaskB)

        # same transform on two images
        imA_transform, imB_transform, maskA_temp, maskB_temp = self.transform(imA, imB, maskA, maskB)
        # cancel normalization for masks
        maskA_transform = (maskA_temp * self.stdNorm) + self.meanNorm
        maskB_transform = (maskB_temp * self.stdNorm) + self.meanNorm

        return (imA_transform, imB_transform, maskA_transform, maskB_transform,
                id-1, (path), (torch.tensor(classV)), (torch.tensor(classV_norm)), weightVector)
