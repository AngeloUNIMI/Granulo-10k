import torch
import numpy as np
import matplotlib.pyplot as plt
from util.normImageTo255 import normImageTo255


def visImage(x, title):
    x2 = torch.squeeze(x, 0)
    x3 = x2.numpy()
    x4 = np.swapaxes(x3, 0, 2)
    x4 = np.swapaxes(x4, 0, 1)
    x5 = normImageTo255(x4)
    plt.axis('off')
    plt.title(title)
    plt.imshow(x5.astype('uint8'))
    plt.show()
