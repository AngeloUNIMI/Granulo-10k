import os


def getNumOfPanels(dirDbTest):
    count = 0
    for panNumber, (subdir) in enumerate(os.listdir(dirDbTest)):
        count += 1
    return count
