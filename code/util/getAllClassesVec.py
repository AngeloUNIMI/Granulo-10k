import csv
import statistics


def getAllClassesVec(csvFileFull):
    # open csv
    allMisure = list()
    allPanNumbers = list()
    with open(csvFileFull, newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                columnNames = row
            else:
                #print(row)
                panNumber = int(row[0])
                misureVec = list(map(float, row[1:4]))
                allMisure.append(misureVec)
                allPanNumbers.append(panNumber)
                #print(classVec)
                #pause()
            line_count += 1
        print('Processed {} lines.'.format(line_count))
    return allMisure, allPanNumbers, columnNames

def getInfoSpessore(csvInfoSpessore):
    fileNameSpessore = list()
    with open(csvInfoSpessore, newline='', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            fileNameSpessore.append(row[0])
    return fileNameSpessore


def normalizeMeasures(allMisure):
    # init
    allHeights = []
    allWidths = []
    allThickness = []

    # read
    for measure in allMisure:
        allHeights.append(measure[0])
        allWidths.append(measure[1])
        allThickness.append(measure[2])

    statisticsLabel = {
        'meanHeight': statistics.mean(allHeights),
        'stdHeight': statistics.stdev(allHeights),
        'meanWidth': statistics.mean(allWidths),
        'stdWidth': statistics.stdev(allWidths),
        'meanThickness': statistics.mean(allThickness),
        'stdThickness': statistics.stdev(allThickness)
    }

    # write
    allMisureNormalized = list()
    for measure in allMisure:
        heigthNorm = (measure[0] - statisticsLabel['meanHeight']) / statisticsLabel['stdHeight']
        widthNorm = (measure[1] - statisticsLabel['meanWidth']) / statisticsLabel['stdWidth']
        thickNorm = (measure[2] - statisticsLabel['meanThickness']) / statisticsLabel['stdThickness']
        allMisureNormalized.append([heigthNorm, widthNorm, thickNorm])

    return allMisureNormalized, statisticsLabel
