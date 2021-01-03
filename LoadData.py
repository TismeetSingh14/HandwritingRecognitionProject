import os
import random
import cv2
import numpy as np

class SampleImages:
    def __init__(self,trueText, filePath):
        self.trueText = trueText
        self.filePath = filePath

class BatchImages:
    def __init__(self,trueText, image):
        self.image = np.stack(image,axis = 0)
        self.trueText = trueText

class Loader:
    def __init__(self, filePath, batchSize, imgSize, maxTextLen):

        assert filePath[-1] == '/'

        self.dataAugmentation = False
        self.currIdx = 0
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.samples = []

        f = open(filePath + 'words.txt')
        chars = set()
        badImages = []
        # badImagesReference = ['a01-117-05-02.png', 'r06-022-03-05.png']

        for line in f:
            if not line or line[0] == '#':
                continue

            lineSplit = line.strip().split(' ')

            assert len(lineSplit) >= 9

            fileNameSplit = lineSplit[0].split('-')
            fileName = filePath + 'words/' + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + lineSplit[0] + '.png'

            trueText = self.labelTruncate(' '.join(lineSplit[8:]), maxTextLen)

            chars = chars.union(set(list(trueText)))

            if not os.path.getsize(fileName):
                badImages.append(lineSplit[0] + '.png')
                continue

            self.samples.append(SampleImages(trueText, fileName))
        
        # if set(badImages) != set(badImagesReference)
        #     print("Warning, damaged images found:", badImages)
        #     print("Damaged images expected:", badImagesReference)
        
        indexSplit = int(0.95 * len(self.samples))
        self.trainSampleImages = samples[:indexSplit] 
        self.validationSampleImages = samples[indexSplit:]

        self.trainWords = [x.trueText for x in self.trainSampleImages]
        self.validationWords = [x.trueText for x in self.validationSampleImages]

        self.samplesPerEpoch = 18000

        self.trainSet()
        self.charList = sorted(list(chars))

        def labelTruncate(self, text, maxTextLen):
            cost = 0

            for i in range(len(text)):

                if i != 0 and text[i] == text[i-1]:
                    cost += 2

                else:
                    cost += 1

                if cost > maxTextLen:
                    return text[:i]
            return text   

        def trainSet(self):
            self.dataAugmentation = False
            self.currIdx = 0
            self.samples = self.validationSampleImages

        def validationSet(self):
            self.dataAugmentation = False
            self.currIdx = 0
            self.samples = self.validationSampleImages

        def getIterator(self):
            return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize) 

        def hasNext(self):
            return self.currIdx + self.batchSize <= len(self.samples)

        def nextBatch(self):
            rangeOfBatch = range(self.currIdx, self.currIdx + self.batchSize)
            trueText = [self.samples[i].trueText for i in rangeOfBatch]

            imgs = [
                preprocess(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imgSize, self.dataAugmentation) for i in rangeOfBatch]

            self.currIdx += self.batchSize
            return BatchImages(trueText, imgs)   