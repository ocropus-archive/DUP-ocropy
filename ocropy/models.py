### This file mainly serves to document interfaces that may implemented
### in Python and are commonly used.

class RecognizeLineSample:
    def recognizeLine(self,fst,image):
        pass
    def startTraining(self,type="adaptation"):
        pass
    def finishTraining(self):
        pass
    def addTrainingLine(self,image,transcription):
        pass
    def addTrainingLine(self,segmentation,image,transcription):
        pass
    def align(self,chars,seg,costs,image,transcription):
        pass
    def epoch(self,n):
        pass
    def save(self,file):
        pass
    def load(self,file):
        pass

class ModelSample:
    def add(self,image,c):
        pass
    def outputs(self,ov,image):
        pass
    def train(self,dataset):
        pass
    def cadd(self,image,c):
        pass
    def coutputs(self,ov,image):
        pass
    def ctrain(self,dataset):
        pass
    def updateModel(self):
        pass
