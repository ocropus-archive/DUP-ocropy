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
        """Train with integer classes.  This is the methods implemented by the C version of IModel."""
        assert type(c)==int
        raise Exception("unimplemented")
    def outputs(self,ov,image):
        """Train with integer classes.  This is the methods implemented by the C version of IModel."""
        raise Exception("unimplemented")
    def train(self,dataset):
        """Train with integer datasets.  This is the methods implemented by the C version of IModel."""
        raise Exception("unimplemented")
    def cadd(self,image,c):
        """Add a training sample.  The class should be a string."""
        assert type(c)==str
        raise Exception("unimplemented")
    def coutputs(self,image):
        """Compute the outputs for the given input image.  The result is
        a list of (cls,probability) tuples. (NB: that's probability, not cost.)"""
        raise Exception("unimplemented")
    def updateModel(self):
        """Update the model after adding some training scamples with
        add or cadd."""
        raise Exception("unimplemented")
    def ctrain(self,dataset):
        """Train on a dataset."""
        raise Exception("unimplemented")
