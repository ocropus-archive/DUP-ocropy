import sys,os,re,glob,math,glob,signal
import iulib,ocropus
from scipy.ndimage import interpolation

class IsolatedCharacterClassifier:
    """A simple isolated character classifier using OCRopus's neural
    network (or another classifier).  This performs resizing and
    multi-character encoding if necessary.  It's kind of obsoleted by
    the new IModel code, which includes feature extraction."""
    def __init__(self,kind="mappedmlp",r=30,flip=1):
        self.r = r
        self.init(kind)
        self.v = iulib.floatarray()
	self.flip = flip
    def init(self,kind):
        self.model = ocropus.make_IModel(kind)
    def load(self,file):
        self.model = ocropus.load_IModel(file)
    def save(self,file):
        ocropus.save_component(file,self.model)
    def classify(self,image):
        """Takes an image as an input and returns a string
        (possibly a multi-characters classification)."""
	if self.flip: image = image.T[:,::-1]
        image = center_maxsize(image,self.r)
        iulib.narray_of_numpy(self.v,image)
        result = ocropus.OutputVector()
        self.model.outputs(result,self.v)
        outputs = []
        for i in range(result.nkeys()):
            v = result.value(i)
            cost = -math.log(max(1e-6,v))
            k = result.key(i)
            try:
                cls = multi_chr(k)
            except:
                print "bad multi_chr [%d] %x %g"%(i,k,v)
                self.model.command("debug_map")
                continue
            outputs.append((cost,cls))
        outputs.sort()
        outputs = [(o[1],o[0]) for o in outputs]
        return outputs
    def train(self,image,cls):
        """Add a training example to the model.  The cls argument
        should be a string that's either one unicode character or
        three ASCII character or less."""
	if self.flip: image = image.T[:,::-1]
        if type(cls)==str: cls = multi_ord(cls)
        assert amin(image)>=0 and amax(image)<=1
        # pattern = center_maxsize(image,self.r)
        # iulib.narray_of_numpy(self.v,pattern.T[:,::-1])
        iulib.narray_of_numpy(self.v,image)
        # self.v.resize(self.v.length())
        self.model.add(self.v,cls)
    def updateModel(self):
        self.model.updateModel()

