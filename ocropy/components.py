import iulib,ocropus

### For the most part, these just wrap the corresponding functions
### in ocropus.i.  But you should use the ocropy versions because
### they will get hooks for pure Python implementations of these
### interfaces.

def make_ICleanupGray(name):
    return ocropus.make_ICleanupGray(name)
def make_ICleanupBinary(name):
    return ocropus.make_ICleanupBinary(name)
def make_ITextImageClassification(name):
    return ocropus.make_ITextImageClassification(name)
def make_IRecognizeLine(name):
    return ocropus.make_IRecognizeLine(name)
def make_IBinarize(name):
    return ocropus.make_IBinarize(name)
def make_ISegmentPage(name):
    return ocropus.make_ISegmentPage(name)
def make_ISegmentLine(name):
    return ocropus.make_ISegmentLine(name)
def make_IExtractor(name):
    return ocropus.make_IModel(name)
def make_IModel(name):
    return ocropus.make_IModel(name)
def make_IComponent(name):
    return ocropus.make_IComponent(name)
def make_IDataset(name):
    return ocropus.make_IDataset(name)
def make_IExtDataset(name):
    return ocropus.make_IExtDataset(name)
def make_IFeatureMap(name):
    return ocropus.make_IFeatureMap(name)
def make_IGrouper(name):
    return ocropus.make_IGrouper(name)
def make_IDistComp(name):
    return ocropus.make_IDistComp(name)

def load_ICleanupGray(file):
    return ocropus.load_ICleanupGray(file)
def load_ICleanupBinary(file):
    return ocropus.load_ICleanupBinary(file)
def load_ITextImageClassification(file):
    return ocropus.load_ITextImageClassification(file)
def load_IRecognizeLine(file):
    return ocropus.load_IRecognizeLine(file)
def load_IBinarize(file):
    return ocropus.load_IBinarize(file)
def load_ISegmentPage(file):
    return ocropus.load_ISegmentPage(file)
def load_ISegmentLine(file):
    return ocropus.load_ISegmentLine(file)
def load_IExtractor(file):
    return ocropus.load_IExtractor(file)
def load_IModel(file):
    return ocropus.load_IModel(file)
def load_IComponent(file):
    return ocropus.load_IComponent(file)
def load_IDataset(file):
    return ocropus.load_IDataset(file)
def load_IFeatureMap(file):
    return ocropus.load_IFeatureMap(file)
def load_IGrouper(file):
    return ocropus.load_IGrouper(file)
def load_linerec(file):
    return ocropus.load_linerec(file)

def save_component(file,component):
    ocropus.save_component(file,component)
