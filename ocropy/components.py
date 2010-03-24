import iulib,ocropus,sys,re

### For the most part, these just wrap the corresponding functions
### in ocropus.i.  But you should use the ocropy versions because
### they will get hooks for pure Python implementations of these
### interfaces.


def make_component(name,interface):
    if "." in name:
        match = re.search(r'^([a-zA-Z0-9_.]+)\.([^.]+)$',name)
        assert match,"bad Python component name"
        module = match.group(1)
        element = match.group(2)
        exec "import %s; constructor=%s.%s"%(module,module,element)
        return constructor()
    else:
        exec "constructor = ocropus.make_%s"%interface
        return constructor(name)

def make_ICleanupGray(name):
    return make_component(name,"ICleanupGray")
def make_ICleanupBinary(name):
    return make_component(name,"ICleanupBinary")
def make_ITextImageClassification(name):
    return make_component(name,"ITextImageClassification")
def make_IRecognizeLine(name):
    return make_component(name,"IRecognizeLine")
def make_IBinarize(name):
    return make_component(name,"IBinarize")
def make_ISegmentPage(name):
    return make_component(name,"ISegmentPage")
def make_ISegmentLine(name):
    return make_component(name,"ISegmentLine")
def make_IExtractor(name):
    return make_component(name,"IExtractor")
def make_IModel(name):
    return make_component(name,"IModel")
def make_IComponent(name):
    return make_component(name,"IComponent")
def make_IDataset(name):
    return make_component(name,"IDataset")
def make_IExtDataset(name):
    return make_component(name,"IExtDataset")
def make_IFeatureMap(name):
    return make_component(name,"IFeatureMap")
def make_IGrouper(name):
    return make_component(name,"IGrouper")
def make_IDistComp(name):
    return make_component(name,"IDistComp")

### FIXME still need to figure out how to 
### handle loading/saving Python components

def load_generic(file,interface):
    if ".pymodel" in file:
        with open(file,"rb") as stream:
            result = cPickle.load(stream)
        return result
    else:
        exec "loader = ocropus.load_%s"%interface
        result = loader(file)
        return result

def load_ICleanupGray(file):
    return load_generic(file,"ICleanupGray")
def load_ICleanupBinary(file):
    return load_generic(file,"ICleanupBinary")
def load_ITextImageClassification(file):
    return load_generic(file,"ITextImageClassification")
def load_IRecognizeLine(file):
    return load_generic(file,"IRecognizeLine")
def load_IBinarize(file):
    return load_generic(file,"IBinarize")
def load_ISegmentPage(file):
    return load_generic(file,"ISegmentPage")
def load_ISegmentLine(file):
    return load_generic(file,"ISegmentLine")
def load_IExtractor(file):
    return load_generic(file,"IExtractor")
def load_IModel(file):
    return load_generic(file,"IModel")
def load_IComponent(file):
    return load_generic(file,"IComponent")
def load_IDataset(file):
    return load_generic(file,"IDataset")
def load_IFeatureMap(file):
    return load_generic(file,"IFeatureMap")
def load_IGrouper(file):
    return load_generic(file,"IGrouper")
def load_linerec(file):
    if ".pymodel" in file:
        with open(file,"rb") as stream:
            result = cPickle.load(stream)
        return result
    else:
        result = ocropus.load_linerec(file)
        return result

def save_component(file,component):
    if ".pymodel" in file:
        with open(file,"wb") as stream:
            cPickle.dump(stream)
    else:
        ocropus.save_component(file,component)
