import iulib,ocropus

### For the most part, these just wrap the corresponding functions
### in ocropus.i.  But you should use the ocropy versions because
### they will get hooks for pure Python implementations of these
### interfaces.

component_registry = {}

def register_constructor(name,f):
    # FIXME check that components have name() etc.
    component_registry[name] = f

def get_components(kind=None):
    # FIXME add Python component registry here
    componentlist = ocropus.ComponentList()
    select = 0
    result = []
    for i in range(componentlist.length()):
        n = componentlist.name(i)
        k = componentlist.kind(i)
        if kind is not None and kind!=k: continue
        result.append(n)
    return result

def make_ICleanupGray(name):
    if component_registry.has_key(name):
        result = component_registry[name]()
    else:
        result = ocropus.make_ICleanupGray(name)
    assert result.interface() == "ICleanupGray"
    return result
def make_ICleanupBinary(name):
    if component_registry.has_key(name):
        result = component_registry[name]()
    else:
        result = ocropus.make_ICleanupBinary(name)
    assert result.interface() == "ICleanupBinary"
    return result
def make_ITextImageClassification(name):
    if component_registry.has_key(name):
        result = component_registry[name]()
    else:
        result = ocropus.make_ITextImageClassification(name)
    assert result.interface() == "ITextImageClassification"
    return result
def make_IRecognizeLine(name):
    if component_registry.has_key(name):
        result = component_registry[name]()
    else:
        result = ocropus.make_IRecognizeLine(name)
    assert result.interface() == "IRecognizeLine"
    return result
def make_IBinarize(name):
    if component_registry.has_key(name):
        result = component_registry[name]()
    else:
        result = ocropus.make_IBinarize(name)
    assert result.interface() == "IBinarize"
    return result
def make_ISegmentPage(name):
    if component_registry.has_key(name):
        result = component_registry[name]()
    else:
        result = ocropus.make_ISegmentPage(name)
    assert result.interface() == "ISegmentPage"
    return result
def make_ISegmentLine(name):
    if component_registry.has_key(name):
        result = component_registry[name]()
    else:
        result = ocropus.make_ISegmentLine(name)
    assert result.interface() == "ISegmentLine"
    return result
def make_IExtractor(name):
    if component_registry.has_key(name):
        result = component_registry[name]()
    else:
        result = ocropus.make_IExtractor(name)
    assert result.interface() == "IExtractor"
    return result
def make_IModel(name):
    if component_registry.has_key(name):
        result = component_registry[name]()
    else:
        result = ocropus.make_IModel(name)
    assert result.interface() == "IModel"
    return result
def make_IComponent(name):
    if component_registry.has_key(name):
        result = component_registry[name]()
    else:
        result = ocropus.make_IComponent(name)
    assert result.interface() == "IComponent"
    return result
def make_IDataset(name):
    if component_registry.has_key(name):
        result = component_registry[name]()
    else:
        result = ocropus.make_IDataset(name)
    assert result.interface() == "IDataset"
    return result
def make_IExtDataset(name):
    if component_registry.has_key(name):
        result = component_registry[name]()
    else:
        result = ocropus.make_IExtDataset(name)
    assert result.interface() == "IExtDataset"
    return result
def make_IFeatureMap(name):
    if component_registry.has_key(name):
        result = component_registry[name]()
    else:
        result = ocropus.make_IFeatureMap(name)
    assert result.interface() == "IFeatureMap"
    return result
def make_IGrouper(name):
    if component_registry.has_key(name):
        result = component_registry[name]()
    else:
        result = ocropus.make_IGrouper(name)
    assert result.interface() == "IGrouper"
    return result
def make_IDistComp(name):
    if component_registry.has_key(name):
        result = component_registry[name]()
    else:
        result = ocropus.make_IDistComp(name)
    assert result.interface() == "IDistComp"
    return result

### FIXME still need to figure out how to 
### handle loading/saving Python components

def load_ICleanupGray(file):
    result = ocropus.load_ICleanupGray(file)
    return result
def load_ICleanupBinary(file):
    result = ocropus.load_ICleanupBinary(file)
    return result
def load_ITextImageClassification(file):
    result = ocropus.load_ITextImageClassification(file)
    return result
def load_IRecognizeLine(file):
    result = ocropus.load_IRecognizeLine(file)
    return result
def load_IBinarize(file):
    result = ocropus.load_IBinarize(file)
    return result
def load_ISegmentPage(file):
    result = ocropus.load_ISegmentPage(file)
    return result
def load_ISegmentLine(file):
    result = ocropus.load_ISegmentLine(file)
    return result
def load_IExtractor(file):
    result = ocropus.load_IExtractor(file)
    return result
def load_IModel(file):
    result = ocropus.load_IModel(file)
    return result
def load_IComponent(file):
    result = ocropus.load_IComponent(file)
    return result
def load_IDataset(file):
    result = ocropus.load_IDataset(file)
    return result
def load_IFeatureMap(file):
    result = ocropus.load_IFeatureMap(file)
    return result
def load_IGrouper(file):
    result = ocropus.load_IGrouper(file)
    return result
def load_linerec(file):
    result = ocropus.load_linerec(file)
    return result

def save_component(file,component):
    ocropus.save_component(file,component)
