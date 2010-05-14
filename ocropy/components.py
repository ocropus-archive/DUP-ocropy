import iulib,ocropus,sys,re
from simplerec import CmodelLineRecognizer

### For the most part, these just wrap the corresponding functions
### in ocropus.i.  But you should use the ocropy versions because
### they will get hooks for pure Python implementations of these
### interfaces.


def make_component(spec,interface):
    """Instantiate an OCRopus component.  This takes care of both the C++ and
    Python components.  It also allows initialization parameters to be passed
    to components.  There are three basic forms.

    SomePackage.SomeClass(x,y,z) calls a Python constructor with the given arguments.

    SomePackage.SomeClass:param=value:param=value instantiates a Python class with
    no arguments, then sets parameters to values using the pset method.

    SomeClass:param=value:param=value instantiates a C++ class with no arguments,
    then sets parameters to values using the pset method.
    """
    names = spec.split(":")
    name = names[0]
    # components of the form a.b(c) are assumet to be constructors
    # and evaluated directly
    if re.match(r'^([a-zA-Z0-9_.]+)\(',name):
        match = re.search(r'^([a-zA-Z0-9_.]+)\.([^().]+)\(',name)
        assert match,"bad Python component name"
        module = match.group(1)
        exec "import %s; result=%s"%(module,name)
        return result
    # components of the form a.b:c=d are assumed to be constructors
    # and parameters are set via pset
    elif "." in name:
        match = re.search(r'^([a-zA-Z0-9_.]+)\.([^().]+)$',name)
        assert match,"bad Python component name"
        module = match.group(1)
        element = match.group(2)
        exec "import %s; constructor=%s.%s"%(module,module,element)
        result = constructor()
    # components of the form name:c=d (no dot in the component name)
    # are assumed to be C++ components; the parameters are set
    # via pset
    else:
        exec "constructor = ocropus.make_%s"%interface
        result = constructor(name)
    for param in names[1:]:
        k,v = param.split("=",1)
        try: v = float(v)
        except ValueError: pass
        result.pset(k,v)
    return result

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
    """Loads a line recognizer.  This handles a bunch of special cases
    due to the way OCRopus has evolved.  In the long term, .pymodel is the
    preferred format.

    For files ending in .pymodel, just unpickles the contents of the file.

    For files ending in .cmodel, loads the character model using load_IModel
    (it has to be a C++ character classifier), and then instantiates a
    CmodelLineRecognizer with the cmodel as an argument.  Additional parameters
    can be passed as in my.cmodel:best=5.  The line recognizer used can be
    overridden as in my.cmodel:class=MyLineRecognizer:best=17.

    For anything else, uses native load_linerec (which has its own special cases)."""

    if ".pymodel" in file:
        with open(file,"rb") as stream:
            result = cPickle.load(stream)
        return result

    elif ".cmodel" in file:
        names = file.split(":")
        cmodel = load_IModel(names[0])
        options = {}
        for param in names[1:]:
            k,v = param.split("=",1)
            try: v = int(v)
            except ValueError:
                try: v = float(v)
                except ValueError: pass
            options[k] = v
        # print options
        constructor = CmodelLineRecognizer
        if options.has_key("class"):
            constructor = eval(options["class"])
            del options["class"]
        result = constructor(cmodel=cmodel,**options)
        # print "# cmodel",cmodel,"result",result
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
