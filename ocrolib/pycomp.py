################################################################
### Baseclass for new, pure Python components
### (Separate module to avoid circular imports.)
################################################################

class PyComponent:
    """Defines common methods similar to CommonComponent, but for Python
    classes. Use of this base class is optional."""
    def init(self):
        pass
    def name(self):
        return "%s"%self
    def description(self):
        return "%s"%self
    def set(self,**kw):
        kw = set_params(self,kw)
        assert kw=={},"extra params to %s: %s"%(self,kw)
    def pset(self,key,value):
        if hasattr(self,key):
            self.__dict__[key] = value
    def pget(self,key):
        return getattr(self,key)
    def pgetf(self,key):
        return float(getattr(self,key))
    
