print "   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
print "   !!! importing old ocropy library"
print "   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

__all__ = ["dbtables","ocrobook","fgen"]
from ocropus import *
from iulib import *
from narray import *
from components import *
from improc import *
from lang import *
from utils import *
from hocr import *

# the following code is obsolete because it uses the internal APIs

__all__ += ["oldsimpleti"]
from oldquickcheck import *
from oldalignment import *
from oldlinerec import OldSimpleLineRecognizer
# from oldlinerec2 import OldFullLineRecognizer
