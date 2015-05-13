# -*- coding: utf-8 -*-

################################################################
### Ligatures-related data.
################################################################

import re
from pylab import uint32

### These aren't formal ligatures, they are character pairs
### that are frequently touching in Latin script documents.

common_ligature_table = """
000 00 La Th ac ai ak al all am an ar as be bo ca ch co ct
di dr ec ed ee es ff ffi fl fr ft gh gi gr gu hi il
in ir is ki li ll ma mi mm ni oc oo pe po re ri rin
rm rn ro r rs rt ru rv ry se sl so ss st ta te th ti to tr
ts tt tu ul um un ur vi wi wn
a. c. e. m. n. t. z. A. C. E. K. L. M. N. R.
a, c, e, m, n, t, z, A, C, E, K, L, M, N, R,
a- b- e- d- g- m- n- o- p- u-
"B "D "F "H "K "L "P "R "T "W "Z "b "h "l
'B 'D 'F 'H 'K 'L 'P 'R 'T 'W 'Z 'b 'h 'l
d" f" l" 
""" 

""" " """

common_chars = [ u'„', u'“' ]

def common_ligatures(s):
    if len(s)>=2 and s[:2] in common_ligature_table:
        yield s[:2]
    if len(s)>=3 and s[:3] in common_ligature_table:
        yield s[:3]

class LigatureTable:
    def __init__(self):
        self.lig2code = {}
        self.code2lig = {}
        # add codes for common control characters so that they
        # show up a little more nicely in displays (we're using
        # non-standard greek letters here to avoid conflict with
        # greek alphabet
        self.add("",0)
        self.add("<RHO>",2)
        self.add("<SIG>",3)
        self.add("<PHI>",4)
        # ensure that ASCII is always present
        # note that "_" and "~" always have a special meaning
        # but are treated like other ASCII characters
        for i in range(32,1024):
            self.add(unichr(i),i)
        for c in common_chars:
            self.add(c,ord(c))
    def add(self,name,code,override=1):
        assert isinstance(name, unicode) or not re.search(r'[\x80-\xff]',name)
        if not override and self.lig2code.get(name) is not None:
            raise Exception("character '%s' (%d) already in ligature table"%(name,self.ord(name)))
        self.lig2code[name] = code
        self.code2lig[code] = unicode(name)
    def ord(self,name):
        if name=="": return 0 # epsilon
        result = self.lig2code.get(name,-1)
        # fall back for non-ligature unicode characters
        if result<0 and len(name)==1: return ord(name)
        return result
    def chr(self,code):
        result = self.code2lig.get(code,None)
        if code<0: return u"~"
        if code<0x10000 and result is None: return unichr(code)
        return result
    def writeText(self,name):
        with open(name,"w") as stream:
            for name,code in self.lig2code.items():
                stream.write("%s %d\n"%(name,uint32(code)))
                    
lig = LigatureTable()
ligcode = 0x200000

# special ligatures

for l in ["~~","~~~","~~~~"]:
    lig.add(l,ligcode)
    ligcode += 1

# add ASCII ligatures (DO NOT CHANGE THIS)

ucase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
lcase = "abcdefghijklmnopqrstuvwxyz"
digits = "0123456789"
quotes = "".join(["'",'"'])
punctuation = ".,:;?!"  # "-" is missing here, added below
common = ucase+lcase+digits+quotes+punctuation

for c1 in common:
    for c2 in common:
        lig.add(c1+c2,ligcode)
        ligcode += 1

# add triple ligatures (DO NOT CHANGE THIS)

doubles = "oo OO 00 ff mm".split()

for c1 in doubles:
    for c2 in common:
        lig.add(c1+c2,ligcode)
        ligcode += 1

# add German and additional German ligatures (DO NOT CHANGE THIS)
        
german = u"ÄÖÜäöüß"

for c in german:
    lig.add(c,ord(c))

for c in german:
    for c2 in common:
        lig.add(c+c2,ligcode)
        ligcode += 1

# add ligatures involving a "-"

for c1 in common:
    lig.add(c1+"-",ligcode)
    lig.add("-"+c1,ligcode)
    ligcode += 1

# ADD ADDITIONAL CHARACTERS AND LIGATURES BELOW
