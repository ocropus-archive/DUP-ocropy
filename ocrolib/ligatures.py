# -*- coding: utf-8 -*-

import sys,os,unicodedata,re
from pylab import uint32,uint16,uint64
import ocropus
import openfst

epsilon = openfst.epsilon
assert epsilon==ocropus.L_EPSILON
rho = ocropus.L_RHO
sigma = ocropus.L_SIGMA
phi = ocropus.L_PHI

class LigatureTable:
    def __init__(self):
        self.lig2code = {}
        self.code2lig = {}
        # add codes for common control characters so that they
        # show up a little more nicely in displays (we're using
        # non-standard greek letters here to avoid conflict with
        # greek alphabet
        self.add("ϵ",epsilon)
        self.add("ϱ",rho)
        self.add("ϭ",sigma)
        self.add("ϕ",phi)
        # ensure that ASCII is always present
        # note that "_" and "~" always have a special meaning
        # but are treated like other ASCII characters
        for i in range(32,127):
            self.add(unichr(i),i)
    def add(self,name,code,override=0):
        if not override and self.ord(name)>=0:
            raise Exception("character '%s' (%d) already in ligature table"%(name,self.ord(name)))
        self.lig2code[name] = code
        self.code2lig[code] = name
    def ord(self,name):
        return self.lig2code.get(name,-1)
    def chr(self,code):
        return self.code2lig.get(code)
    def SymbolTable(self,name="ligature_table"):
        result = openfst.SymbolTable(name)
        for name,code in self.lig2code.items():
            if type(name)==unicode:
                name = name.encode("utf-8")
            result.AddSymbol(name,code)
        return result
    def writeText(self,name):
        with open(name,"w") as stream:
            for name,code in self.lig2code.items():
                stream.write("%s %d\n"%(name,uint32(code)))
                    
common_ligatures = """oo OO 00 000 fi ffi ffl st sz ar""".split()
common_ligatures = {}.fromkeys(common_ligatures,1)

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
punctuation = ".,:;?!"
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

# ADD ADDITIONAL CHARACTERS AND LIGATURES BELOW
