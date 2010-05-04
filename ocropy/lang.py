import sys,os,re,glob,math,glob,signal
import iulib,ocropus
from pylab import amax,zeros

### character properties

def size_category(c):
    if len(c)>1: raise "isolated characters only"
    if c in "acemnorsuvwxyz": return "x"
    if c in "ABCDEFGHIJKLMNOPQRSTUVWXYZbdfhklt!?": return "k"
    if c in "gpqy": return "y"
    if c in ".,": return "."
    if c in """'"`""": return "'"
    return None

### commonly confused characters in OCR

ocr_confusions_list = [
    ["c","C"],
    ["l","1","I","|","/"],
    ["o","O","0"],
    ["s","S"],
    ["u","U"],
    ["v","V"],
    ["w","W"],
    ["x","X"],
    ["z","Z"],
    [",","'",".","`"],
]

ocr_confusions = {}

for e in ocr_confusions_list:
    for i in range(len(e)):
        ocr_confusions[e[i]] = e

