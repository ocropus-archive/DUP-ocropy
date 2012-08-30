################################################################
### A simple set of functions for embedding native C code
### inside Python.
################################################################

import os,sys,re,string,hashlib
import numpy
from  numpy.ctypeslib import ndpointer
from ctypes import c_int,c_float,c_double,c_byte
import ctypes
import timeit
from pylab import prod

import os,time,errno,contextlib

@contextlib.contextmanager
def lockfile(fname,delay=0.5):
    while 1:
        try: 
            fd = os.open(fname,os.O_RDWR|os.O_CREAT|os.O_EXCL)
        except OSError as e: 
            if e.errno!=errno.EEXIST: raise
            time.sleep(delay)
            continue
        else:
            break
    try:
        yield fd
    finally:
        os.close(fd)
        os.unlink(fname)

I = c_int
F = c_float
D = c_double
B = c_byte

for d in range(1,4):
    for T,t in [("I","int32"),("F","float32"),("D","float64"),("B","int8"),("U","uint8")]:
        exec "A%d%s = ndpointer(dtype='%s',ndim=%d,flags='CONTIGUOUS,ALIGNED')"%(d,T,t,d)

class CompileError(Exception):
    pass

def compile_and_find(c_string,prefix=".pynative",opt="-g -O4",libs="-lm",
                     options="-shared -fopenmp -std=c99 -fPIC",verbose=0):
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    m = hashlib.md5()
    m.update(c_string)
    base = m.hexdigest()
    if verbose: print "hash",base,"for",c_string[:20],"..."
    with lockfile(os.path.join(prefix,base+".lock")):
        so = os.path.join(prefix,base+".so")
        if os.path.exists(so):
            if verbose: print "returning existing",so
            return so
        source = os.path.join(prefix,base+".c")
        with open(source,"w") as stream:
            stream.write(c_string)
        cmd = "gcc "+opt+" "+libs+" "+options+" "+source+" -o "+so
        if verbose: print "#",cmd
        if os.system(cmd)!=0:
            if verbose: print "compilation failed"
            raise CompileError()
        return so

def compile_and_load(c_string,**keys):
    path = compile_and_find(c_string,**keys)
    return ctypes.CDLL(path)
